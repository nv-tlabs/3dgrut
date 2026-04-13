#include "renderer_core.h"

#include <anari/anari.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {

// ANARI extension / channel name constants used when configuring the frame
// object and mapping its buffers.
constexpr const char *kCudaFrameExtension = "ANARI_NV_FRAME_BUFFERS_CUDA";
constexpr const char *kChannelColor = "channel.color";
constexpr const char *kChannelColorCUDA = "channel.colorCUDA";
constexpr const char *kChannelDepth = "channel.depth";

// Return the value at the given percentile [0, 1] from an unsorted vector.
// Uses std::nth_element for O(n) average-case performance.  The input is
// taken by value so the caller's data is not reordered.
float percentileValue(std::vector<float> values, float percentile) {
  if (values.empty())
    return 0.f;
  percentile = std::clamp(percentile, 0.f, 1.f);
  const size_t idx = static_cast<size_t>(percentile * static_cast<float>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return values[idx];
}

// Map an ANARIDataType for a colour pixel to its byte width.
// Returns 0 for unrecognised types (treated as an error by callers).
size_t bytesPerPixel(ANARIDataType pixelType) {
  if (pixelType == ANARI_UFIXED8_RGBA_SRGB || pixelType == ANARI_UFIXED8_VEC4)
    return 4;
  if (pixelType == ANARI_FLOAT32_VEC4)
    return 16;
  return 0;
}

// Build an ANARI World that visualises every Gaussian as a unit sphere
// instanced N times with per-instance affine transforms and colours.
//
// Scene structure (ANARI object graph):
//
//   World
//   ├── Instance (transform array)
//   │   ├── Group
//   │   │   └── Surface
//   │   │       ├── Geometry  -- single unit sphere at the origin
//   │   │       └── Material  -- "matte", colour sourced from per-instance array
//   │   ├── transform[]       -- N column-major 4x4 TRS matrices (one per
//   Gaussian) │   └── color[]           -- N RGB colours
//
// Lighting is not included in the world returned by this function; the caller
// (GaussianRendererCore) attaches a persistent directional light separately so
// that it can be updated without a full world rebuild.  Each Gaussian's
// position, orientation, and anisotropic scale (times the global scaleFactor)
// are baked into its instance transform via buildTransform().
anari::World buildScene(anari::Device device, const GaussianData &data, float scaleFactor) {
  uint32_t N = static_cast<uint32_t>(data.positions.size());

  // Prototype geometry: a single unit sphere at the origin.  Every Gaussian
  // will be an instance of this sphere, scaled and placed by its TRS matrix.
  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  vec3 origin = {0.f, 0.f, 0.f};
  anari::setParameterArray1D(device, geometry, "vertex.position", &origin, 1);
  anari::setParameter(device, geometry, "radius", 1.0f);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", "color");
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto group = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, group, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, group);

  // Per-instance TRS transforms and colours, populated from the decoded
  // GaussianData.
  auto xfmArray = anari::newArray1D(device, ANARI_FLOAT32_MAT4, N);
  auto colArray = anari::newArray1D(device, ANARI_FLOAT32_VEC3, N);
  {
    auto *xfms = anari::map<mat4>(device, xfmArray);
    auto *cols = anari::map<vec3>(device, colArray);
    for (uint32_t i = 0; i < N; i++) {
      xfms[i] = buildTransform(data.positions[i], data.quats[i], data.scales[i], scaleFactor);
      cols[i] = data.colors[i];
    }
    anari::unmap(device, xfmArray);
    anari::unmap(device, colArray);
  }

  auto instance = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, instance, "group", group);
  anari::setAndReleaseParameter(device, instance, "transform", xfmArray);
  anari::setAndReleaseParameter(device, instance, "color", colArray);
  anari::commitParameters(device, instance);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "instance", &instance, 1);
  anari::release(device, instance);
  anari::commitParameters(device, world);

  return world;
}

} // namespace

// Release all ANARI objects in reverse creation order, after ensuring any
// outstanding framebuffer mappings have been dropped.
GaussianRendererCore::~GaussianRendererCore() {
  clearFrameMappings();
  if (m_device) {
    if (m_frameObj)
      anari::release(m_device, m_frameObj);
    if (m_rendererObj)
      anari::release(m_device, m_rendererObj);
    if (m_headlightObj)
      anari::release(m_device, m_headlightObj);
    if (m_lightObj)
      anari::release(m_device, m_lightObj);
    if (m_cameraObj)
      anari::release(m_device, m_cameraObj);
    if (m_world)
      anari::release(m_device, m_world);
    anari::release(m_device, m_device);
  }
  if (m_library)
    anariUnloadLibrary(m_library);
}

// One-time initialisation sequence:
//
//  1. Validate inputs and load + filter the PLY.
//  2. Compute scene bounds and camera heuristics so that the default camera
//     looks at the centre of the scene from a reasonable distance.
//  3. Load the requested ANARI library, create a device, and probe for
//     optional CUDA framebuffer support.
//  4. Allocate the core ANARI objects (camera, renderer, frame) and configure
//     the frame's output channels (sRGB colour + float depth).
//  5. Mark everything dirty and flush via applyPendingUpdates(), which builds
//     the world and commits all objects for the first time.
bool GaussianRendererCore::init(const InitOptions &options, std::string *errorMessage) {
  if (m_initialized) {
    if (errorMessage)
      *errorMessage = "Renderer core is already initialized.";
    return false;
  }

  if (options.plyPath.empty()) {
    if (errorMessage)
      *errorMessage = "PLY path is empty.";
    return false;
  }
  if (options.frameSize[0] == 0 || options.frameSize[1] == 0) {
    if (errorMessage)
      *errorMessage = "Frame size must be non-zero.";
    return false;
  }

  m_data = loadPLY(options.plyPath, options.opacityThreshold);
  if (m_data.positions.empty()) {
    if (errorMessage)
      *errorMessage = "No Gaussians survived filtering.";
    return false;
  }

  m_sceneBounds = computeSceneBounds(m_data);
  m_scaleFactor = options.scaleFactor;
  m_useFloat32Color = options.useFloat32Color;
  m_frameSize = options.frameSize;
  m_rendererConfig = options.rendererConfig;
  computeCameraHeuristics();

  // Default camera: look toward +Z from a point offset by focusDistance along
  // -Z, with Y-down convention matching typical 3DGS training data.
  m_camera.eye = {m_focusCenter[0], m_focusCenter[1], m_focusCenter[2] - m_focusDistance};
  m_camera.dir = {0.f, 0.f, 1.f};
  m_camera.up = {0.f, -1.f, 0.f};
  m_camera.aspect = float(m_frameSize[0]) / float(m_frameSize[1]);

  m_library = anariLoadLibrary(options.libraryName.c_str(), statusCallback, this);
  if (!m_library) {
    if (errorMessage)
      *errorMessage = "Failed to load ANARI library '" + options.libraryName + "'.";
    return false;
  }

  m_device = anariNewDevice(m_library, "default");
  if (!m_device) {
    if (errorMessage)
      *errorMessage = "Failed to create ANARI device from library '" + options.libraryName + "'.";
    return false;
  }

  m_supportsCudaFrameBuffers = checkCudaFrameExtension();
  if (!m_supportsCudaFrameBuffers)
    std::fprintf(stderr,
                 "[INFO] ANARI device does not expose %s -- "
                 "CUDA framebuffer path disabled, using host readback.\n",
                 kCudaFrameExtension);

  m_cameraObj = anari::newObject<anari::Camera>(m_device, "perspective");
  m_rendererObj = anari::newObject<anari::Renderer>(m_device, "default");
  m_lightObj = anari::newObject<anari::Light>(m_device, "directional");
  m_headlightObj = anari::newObject<anari::Light>(m_device, "directional");
  m_frameObj = anari::newObject<anari::Frame>(m_device);

  if (!m_cameraObj || !m_rendererObj || !m_lightObj || !m_headlightObj || !m_frameObj) {
    if (errorMessage)
      *errorMessage = "Failed to create ANARI camera/renderer/light/frame objects.";
    return false;
  }

  m_worldDirty = true;
  m_frameSizeDirty = true;
  m_cameraDirty = true;
  m_rendererDirty = true;

  anari::setParameter(m_device, m_frameObj, kChannelColor, m_useFloat32Color ? ANARI_FLOAT32_VEC4 : ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(m_device, m_frameObj, kChannelDepth, ANARI_FLOAT32);
  anari::setParameter(m_device, m_frameObj, "camera", m_cameraObj);
  anari::setParameter(m_device, m_frameObj, "renderer", m_rendererObj);
  anari::commitParameters(m_device, m_frameObj);

  if (!applyPendingUpdates(errorMessage))
    return false;

  m_initialized = true;
  return true;
}

// Flush any dirty parameters, issue a synchronous render, and record the
// reported render duration (queried as a non-blocking ANARI property so it
// does not add latency).
bool GaussianRendererCore::run(std::string *errorMessage) {
  if (!m_initialized) {
    if (errorMessage)
      *errorMessage = "Renderer core is not initialized.";
    return false;
  }

  if (!applyPendingUpdates(errorMessage))
    return false;

  anari::render(m_device, m_frameObj);
  anari::wait(m_device, m_frameObj);

  m_lastDurationSeconds = 0.f;
  anari::getProperty(m_device, m_frameObj, "duration", m_lastDurationSeconds, ANARI_NO_WAIT);
  return true;
}

void GaussianRendererCore::setFrameSize(const uvec2 &size) {
  if (size[0] == 0 || size[1] == 0)
    return;
  m_frameSize = size;
  m_camera.aspect = float(m_frameSize[0]) / float(m_frameSize[1]);
  m_frameSizeDirty = true;
  m_cameraDirty = true;
}

void GaussianRendererCore::setCamera(const CameraState &camera) {
  m_camera = camera;
  m_cameraDirty = true;
}

void GaussianRendererCore::setRendererConfig(const RendererConfig &config) {
  m_rendererConfig = config;
  m_rendererDirty = true;
}

// Unlike the other setters, setScaleFactor flushes immediately because it
// triggers a full world rebuild (every instance transform changes).
bool GaussianRendererCore::setScaleFactor(float scaleFactor, std::string *errorMessage) {
  if (!m_initialized) {
    if (errorMessage)
      *errorMessage = "Renderer core is not initialized.";
    return false;
  }
  m_scaleFactor = scaleFactor;
  m_worldDirty = true;
  return applyPendingUpdates(errorMessage);
}

// Map the host-accessible colour channel.  Returns pixel data as packed
// uint32_t (RGBA sRGB).  Auto-unmaps any previous mapping to prevent leaks.
anari::MappedFrameData<uint32_t> GaussianRendererCore::mapColorHost(std::string *errorMessage) {
  anari::MappedFrameData<uint32_t> mapped{};
  if (!m_initialized) {
    if (errorMessage)
      *errorMessage = "Renderer core is not initialized.";
    return mapped;
  }

  if (m_hostMapped)
    unmapColorHost();

  mapped = anari::map<uint32_t>(m_device, m_frameObj, kChannelColor);
  if (!mapped.data) {
    if (errorMessage)
      *errorMessage = "Failed to map host color framebuffer.";
    return mapped;
  }

  m_hostMapped = true;
  return mapped;
}

void GaussianRendererCore::unmapColorHost() {
  if (!m_hostMapped)
    return;
  anari::unmap(m_device, m_frameObj, kChannelColor);
  m_hostMapped = false;
}

// Map the CUDA device-pointer colour channel (via the
// ANARI_NV_FRAME_BUFFERS_CUDA extension).  The returned void* points to GPU
// memory; callers can use it with cudaMemcpy or pass it directly to a
// downstream CUDA kernel.
anari::MappedFrameData<void> GaussianRendererCore::mapColorCUDA(std::string *errorMessage) {
  anari::MappedFrameData<void> mapped{};
  if (!m_initialized) {
    if (errorMessage)
      *errorMessage = "Renderer core is not initialized.";
    return mapped;
  }
  if (!m_supportsCudaFrameBuffers) {
    if (errorMessage)
      *errorMessage = "CUDA framebuffer not supported by this ANARI device.";
    return mapped;
  }

  if (m_cudaMapped)
    unmapColorCUDA();

  mapped = anari::map<void>(m_device, m_frameObj, kChannelColorCUDA);
  if (!mapped.data) {
    if (errorMessage)
      *errorMessage = "Failed to map CUDA color framebuffer.";
    return mapped;
  }

  m_cudaMapped = true;
  return mapped;
}

void GaussianRendererCore::unmapColorCUDA() {
  if (!m_cudaMapped)
    return;
  anari::unmap(m_device, m_frameObj, kChannelColorCUDA);
  m_cudaMapped = false;
}

MappedFrameInfo GaussianRendererCore::mapColorCUDAInfo(std::string *errorMessage) {
  MappedFrameInfo info{};
  auto mapped = mapColorCUDA(errorMessage);
  if (!mapped.data)
    return info;
  info.data = mapped.data;
  info.width = mapped.width;
  info.height = mapped.height;
  info.bytesPerPixel = static_cast<uint32_t>(bytesPerPixel(mapped.pixelType));
  info.isFloat = (mapped.pixelType == ANARI_FLOAT32_VEC4);
  return info;
}

#ifdef GRUT_HAS_CUDA
// High-level helper: map the CUDA colour buffer, 2D-copy it into a
// caller-supplied pitched device buffer, and unmap -- all in one call.
// Handles both synchronous (stream == nullptr) and asynchronous paths.
// The 2D copy accommodates destination pitch that may differ from the source
// row stride (e.g. textures with alignment padding).
bool GaussianRendererCore::copyColorCUDAToDevice(void *dstPtr, size_t dstPitchBytes, cudaStream_t stream, std::string *errorMessage) {
  if (!dstPtr) {
    if (errorMessage)
      *errorMessage = "Destination device pointer is null.";
    return false;
  }

  auto mapped = mapColorCUDA(errorMessage);
  if (!mapped.data)
    return false;

  const size_t rowBytes = mapped.width * bytesPerPixel(mapped.pixelType);
  if (rowBytes == 0) {
    if (errorMessage)
      *errorMessage = "Unsupported color pixel type for CUDA copy.";
    unmapColorCUDA();
    return false;
  }
  if (dstPitchBytes < rowBytes) {
    if (errorMessage)
      *errorMessage = "Destination pitch is smaller than source row bytes.";
    unmapColorCUDA();
    return false;
  }

  cudaError_t err = cudaSuccess;
  if (stream) {
    err = cudaMemcpy2DAsync(dstPtr, dstPitchBytes, mapped.data, rowBytes, rowBytes, mapped.height, cudaMemcpyDeviceToDevice, stream);
  } else {
    err = cudaMemcpy2D(dstPtr, dstPitchBytes, mapped.data, rowBytes, rowBytes, mapped.height, cudaMemcpyDeviceToDevice);
  }
  unmapColorCUDA();

  if (err != cudaSuccess) {
    if (errorMessage)
      *errorMessage = std::string("cudaMemcpy2D failed: ") + cudaGetErrorString(err);
    return false;
  }

  return true;
}
#else
bool GaussianRendererCore::copyColorCUDAToDevice(void *, size_t, cudaStream_t, std::string *errorMessage) {
  if (errorMessage)
    *errorMessage = "CUDA support not compiled (GRUT_HAS_CUDA not defined).";
  return false;
}
#endif

// Commit any ANARI objects whose parameters have changed since the last
// render.  Each dirty flag gates a specific object commit so we only pay for
// state that actually changed.  The world rebuild is the most expensive
// (re-creates all instance transforms), while camera and renderer commits are
// lightweight parameter updates.
bool GaussianRendererCore::applyPendingUpdates(std::string *errorMessage) {
  if (!m_initialized && !m_device)
    return true;

  if (m_worldDirty && !rebuildWorld(errorMessage))
    return false;

  if (m_frameSizeDirty) {
    anari::setParameter(m_device, m_frameObj, "size", m_frameSize);
    anari::commitParameters(m_device, m_frameObj);
    m_frameSizeDirty = false;
  }

  if (m_cameraDirty) {
    anari::setParameter(m_device, m_cameraObj, "position", m_camera.eye);
    anari::setParameter(m_device, m_cameraObj, "direction", m_camera.dir);
    anari::setParameter(m_device, m_cameraObj, "up", m_camera.up);
    anari::setParameter(m_device, m_cameraObj, "aspect", m_camera.aspect);
    anari::setParameter(m_device, m_cameraObj, "fovy", m_camera.fovy);
    anari::commitParameters(m_device, m_cameraObj);

    if (m_headlightObj && m_world) {
      const float irr = m_rendererConfig.headlightEnabled ? m_rendererConfig.headlightIntensity : 0.f;
      anari::setParameter(m_device, m_headlightObj, "direction", m_camera.dir);
      anari::setParameter(m_device, m_headlightObj, "irradiance", irr);
      anari::commitParameters(m_device, m_headlightObj);
      anari::commitParameters(m_device, m_world);
    }

    m_cameraDirty = false;
  }

  if (m_rendererDirty) {
    anari::setParameter(m_device, m_rendererObj, "background", m_rendererConfig.bgColor);
    anari::setParameter(m_device, m_rendererObj, "ambientRadiance", m_rendererConfig.ambientRadiance);
    anari::setParameter(m_device, m_rendererObj, "pixelSamples", m_rendererConfig.spp);
    anari::commitParameters(m_device, m_rendererObj);

    if (m_world) {
      if (m_lightObj) {
        anari::setParameter(m_device, m_lightObj, "direction", m_rendererConfig.lightDirection);
        anari::setParameter(m_device, m_lightObj, "irradiance", m_rendererConfig.lightIntensity);
        anari::commitParameters(m_device, m_lightObj);
      }
      if (m_headlightObj) {
        const float irr = m_rendererConfig.headlightEnabled ? m_rendererConfig.headlightIntensity : 0.f;
        anari::setParameter(m_device, m_headlightObj, "direction", m_camera.dir);
        anari::setParameter(m_device, m_headlightObj, "irradiance", irr);
        anari::commitParameters(m_device, m_headlightObj);
      }
      anari::commitParameters(m_device, m_world);
    }

    m_rendererDirty = false;
  }

  return true;
}

// Tear down the current ANARI world and create a fresh one via buildScene().
// Any outstanding framebuffer mappings are released first because the previous
// world's data may become invalid.  The new world is attached to the frame
// object and committed so the next render() picks it up.
bool GaussianRendererCore::rebuildWorld(std::string *errorMessage) {
  clearFrameMappings();

  auto newWorld = buildScene(m_device, m_data, m_scaleFactor);
  if (!newWorld) {
    if (errorMessage)
      *errorMessage = "Failed to build ANARI world.";
    return false;
  }

  if (m_world)
    anari::release(m_device, m_world);
  m_world = newWorld;

  if (m_lightObj) {
    anari::setParameter(m_device, m_lightObj, "direction", m_rendererConfig.lightDirection);
    anari::setParameter(m_device, m_lightObj, "irradiance", m_rendererConfig.lightIntensity);
    anari::commitParameters(m_device, m_lightObj);
  }
  if (m_headlightObj) {
    const float irr = m_rendererConfig.headlightEnabled ? m_rendererConfig.headlightIntensity : 0.f;
    anari::setParameter(m_device, m_headlightObj, "direction", m_camera.dir);
    anari::setParameter(m_device, m_headlightObj, "irradiance", irr);
    anari::commitParameters(m_device, m_headlightObj);
  }
  {
    anari::Light lights[2];
    int numLights = 0;
    if (m_lightObj)
      lights[numLights++] = m_lightObj;
    if (m_headlightObj)
      lights[numLights++] = m_headlightObj;
    if (numLights > 0)
      anari::setParameterArray1D(m_device, m_world, "light", lights, numLights);
    anari::commitParameters(m_device, m_world);
  }

  anari::setParameter(m_device, m_frameObj, "world", m_world);
  anari::commitParameters(m_device, m_frameObj);

  m_worldDirty = false;
  return true;
}

// Query the device's advertised extensions for ANARI_NV_FRAME_BUFFERS_CUDA,
// which allows mapping the framebuffer as a CUDA device pointer instead of
// copying to host.  This is required for the zero-copy GPU display path.
bool GaussianRendererCore::checkCudaFrameExtension() {
  const auto *list = (const char *const *)anariGetObjectInfo(m_device, ANARI_DEVICE, "default", "extension", ANARI_STRING_LIST);
  if (!list)
    return false;
  for (const char *const *it = list; *it != nullptr; ++it) {
    if (std::strcmp(*it, kCudaFrameExtension) == 0)
      return true;
  }
  return false;
}

// Compute an initial camera look-at point (m_focusCenter) and viewing distance
// (m_focusDistance) that frame the loaded Gaussian splat scene well, without
// requiring any camera metadata from the training data.
//
// The approach is deliberately robust to outliers -- Gaussian splat PLY files
// commonly contain a long tail of far-flung or oversized splats that would
// badly skew a naive bounding-box strategy.
//
// Algorithm overview
// ------------------
// 1. Fallback:  Start with the AABB center / 30 % of the diagonal as a safe
//    default in case the point cloud is empty or the statistics below fail.
//
// 2. Robust center:  Use the per-axis *median* (50th percentile) of all
//    Gaussian positions.  The median is insensitive to outlier splats that sit
//    far from the bulk of the scene.
//
// 3. Effective radii:  For each Gaussian compute an "effective radius" --
//    the distance from the robust center plus the largest scale axis of that
//    Gaussian (scaled by m_scaleFactor).  This captures both how far away and
//    how large each splat is, giving a single measure of how much viewing
//    distance is needed to see it.
//
// 4. 90th-percentile radius:  Take the 90th percentile of these effective
//    radii.  Ignoring the top 10 % avoids blowing up the distance because of a
//    handful of extreme outliers while still covering the vast majority of
//    visible content.
//
// 5. Viewing distance:  Multiply the 90th-percentile radius by 2.4 to convert
//    from a scene-space radius to a comfortable viewing distance (roughly the
//    distance at which the content subtends ~45° in the viewport).  The result
//    is then clamped to [2 %, 60 %] of the full bounding-box diagonal so the
//    camera never ends up absurdly close or absurdly far.
//
// 6. Validation:  The heuristic values are adopted only when they are finite
//    and positive; otherwise the AABB-based fallback from step 1 is kept.
void GaussianRendererCore::computeCameraHeuristics() {
  // Step 1 -- AABB-based fallback.
  m_focusCenter = m_sceneBounds.center;
  m_focusDistance = std::max(0.1f, m_sceneBounds.diagonal * 0.3f);

  const size_t n = m_data.positions.size();
  if (n == 0)
    return;

  // Step 2 -- Robust (median) center via per-axis 50th percentile.
  std::vector<float> xs;
  std::vector<float> ys;
  std::vector<float> zs;
  xs.reserve(n);
  ys.reserve(n);
  zs.reserve(n);
  for (const auto &p : m_data.positions) {
    xs.push_back(p[0]);
    ys.push_back(p[1]);
    zs.push_back(p[2]);
  }

  vec3 robustCenter = {percentileValue(xs, 0.5f), percentileValue(ys, 0.5f), percentileValue(zs, 0.5f)};

  // Step 3 -- Effective radius per Gaussian: distance to robust center + largest
  // scale axis (the Gaussian's visual extent along its dominant direction).
  std::vector<float> radii;
  radii.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const auto &p = m_data.positions[i];
    const float dx = p[0] - robustCenter[0];
    const float dy = p[1] - robustCenter[1];
    const float dz = p[2] - robustCenter[2];
    const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    const float maxScale = std::max({m_data.scales[i][0], m_data.scales[i][1], m_data.scales[i][2]});
    radii.push_back(dist + maxScale * m_scaleFactor);
  }

  // Step 4 -- 90th-percentile radius: covers the vast majority of Gaussians
  // while discarding the most extreme outliers.
  const float r90 = percentileValue(radii, 0.90f);
  if (!(r90 > 0.f) || !std::isfinite(r90))
    return;

  // Step 5 -- Convert to viewing distance and clamp to the scene's AABB.
  float distance = 2.4f * r90;
  if (m_sceneBounds.diagonal > 0.f) {
    distance = std::max(distance, 0.02f * std::max(1e-3f, m_sceneBounds.diagonal));
    distance = std::min(distance, 0.6f * m_sceneBounds.diagonal);
  }

  // Step 6 -- Adopt heuristic values only if they are valid.
  if (std::isfinite(distance) && distance > 0.f) {
    m_focusCenter = robustCenter;
    m_focusDistance = distance;
  }
}

void GaussianRendererCore::clearFrameMappings() {
  unmapColorHost();
  unmapColorCUDA();
}

// ANARI status callback registered at device creation.  Forwards warnings and
// errors to stderr; informational and debug messages are silently dropped to
// keep the console clean during normal operation.
void GaussianRendererCore::statusCallback(const void *userData, ANARIDevice, ANARIObject source, ANARIDataType, ANARIStatusSeverity severity, ANARIStatusCode, const char *message) {
  (void)userData;
  if (severity == ANARI_SEVERITY_FATAL_ERROR)
    std::fprintf(stderr, "[FATAL][%p] %s\n", source, message);
  else if (severity == ANARI_SEVERITY_ERROR)
    std::fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  else if (severity == ANARI_SEVERITY_WARNING)
    std::fprintf(stderr, "[WARN ][%p] %s\n", source, message);
}
