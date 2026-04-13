// GaussianRendererCore -- headless ANARI renderer for 3D Gaussian Splats.
//
// This class owns the full ANARI pipeline (library, device, world, camera,
// renderer, frame) and exposes a small public API that lets a front-end (e.g.
// an interactive GLFW viewer or an offline batch tool) drive rendering without
// knowing ANARI internals.
//
// The renderer is device-agnostic: any ANARI library (e.g. "visrtx", "helide")
// can be selected at runtime via InitOptions::libraryName.  CUDA framebuffer
// support is detected and exposed opportunistically but is not required.
//
// Typical usage:
//   1.  Create a GaussianRendererCore and call init() with a PLY path.
//   2.  On each frame: call setCamera() / setRendererConfig() / setFrameSize()
//       as needed, then run() to render.
//   3.  Retrieve the result via mapColorHost() (CPU readback) or, when
//       available, copyColorCUDAToDevice() (zero-copy GPU path).
//
// Thread safety: none -- all calls must be serialised by the caller.
#pragma once

#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>

#ifdef GRUT_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void *;
#endif

#include <string>

#include "gaussian_common.h"

// Parameters forwarded to the ANARI renderer object each frame.
struct RendererConfig {
  vec4 bgColor{0.1f, 0.1f, 0.1f, 1.f};   // background clear colour (RGBA)
  float ambientRadiance{1.0f};           // intensity of ambient fill light
  int spp{1};                            // samples per pixel
  vec3 lightDirection{-1.f, -1.f, -1.f}; // directional light direction
  float lightIntensity{3.0f};            // directional light irradiance
  bool headlightEnabled{true};           // camera-following directional light
  float headlightIntensity{5.0f};        // irradiance of the headlight
};

// Minimal perspective-camera description passed into the ANARI camera.
struct CameraState {
  vec3 eye{0.f, 0.f, 0.f};                   // camera world-space position
  vec3 dir{0.f, 0.f, 1.f};                   // forward direction (unit vector)
  vec3 up{0.f, -1.f, 0.f};                   // up vector (Y-down convention)
  float aspect{16.f / 9.f};                  // viewport width / height
  float fovy{3.14159265358979323846f / 3.f}; // vertical FOV in radians (ANARI default)
};

// Everything needed to bootstrap the renderer.
struct InitOptions {
  std::string plyPath;               // path to a 3DGS .ply file
  std::string libraryName{"visrtx"}; // ANARI library to load (e.g. "visrtx", "helide")
  float scaleFactor{1.0f};           // global multiplier on Gaussian scales
  float opacityThreshold{0.05f};     // discard Gaussians below this opacity
  uvec2 frameSize{1920, 1080};       // initial framebuffer resolution
  RendererConfig rendererConfig{};   // initial renderer settings
  bool useFloat32Color{false};       // true = FLOAT32_VEC4, false = UFIXED8_RGBA_SRGB
};

// Binding-friendly description of a mapped CUDA framebuffer, free of ANARI
// header dependencies so that Python bindings can consume it directly.
struct MappedFrameInfo {
  const void *data = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t bytesPerPixel = 0; // 4 for uint8 RGBA, 16 for float32 RGBA
  bool isFloat = false;
};

class GaussianRendererCore {
public:
  GaussianRendererCore() = default;
  ~GaussianRendererCore();

  // --- Lifecycle ----------------------------------------------------------

  /// Load a .ply, create the ANARI device (via the library named in
  /// options.libraryName) and ANARI objects, and perform the first
  /// render-ready commit.  Must be called exactly once; returns false (with
  /// a message) on any failure.
  bool init(const InitOptions &options, std::string *errorMessage = nullptr);

  /// Render one frame synchronously (blocks until complete).  Dirty state set
  /// by the setters below is flushed automatically before the render call.
  bool run(std::string *errorMessage = nullptr);

  // --- Per-frame parameter setters ----------------------------------------
  // These mark the corresponding ANARI object dirty; the actual ANARI
  // commit happens lazily inside run() via applyPendingUpdates().

  void setFrameSize(const uvec2 &size);
  void setCamera(const CameraState &camera);
  void setRendererConfig(const RendererConfig &config);

  /// Change the global Gaussian scale factor and immediately rebuild the ANARI
  /// world (because every instance transform depends on it).
  bool setScaleFactor(float scaleFactor, std::string *errorMessage = nullptr);

  // --- Framebuffer access -------------------------------------------------

  /// Map the rendered colour buffer to host (CPU) memory.  The caller must
  /// call unmapColorHost() when done.  Repeated calls auto-unmap the previous
  /// mapping.
  anari::MappedFrameData<uint32_t> mapColorHost(std::string *errorMessage = nullptr);
  void unmapColorHost();

  /// Map the rendered colour buffer as a CUDA device pointer (requires the
  /// ANARI_NV_FRAME_BUFFERS_CUDA extension).  Must be followed by
  /// unmapColorCUDA().
  anari::MappedFrameData<void> mapColorCUDA(std::string *errorMessage = nullptr);
  void unmapColorCUDA();

  /// Map the CUDA colour buffer and return a MappedFrameInfo describing the
  /// pointer, dimensions, and pixel format.  The caller must call
  /// unmapColorCUDA() when done.  Designed for use by Python bindings that
  /// cannot include ANARI headers.
  MappedFrameInfo mapColorCUDAInfo(std::string *errorMessage = nullptr);

  /// Convenience: map the CUDA colour buffer, perform a device-to-device 2D
  /// memcpy into |dstPtr| (pitched), and unmap.  Supports both synchronous
  /// (stream == nullptr) and async copies.
  bool copyColorCUDAToDevice(void *dstPtr, size_t dstPitchBytes, cudaStream_t stream = nullptr, std::string *errorMessage = nullptr);

  // --- Read-only accessors ------------------------------------------------

  const vec3 &sceneCenter() const { return m_sceneBounds.center; }
  float sceneDiagonal() const { return m_sceneBounds.diagonal; }
  const vec3 &focusCenter() const { return m_focusCenter; }
  float focusDistance() const { return m_focusDistance; }
  size_t gaussianCount() const { return m_data.positions.size(); }
  float lastDurationSeconds() const { return m_lastDurationSeconds; }
  bool supportsCudaFrameBuffers() const { return m_supportsCudaFrameBuffers; }
  const uvec2 &frameSize() const { return m_frameSize; }

private:
  // --- Internal helpers ---------------------------------------------------

  /// Flush all dirty flags to their ANARI objects.  Called at the start of
  /// every run() and once at the end of init().
  bool applyPendingUpdates(std::string *errorMessage);

  /// Rebuild the ANARI world from scratch using the current m_data and
  /// m_scaleFactor.  This is expensive (re-creates all instance transforms).
  bool rebuildWorld(std::string *errorMessage);

  /// Query the device for the ANARI_NV_FRAME_BUFFERS_CUDA extension.
  bool checkCudaFrameExtension();

  /// Derive m_focusCenter and m_focusDistance from the loaded Gaussian data
  /// using outlier-robust percentile statistics (see implementation for the
  /// full algorithm description).
  void computeCameraHeuristics();

  /// Release any outstanding host or CUDA framebuffer mappings.
  void clearFrameMappings();

  /// ANARI status callback -- routes device messages to stderr, filtered by
  /// severity (warnings and above).
  static void statusCallback(const void *userData, ANARIDevice, ANARIObject source, ANARIDataType, ANARIStatusSeverity severity, ANARIStatusCode, const char *message);

private:
  // --- State flags --------------------------------------------------------
  bool m_initialized{false};
  bool m_supportsCudaFrameBuffers{false};

  // --- Scene data ---------------------------------------------------------
  GaussianData m_data;
  SceneBounds m_sceneBounds;
  vec3 m_focusCenter{0.f, 0.f, 0.f}; // heuristic camera look-at point
  float m_focusDistance{1.f};        // heuristic viewing distance

  // --- User-facing parameters (latched, applied lazily) -------------------
  float m_scaleFactor{1.f};
  bool m_useFloat32Color{false};
  uvec2 m_frameSize{1920, 1080};
  CameraState m_camera;
  RendererConfig m_rendererConfig;

  // --- Dirty flags (one per ANARI object that can change between frames) --
  bool m_worldDirty{false};
  bool m_frameSizeDirty{false};
  bool m_cameraDirty{false};
  bool m_rendererDirty{false};

  // --- Framebuffer mapping state ------------------------------------------
  bool m_hostMapped{false};
  bool m_cudaMapped{false};

  // --- Diagnostics --------------------------------------------------------
  float m_lastDurationSeconds{0.f};

  // --- ANARI object handles (owned; released in destructor) ---------------
  anari::Library m_library{nullptr};
  anari::Device m_device{nullptr};
  anari::World m_world{nullptr};
  anari::Camera m_cameraObj{nullptr};
  anari::Renderer m_rendererObj{nullptr};
  anari::Light m_lightObj{nullptr};
  anari::Light m_headlightObj{nullptr};
  anari::Frame m_frameObj{nullptr};
};
