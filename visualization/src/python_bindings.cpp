#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "renderer_core.h"

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// Thin wrapper returned by map_color_cuda().  Exposes __cuda_array_interface__
// so that polyscope, torch, and cupy can consume the device pointer directly.
struct MappedCUDAFrame {
  const void *data;
  uint32_t width;
  uint32_t height;
  bool isFloat;

  py::dict cuda_array_interface() const {
    py::dict d;
    d["shape"] = py::make_tuple(height, width, 4);
    d["typestr"] = isFloat ? "<f4" : "|u1";
    d["data"] = py::make_tuple(reinterpret_cast<uintptr_t>(data), true /* read-only */);
    d["version"] = 3;
    d["strides"] = py::none();
    return d;
  }
};

// Context-manager wrapper so that map/unmap is exception-safe:
//   with renderer.map_color_cuda() as frame:
//       buffer.update_data_from_device(frame)
struct CUDAFrameContext {
  GaussianRendererCore &renderer;
  MappedCUDAFrame frame{};

  explicit CUDAFrameContext(GaussianRendererCore &r) : renderer(r) {}

  MappedCUDAFrame &enter() {
    std::string err;
    auto info = renderer.mapColorCUDAInfo(&err);
    if (!info.data)
      throw std::runtime_error(err);
    frame.data = info.data;
    frame.width = info.width;
    frame.height = info.height;
    frame.isFloat = info.isFloat;
    return frame;
  }

  void exit(const py::object &, const py::object &, const py::object &) { renderer.unmapColorCUDA(); }
};

static void throw_on_error(bool ok, const std::string &msg) {
  if (!ok)
    throw std::runtime_error(msg);
}

PYBIND11_MODULE(_gaussian_renderer_core, m) {
  m.doc() = "Python bindings for GaussianRendererCore (ANARI)";

  // --- RendererConfig ----------------------------------------------------

  py::class_<RendererConfig>(m, "RendererConfig")
      .def(py::init<>())
      .def_readwrite("bg_color", &RendererConfig::bgColor)
      .def_readwrite("ambient_radiance", &RendererConfig::ambientRadiance)
      .def_readwrite("spp", &RendererConfig::spp)
      .def_readwrite("light_direction", &RendererConfig::lightDirection)
      .def_readwrite("light_intensity", &RendererConfig::lightIntensity)
      .def_readwrite("headlight_enabled", &RendererConfig::headlightEnabled)
      .def_readwrite("headlight_intensity", &RendererConfig::headlightIntensity);

  // --- CameraState -------------------------------------------------------

  py::class_<CameraState>(m, "CameraState")
      .def(py::init<>())
      .def_readwrite("eye", &CameraState::eye)
      .def_readwrite("dir", &CameraState::dir)
      .def_readwrite("up", &CameraState::up)
      .def_readwrite("aspect", &CameraState::aspect)
      .def_readwrite("fovy", &CameraState::fovy);

  // --- InitOptions -------------------------------------------------------

  py::class_<InitOptions>(m, "InitOptions")
      .def(py::init<>())
      .def_readwrite("ply_path", &InitOptions::plyPath)
      .def_readwrite("library_name", &InitOptions::libraryName)
      .def_readwrite("scale_factor", &InitOptions::scaleFactor)
      .def_readwrite("opacity_threshold", &InitOptions::opacityThreshold)
      .def_readwrite("frame_size", &InitOptions::frameSize)
      .def_readwrite("renderer_config", &InitOptions::rendererConfig)
      .def_readwrite("use_float32_color", &InitOptions::useFloat32Color);

  // --- MappedCUDAFrame ---------------------------------------------------

  py::class_<MappedCUDAFrame>(m, "MappedCUDAFrame")
      .def_readonly("width", &MappedCUDAFrame::width)
      .def_readonly("height", &MappedCUDAFrame::height)
      .def_readonly("is_float", &MappedCUDAFrame::isFloat)
      .def_property_readonly("__cuda_array_interface__", &MappedCUDAFrame::cuda_array_interface)
      .def("data_ptr", [](const MappedCUDAFrame &self) { return reinterpret_cast<uintptr_t>(self.data); });

  // --- CUDAFrameContext (context manager) --------------------------------

  py::class_<CUDAFrameContext>(m, "CUDAFrameContext").def("__enter__", &CUDAFrameContext::enter, py::return_value_policy::reference_internal).def("__exit__", &CUDAFrameContext::exit);

  // --- GaussianRendererCore ----------------------------------------------

  py::class_<GaussianRendererCore>(m, "GaussianRendererCore")
      .def(py::init<>())

      // Lifecycle
      .def(
          "init",
          [](GaussianRendererCore &self, const InitOptions &opts) {
            std::string err;
            throw_on_error(self.init(opts, &err), err);
          },
          py::arg("options"))
      .def("run",
           [](GaussianRendererCore &self) {
             std::string err;
             throw_on_error(self.run(&err), err);
           })

      // Setters
      .def("set_frame_size", &GaussianRendererCore::setFrameSize, py::arg("size"))
      .def("set_camera", &GaussianRendererCore::setCamera, py::arg("camera"))
      .def("set_renderer_config", &GaussianRendererCore::setRendererConfig, py::arg("config"))
      .def(
          "set_scale_factor",
          [](GaussianRendererCore &self, float sf) {
            std::string err;
            throw_on_error(self.setScaleFactor(sf, &err), err);
          },
          py::arg("scale_factor"))

      // CUDA framebuffer access (context manager)
      .def("map_color_cuda", [](GaussianRendererCore &self) { return CUDAFrameContext(self); })

      // Direct map/unmap for advanced use
      .def("map_color_cuda_info",
           [](GaussianRendererCore &self) {
             std::string err;
             auto info = self.mapColorCUDAInfo(&err);
             if (!info.data)
               throw std::runtime_error(err);
             MappedCUDAFrame f;
             f.data = info.data;
             f.width = info.width;
             f.height = info.height;
             f.isFloat = info.isFloat;
             return f;
           })
      .def("unmap_color_cuda", &GaussianRendererCore::unmapColorCUDA)

      // Read-only accessors
      .def_property_readonly("scene_center", &GaussianRendererCore::sceneCenter)
      .def_property_readonly("scene_diagonal", &GaussianRendererCore::sceneDiagonal)
      .def_property_readonly("focus_center", &GaussianRendererCore::focusCenter)
      .def_property_readonly("focus_distance", &GaussianRendererCore::focusDistance)
      .def_property_readonly("gaussian_count", &GaussianRendererCore::gaussianCount)
      .def_property_readonly("last_duration_seconds", &GaussianRendererCore::lastDurationSeconds)
      .def_property_readonly("supports_cuda_frame_buffers", &GaussianRendererCore::supportsCudaFrameBuffers)
      .def_property_readonly("frame_size", &GaussianRendererCore::frameSize);
}
