/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Interactive Gaussian Viewer -- GLFW + Dear ImGui + ANARI
// Architecture modeled after VIDILabs/open-volume-renderer main_app.cpp:
// async double-buffered rendering with TransactionalValue handoff.

// clang-format off
#define GLFW_INCLUDE_NONE
#include <glad/gl.h>          // must precede cuda_gl_interop.h / GLFW
#include <GLFW/glfw3.h>
#ifdef GRUT_HAS_CUDA
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#endif
// clang-format on

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "renderer_core.h"

#include <atomic>
#include <cfloat>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

// ═══════════════════════════════════════════════════════════════════════════════
//  Synchronization primitives (same pattern as vidi::TransactionalValue /
//  vidi::AsyncLoop from open-volume-renderer)
// ═══════════════════════════════════════════════════════════════════════════════

// Triple-buffer with atomic swap: producer writes to staging, consumer reads
// from committed.  No locks on the hot path.
template <typename T> class TransactionalValue {
public:
  TransactionalValue() = default;
  explicit TransactionalValue(const T &v) : committed_(v), staging_(v) {}

  TransactionalValue &operator=(const T &v) {
    std::lock_guard<std::mutex> lk(mtx_);
    staging_ = v;
    dirty_.store(true, std::memory_order_release);
    return *this;
  }

  template <typename Fn> void assign(Fn &&fn) {
    std::lock_guard<std::mutex> lk(mtx_);
    fn(staging_);
    dirty_.store(true, std::memory_order_release);
  }

  bool update() {
    if (!dirty_.load(std::memory_order_acquire))
      return false;
    std::lock_guard<std::mutex> lk(mtx_);
    committed_ = staging_;
    dirty_.store(false, std::memory_order_release);
    return true;
  }

  template <typename Fn> bool update(Fn &&fn) {
    if (!dirty_.load(std::memory_order_acquire))
      return false;
    std::lock_guard<std::mutex> lk(mtx_);
    committed_ = staging_;
    dirty_.store(false, std::memory_order_release);
    fn(committed_);
    return true;
  }

  const T &get() const { return committed_; }
  const T &ref() const { return committed_; }

private:
  T committed_{};
  T staging_{};
  std::mutex mtx_;
  std::atomic<bool> dirty_{false};
};

// Repeatedly calls a function on a background thread until stopped.
class AsyncLoop {
public:
  explicit AsyncLoop(std::function<void()> fn) : fn_(std::move(fn)) {}
  ~AsyncLoop() { stop(); }

  void start() {
    if (running_.load())
      return;
    running_.store(true);
    thread_ = std::thread([this] {
      while (running_.load())
        fn_();
    });
  }

  void stop() {
    running_.store(false);
    if (thread_.joinable())
      thread_.join();
  }

private:
  std::function<void()> fn_;
  std::thread thread_;
  std::atomic<bool> running_{false};
};

// Simple FPS counter that measures every N frames.
struct FPSCounter {
  static constexpr int WINDOW = 10;
  int frame{0};

  bool count() {
    frame++;
    auto now = std::chrono::steady_clock::now();
    if (frame % WINDOW == 0) {
      double elapsed = std::chrono::duration<double>(now - last_).count();
      if (elapsed > 1e-6)
        fps_.store(WINDOW / elapsed, std::memory_order_relaxed);
      last_ = now;
      return true;
    }
    return false;
  }

  double value() const { return fps_.load(std::memory_order_relaxed); }

private:
  std::atomic<double> fps_{0.0};
  std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

// Background FPS estimator with adaptive smoothing:
// - low FPS updates react immediately (alpha -> 1)
// - high FPS updates are smoothed (alpha -> 0.15)
class AdaptiveFpsEma {
public:
  void addSampleSeconds(double deviceSeconds) {
    if (!(deviceSeconds > 1e-6))
      return;

    const double instantFps = 1.0 / deviceSeconds;
    const double alpha = computeAlpha(instantFps);
    const double prev = fps_.load(std::memory_order_relaxed);
    const double next = prev > 0.0 ? (1.0 - alpha) * prev + alpha * instantFps : instantFps;
    fps_.store(next, std::memory_order_relaxed);
  }

  double value() const { return fps_.load(std::memory_order_relaxed); }

private:
  static double computeAlpha(double instantFps) {
    if (instantFps <= kLowFpsNoSmoothing)
      return kAlphaNoSmoothing;
    if (instantFps >= kHighFpsSmoothed)
      return kAlphaSmoothed;

    const double t = (instantFps - kLowFpsNoSmoothing) / (kHighFpsSmoothed - kLowFpsNoSmoothing);
    return kAlphaNoSmoothing + (kAlphaSmoothed - kAlphaNoSmoothing) * t;
  }

  static constexpr double kLowFpsNoSmoothing = 1.0;
  static constexpr double kHighFpsSmoothed = 30.0;
  static constexpr double kAlphaNoSmoothing = 1.0;
  static constexpr double kAlphaSmoothed = 0.15;

  std::atomic<double> fps_{0.0};
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Camera
// ═══════════════════════════════════════════════════════════════════════════════

struct OrbitCamera {
  vec3 center{0.f, 0.f, 0.f};
  float distance{1.f};
  float yaw{0.f};
  float pitch{0.f};
  vec3 up{0.f, -1.f, 0.f};

  CameraState state(float aspect) const {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    vec3 offset = {cy * cp * distance, sp * distance, sy * cp * distance};
    vec3 eye = {center[0] + offset[0], center[1] + offset[1], center[2] + offset[2]};
    float len = std::sqrt(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]);
    vec3 dir = {-offset[0] / len, -offset[1] / len, -offset[2] / len};
    return {eye, dir, up, aspect};
  }

  void orbit(float dx, float dy) {
    yaw -= dx;
    pitch = std::clamp(pitch - dy, -1.5f, 1.5f);
  }

  void pan(float dx, float dy) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    vec3 right = {-sy, 0.f, cy};
    vec3 camUp = {0.f, 1.f, 0.f};
    for (int i = 0; i < 3; i++)
      center[i] += right[i] * dx * distance * 0.002f + camUp[i] * dy * distance * 0.002f;
  }

  void dolly(float amount) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    vec3 dir = {-cy * cp, -sp, -sy * cp};
    for (int i = 0; i < 3; i++)
      center[i] += dir[i] * amount * distance * 0.05f;
  }

  void strafe(float amount) {
    float sy = std::sin(yaw), cy = std::cos(yaw);
    vec3 right = {-sy, 0.f, cy};
    for (int i = 0; i < 3; i++)
      center[i] += right[i] * amount * distance * 0.05f;
  }

  void zoom(float delta) {
    distance *= (1.f - delta * 0.1f);
    if (distance < 0.01f)
      distance = 0.01f;
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Shared data structures between threads
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
//  Application
// ═══════════════════════════════════════════════════════════════════════════════

struct App {
  GaussianRendererCore renderer_core;
  std::mutex renderer_mutex;

  // Double-buffered communication (same pattern as main_app.cpp)
  TransactionalValue<CameraState> camera_shared;
  TransactionalValue<uvec2> frame_size_shared{uvec2{0, 0}};
  TransactionalValue<RendererConfig> renderer_config_shared;
  TransactionalValue<float> scale_factor_shared{1.0f};
  std::atomic<bool> frame_ready{false};

  // Producer-consumer handoff: background thread waits after producing a frame
  // until the foreground has consumed it, preventing mutex starvation.
  std::mutex frame_handoff_mtx_;
  std::condition_variable frame_consumed_cv_;
  bool frame_consumed_{true};

  // Background thread
  AsyncLoop async_loop{std::bind(&App::render_background, this)};
  AdaptiveFpsEma bg_fps_ema;
  FPSCounter fg_fps;

  // GUI-thread local state
  GLFWwindow *window{nullptr};
  GLuint frame_texture{0};
#ifdef GRUT_HAS_CUDA
  cudaGraphicsResource_t frame_texture_cuda{nullptr};
  bool cuda_gl_interop{false};
#endif
  uvec2 fb_size{0, 0};

  OrbitCamera orbit_cam;
  bool camera_modified{true};

  bool gui_enabled{true};
  bool async_enabled{true};
  bool invert_orbit_y{false};

  struct {
    float scaleFactor{1.0f};
    float bgColor[3]{0.1f, 0.1f, 0.1f};
    float ambientRadiance{1.0f};
    int spp{1};
    float lightPhi{225.f};
    float lightTheta{225.f};
    float lightIntensity{3.0f};
    bool headlightEnabled{true};
    float headlightIntensity{3.0f};
  } config;

  float sceneDiagonal{1.f};
  vec3 sceneCenter{0.f, 0.f, 0.f};

  // Mouse state
  double lastMouseX{0}, lastMouseY{0};
  bool lmbDown{false}, rmbDown{false};

  // ─── Background thread ───────────────────────────────────────────────────

  void render_background() {
    // Wait until the foreground has consumed the previous frame before
    // rendering another.  The short timeout lets the AsyncLoop check its
    // stop flag without requiring an external wake-up.
    {
      std::unique_lock<std::mutex> lk(frame_handoff_mtx_);
      frame_consumed_cv_.wait_for(lk, std::chrono::milliseconds(50), [this] { return frame_consumed_; });
      if (!frame_consumed_)
        return;
    }

    std::lock_guard<std::mutex> guard(renderer_mutex);

    if (frame_size_shared.update()) {
      const uvec2 sz = frame_size_shared.ref();
      if (sz[0] == 0 || sz[1] == 0)
        return;
      renderer_core.setFrameSize(sz);
    }
    {
      const uvec2 sz = frame_size_shared.ref();
      if (sz[0] == 0 || sz[1] == 0)
        return;
    }

    if (camera_shared.update())
      renderer_core.setCamera(camera_shared.ref());

    if (renderer_config_shared.update())
      renderer_core.setRendererConfig(renderer_config_shared.ref());

    if (scale_factor_shared.update()) {
      std::string error_message;
      if (!renderer_core.setScaleFactor(scale_factor_shared.ref(), &error_message))
        fprintf(stderr, "Failed to rebuild scene: %s\n", error_message.c_str());
    }

    std::string error_message;
    if (!renderer_core.run(&error_message)) {
      fprintf(stderr, "Background render failed: %s\n", error_message.c_str());
      return;
    }
    const double device_seconds = std::max(0.0, static_cast<double>(renderer_core.lastDurationSeconds()));

    bg_fps_ema.addSampleSeconds(device_seconds);

    {
      std::lock_guard<std::mutex> lk(frame_handoff_mtx_);
      frame_consumed_ = false;
    }
    frame_ready.store(true, std::memory_order_release);
  }

  // ─── GUI thread: push camera ─────────────────────────────────────────────

  void push_camera() {
    if (!camera_modified)
      return;
    camera_modified = false;

    float aspect = fb_size[1] > 0 ? float(fb_size[0]) / float(fb_size[1]) : 1.f;
    camera_shared = orbit_cam.state(aspect);

    if (!async_enabled)
      render_background();
  }

  // ─── GUI thread: draw ────────────────────────────────────────────────────

#ifdef GRUT_HAS_CUDA
  bool check_cuda_gl_interop() const {
    unsigned int num_devices = 0;
    int cuda_devices[8] = {0};
    cudaError_t err = cudaGLGetDevices(&num_devices, cuda_devices, 8, cudaGLDeviceListAll);
    if (err != cudaSuccess) {
      cudaGetLastError();
      return false;
    }

    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess)
      return false;

    for (unsigned int i = 0; i < num_devices; ++i) {
      if (cuda_devices[i] == current_device)
        return true;
    }
    return false;
  }

  void unregister_frame_texture_cuda() {
    if (frame_texture_cuda) {
      cudaGraphicsUnregisterResource(frame_texture_cuda);
      frame_texture_cuda = nullptr;
    }
  }

  bool register_frame_texture_cuda() {
    unregister_frame_texture_cuda();
    if (!cuda_gl_interop)
      return false;

    cudaError_t err = cudaGraphicsGLRegisterImage(&frame_texture_cuda, frame_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to register GL texture with CUDA: %s\n", cudaGetErrorString(err));
      frame_texture_cuda = nullptr;
      return false;
    }
    return true;
  }

  bool upload_cuda_frame_to_texture() {
    if (!frame_texture_cuda)
      return false;

    std::lock_guard<std::mutex> guard(renderer_mutex);
    std::string error_message;
    auto color = renderer_core.mapColorCUDA(&error_message);
    if (!color.data) {
      fprintf(stderr, "Failed to map CUDA framebuffer: %s\n", error_message.c_str());
      return false;
    }

    if (color.pixelType != ANARI_UFIXED8_RGBA_SRGB && color.pixelType != ANARI_UFIXED8_VEC4) {
      fprintf(stderr, "Unsupported CUDA color pixel type for display upload.\n");
      renderer_core.unmapColorCUDA();
      return false;
    }

    const size_t row_bytes = color.width * 4;
    cudaError_t err = cudaGraphicsMapResources(1, &frame_texture_cuda, nullptr);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to map CUDA graphics resource: %s\n", cudaGetErrorString(err));
      renderer_core.unmapColorCUDA();
      return false;
    }

    cudaArray_t texture_array = nullptr;
    err = cudaGraphicsSubResourceGetMappedArray(&texture_array, frame_texture_cuda, 0, 0);
    if (err == cudaSuccess) {
      err = cudaMemcpy2DToArray(texture_array, 0, 0, color.data, row_bytes, row_bytes, color.height, cudaMemcpyDeviceToDevice);
    }

    cudaGraphicsUnmapResources(1, &frame_texture_cuda, nullptr);
    renderer_core.unmapColorCUDA();

    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy CUDA framebuffer into GL texture: %s\n", cudaGetErrorString(err));
      return false;
    }
    return true;
  }
#endif // GRUT_HAS_CUDA

  bool upload_host_frame_to_texture() {
    std::lock_guard<std::mutex> guard(renderer_mutex);
    std::string error_message;
    auto fb = renderer_core.mapColorHost(&error_message);
    if (!fb.data) {
      fprintf(stderr, "Failed to map host framebuffer: %s\n", error_message.c_str());
      return false;
    }

    glBindTexture(GL_TEXTURE_2D, frame_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fb.width, fb.height, GL_RGBA, GL_UNSIGNED_BYTE, fb.data);
    renderer_core.unmapColorHost();
    return true;
  }

  bool use_cuda_display_path() const {
#ifdef GRUT_HAS_CUDA
    return cuda_gl_interop && renderer_core.supportsCudaFrameBuffers();
#else
    return false;
#endif
  }

  void draw() {
    if (frame_ready.exchange(false, std::memory_order_acq_rel)) {
#ifdef GRUT_HAS_CUDA
      if (use_cuda_display_path())
        upload_cuda_frame_to_texture();
      else
#endif
        upload_host_frame_to_texture();
      {
        std::lock_guard<std::mutex> lk(frame_handoff_mtx_);
        frame_consumed_ = true;
      }
      frame_consumed_cv_.notify_one();
    }

    // Fullscreen quad (same approach as reference main_app.cpp draw())
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, frame_texture);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, fb_size[0], fb_size[1]);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)fb_size[0], 0.f, (float)fb_size[1], -1.f, 1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1.f, 1.f, 1.f);

    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 0.f);
    glVertex3f(0.f, 0.f, 0.f);
    glTexCoord2f(0.f, 1.f);
    glVertex3f(0.f, (float)fb_size[1], 0.f);
    glTexCoord2f(1.f, 1.f);
    glVertex3f((float)fb_size[0], (float)fb_size[1], 0.f);
    glTexCoord2f(1.f, 0.f);
    glVertex3f((float)fb_size[0], 0.f, 0.f);
    glEnd();

    // ImGui overlay
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (gui_enabled) {
      ImGui::SetNextWindowSizeConstraints(ImVec2(360, 300), ImVec2(FLT_MAX, FLT_MAX));
      if (ImGui::Begin("Control Panel", nullptr)) {
        ImGui::Checkbox("Invert Orbit Y", &invert_orbit_y);
        ImGui::Separator();

        bool renderer_dirty = false;

        if (ImGui::ColorEdit3("Background", config.bgColor)) {
          renderer_dirty = true;
        }
        if (ImGui::SliderFloat("Ambient Radiance", &config.ambientRadiance, 0.f, 5.f, "%.2f")) {
          renderer_dirty = true;
        }
        if (ImGui::SliderInt("Samples Per Pixel", &config.spp, 1, 64)) {
          renderer_dirty = true;
        }

        ImGui::Separator();
        if (ImGui::Checkbox("Headlight", &config.headlightEnabled))
          renderer_dirty = true;
        if (config.headlightEnabled) {
          if (ImGui::SliderFloat("Headlight Intensity", &config.headlightIntensity, 0.f, 10.f, "%.2f"))
            renderer_dirty = true;
        }

        ImGui::Separator();
        if (ImGui::SliderFloat("Light Phi", &config.lightPhi, 0.f, 360.f, "%.1f"))
          renderer_dirty = true;
        if (ImGui::SliderFloat("Light Theta", &config.lightTheta, -90.f, 90.f, "%.1f"))
          renderer_dirty = true;
        if (ImGui::SliderFloat("Light Intensity", &config.lightIntensity, 0.f, 10.f, "%.2f"))
          renderer_dirty = true;

        if (renderer_dirty) {
          RendererConfig rc;
          rc.bgColor = {config.bgColor[0], config.bgColor[1], config.bgColor[2], 1.f};
          rc.ambientRadiance = config.ambientRadiance;
          rc.spp = config.spp;

          float phi = config.lightPhi * (3.14159265f / 180.f);
          float theta = config.lightTheta * (3.14159265f / 180.f);
          rc.lightDirection = {std::cos(theta) * std::cos(phi), std::sin(theta), std::cos(theta) * std::sin(phi)};
          rc.lightIntensity = config.lightIntensity;
          rc.headlightEnabled = config.headlightEnabled;
          rc.headlightIntensity = config.headlightIntensity;

          renderer_config_shared = rc;
        }

        ImGui::Separator();
        if (ImGui::SliderFloat("Gaussian Scale", &config.scaleFactor, 0.01f, 10.f, "%.3f")) {
          scale_factor_shared = config.scaleFactor;
        }

        ImGui::Separator();
        ImGui::Text("Gaussians: %zu", renderer_core.gaussianCount());
        ImGui::Text("BG FPS (EMA): %.3f", bg_fps_ema.value());
        ImGui::Text("FG FPS: %.1f", fg_fps.value());
      }
      ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // FPS in title bar (same as reference)
    if (fg_fps.count()) {
      std::stringstream title;
      title << std::fixed << std::setprecision(3) << "Interactive Gaussian Viewer  |  fg=" << fg_fps.value() << " fps  bg=" << bg_fps_ema.value() << " fps";
      glfwSetWindowTitle(window, title.str().c_str());
    }
  }

  // ─── Resize ──────────────────────────────────────────────────────────────

  void resize(int w, int h) {
    if (w <= 0 || h <= 0)
      return;
    fb_size = {(unsigned)w, (unsigned)h};

    glBindTexture(GL_TEXTURE_2D, frame_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(fb_size[0]), static_cast<GLsizei>(fb_size[1]), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

#ifdef GRUT_HAS_CUDA
    if (use_cuda_display_path())
      register_frame_texture_cuda();
#endif

    frame_size_shared = fb_size;
    camera_modified = true;
  }

  // ─── Key handling ────────────────────────────────────────────────────────

  void onKey(int key, int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS)
      return;
    switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      break;
    case GLFW_KEY_G:
      gui_enabled = !gui_enabled;
      break;
    case GLFW_KEY_S: {
      std::lock_guard<std::mutex> guard(renderer_mutex);
      std::string error_message;
      auto fb = renderer_core.mapColorHost(&error_message);
      if (!fb.data) {
        fprintf(stderr, "Screenshot map failed: %s\n", error_message.c_str());
      } else {
        stbi_flip_vertically_on_write(1);
        stbi_write_png("screenshot.png", fb.width, fb.height, 4, fb.data, 4 * fb.width);
        renderer_core.unmapColorHost();
        printf("Screenshot saved: screenshot.png\n");
      }
      break;
    }
    default:
      break;
    }
  }

  // ─── Mouse handling ──────────────────────────────────────────────────────

  void onMouseButton(int button, int action, int /*mods*/) {
    if (ImGui::GetIO().WantCaptureMouse)
      return;
    if (button == GLFW_MOUSE_BUTTON_LEFT)
      lmbDown = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
      rmbDown = (action == GLFW_PRESS);
  }

  void onCursorPos(double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse)
      return;
    float dx = float(xpos - lastMouseX);
    float dy = float(ypos - lastMouseY);
    lastMouseX = xpos;
    lastMouseY = ypos;

    if (lmbDown) {
      const float orbit_dy = invert_orbit_y ? -dy : dy;
      orbit_cam.orbit(dx * 0.005f, orbit_dy * 0.005f);
      camera_modified = true;
    }
    if (rmbDown) {
      orbit_cam.pan(dx, dy);
      camera_modified = true;
    }
  }

  void onScroll(double /*xoffset*/, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse)
      return;
    orbit_cam.zoom(float(yoffset));
    camera_modified = true;
  }

  // ─── Keyboard-driven camera movement ──────────────────────────────────

  void updateKeyboardMovement(float dt) {
    constexpr float kOrbitRate = 1.5f; // rad/s
    constexpr float kMoveRate = 1.5f;
    bool moved = false;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
      orbit_cam.dolly(kMoveRate * dt);
      moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
      orbit_cam.dolly(-kMoveRate * dt);
      moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
      orbit_cam.strafe(-kMoveRate * dt);
      moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
      orbit_cam.strafe(kMoveRate * dt);
      moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
      orbit_cam.orbit(0.f, -kOrbitRate * dt);
      moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
      orbit_cam.orbit(0.f, kOrbitRate * dt);
      moved = true;
    }

    if (moved)
      camera_modified = true;
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  GLFW callbacks (forward to App)
// ═══════════════════════════════════════════════════════════════════════════════

static App *g_app = nullptr;

static void glfwResizeCb(GLFWwindow *, int w, int h) { g_app->resize(w, h); }
static void glfwKeyCb(GLFWwindow *window, int k, int sc, int a, int m) {
  ImGui_ImplGlfw_KeyCallback(window, k, sc, a, m);
  if (!ImGui::GetIO().WantCaptureKeyboard)
    g_app->onKey(k, sc, a, m);
}
static void glfwMouseButtonCb(GLFWwindow *window, int b, int a, int m) {
  ImGui_ImplGlfw_MouseButtonCallback(window, b, a, m);
  g_app->onMouseButton(b, a, m);
}
static void glfwCursorPosCb(GLFWwindow *window, double x, double y) {
  ImGui_ImplGlfw_CursorPosCallback(window, x, y);
  g_app->onCursorPos(x, y);
}
static void glfwScrollCb(GLFWwindow *window, double x, double y) {
  ImGui_ImplGlfw_ScrollCallback(window, x, y);
  g_app->onScroll(x, y);
}
static void glfwCharCb(GLFWwindow *window, unsigned int c) { ImGui_ImplGlfw_CharCallback(window, c); }

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

static void printUsage(const char *argv0) {
  printf("Usage: %s <path.ply> [options]\n", argv0);
  printf("  --library NAME          ANARI library (default: visrtx)\n");
  printf("  --scale-factor F        Multiply Gaussian scales (default: 1.0)\n");
  printf("  --opacity-threshold T   Min opacity to keep (default: 0.05)\n");
  printf("  --resolution WxH        Window resolution (default: 1920x1080)\n");
  printf("\nControls:\n");
  printf("  LMB drag    Orbit\n");
  printf("  RMB drag    Pan\n");
  printf("  Scroll      Zoom\n");
  printf("  W/S         Dolly forward/backward\n");
  printf("  A/D         Strafe left/right\n");
  printf("  Q/E         Orbit pitch (up/down)\n");
  printf("  G           Toggle GUI\n");
  printf("  S           Screenshot\n");
  printf("  Escape      Quit\n");
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string plyPath = argv[1];
  std::string libraryName = "visrtx";
  float scaleFactor = 1.0f;
  float opacityThreshold = 0.05f;
  uvec2 winSize = {1920, 1080};

  for (int i = 2; i < argc; i++) {
    if (std::strcmp(argv[i], "--library") == 0 && i + 1 < argc)
      libraryName = argv[++i];
    else if (std::strcmp(argv[i], "--scale-factor") == 0 && i + 1 < argc)
      scaleFactor = std::strtof(argv[++i], nullptr);
    else if (std::strcmp(argv[i], "--opacity-threshold") == 0 && i + 1 < argc)
      opacityThreshold = std::strtof(argv[++i], nullptr);
    else if (std::strcmp(argv[i], "--resolution") == 0 && i + 1 < argc) {
      unsigned w = 0, h = 0;
      if (std::sscanf(argv[++i], "%ux%u", &w, &h) == 2 && w > 0 && h > 0)
        winSize = {w, h};
    } else {
      printUsage(argv[0]);
      return 1;
    }
  }

  // ── GLFW + OpenGL ─────────────────────────────────────────────────────────

  if (!glfwInit()) {
    fprintf(stderr, "Failed to init GLFW\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

  GLFWwindow *window = glfwCreateWindow(winSize[0], winSize[1], "Interactive Gaussian Viewer", nullptr, nullptr);
  if (!window) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  int version = gladLoadGL(glfwGetProcAddress);
  if (!version) {
    fprintf(stderr, "Failed to load OpenGL via glad\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }
  printf("OpenGL %d.%d loaded\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

  // ── App state ─────────────────────────────────────────────────────────────

  App app;
  app.window = window;
  app.config.scaleFactor = scaleFactor;

  InitOptions options;
  options.plyPath = plyPath;
  options.libraryName = libraryName;
  options.scaleFactor = scaleFactor;
  options.opacityThreshold = opacityThreshold;
  options.frameSize = winSize;
  options.rendererConfig.spp = 1;

  std::string error_message;
  if (!app.renderer_core.init(options, &error_message)) {
    fprintf(stderr, "Renderer init failed: %s\n", error_message.c_str());
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }

  app.sceneCenter = app.renderer_core.focusCenter();
  app.sceneDiagonal = app.renderer_core.sceneDiagonal();
  const float initial_distance = app.renderer_core.focusDistance();
  printf("Initial camera focus: (%.3f, %.3f, %.3f)  distance: %.3f  "
         "scene diagonal: %.3f\n",
         app.sceneCenter[0], app.sceneCenter[1], app.sceneCenter[2], initial_distance, app.sceneDiagonal);

  app.orbit_cam.center = app.sceneCenter;
  app.orbit_cam.distance = std::max(0.05f, initial_distance);
  app.orbit_cam.yaw = -1.5707963f;
  app.orbit_cam.pitch = 0.f;

#ifdef GRUT_HAS_CUDA
  app.cuda_gl_interop = app.renderer_core.supportsCudaFrameBuffers() && app.check_cuda_gl_interop();
  if (app.cuda_gl_interop)
    printf("CUDA-GL interop enabled (zero-copy display path).\n");
  else
    printf("CUDA-GL interop unavailable -- using host readback display path.\n");
#else
  printf("Built without CUDA -- using host readback display path.\n");
#endif

  // ── Dear ImGui ────────────────────────────────────────────────────────────

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, false);
  ImGui_ImplOpenGL3_Init("#version 130");

  g_app = &app;

  glGenTextures(1, &app.frame_texture);
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glfwSetFramebufferSizeCallback(window, glfwResizeCb);
  glfwSetKeyCallback(window, glfwKeyCb);
  glfwSetCharCallback(window, glfwCharCb);
  glfwSetMouseButtonCallback(window, glfwMouseButtonCb);
  glfwSetCursorPosCallback(window, glfwCursorPosCb);
  glfwSetScrollCallback(window, glfwScrollCb);

  // Trigger initial resize
  {
    int fw, fh;
    glfwGetFramebufferSize(window, &fw, &fh);
    app.resize(fw, fh);
  }

  // Warm up + start async loop (same as reference constructor)
  app.push_camera();
  app.render_background();
  app.async_loop.start();

  // ── Main loop ─────────────────────────────────────────────────────────────

  auto lastTime = std::chrono::high_resolution_clock::now();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;

    app.updateKeyboardMovement(dt);
    app.push_camera();
    app.draw();

    glfwSwapBuffers(window);
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  // Wake the background thread so it exits the CV wait promptly.
  {
    std::lock_guard<std::mutex> lk(app.frame_handoff_mtx_);
    app.frame_consumed_ = true;
  }
  app.frame_consumed_cv_.notify_one();
  app.async_loop.stop();
#ifdef GRUT_HAS_CUDA
  app.unregister_frame_texture_cuda();
#endif

  glDeleteTextures(1, &app.frame_texture);

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
