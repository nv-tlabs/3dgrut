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

// Interactive Gaussian Viewer -- GLFW + Dear ImGui + ANARI/VisRTX
// Architecture modeled after VIDILabs/open-volume-renderer main_app.cpp:
// async double-buffered rendering with TransactionalValue handoff.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/ext/visrtx/makeVisRTXDevice.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "gaussian_common.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
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
template <typename T>
class TransactionalValue {
public:
  TransactionalValue() = default;
  explicit TransactionalValue(const T &v) : committed_(v), staging_(v) {}

  TransactionalValue &operator=(const T &v) {
    std::lock_guard<std::mutex> lk(mtx_);
    staging_ = v;
    dirty_.store(true, std::memory_order_release);
    return *this;
  }

  template <typename Fn>
  void assign(Fn &&fn) {
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

  template <typename Fn>
  bool update(Fn &&fn) {
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
  double fps{0.0};

  bool count() {
    frame++;
    auto now = std::chrono::high_resolution_clock::now();
    if (frame % WINDOW == 0) {
      double elapsed =
          std::chrono::duration<double>(now - last_).count();
      fps = WINDOW / elapsed;
      last_ = now;
      return true;
    }
    return false;
  }

private:
  std::chrono::high_resolution_clock::time_point last_ =
      std::chrono::high_resolution_clock::now();
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Camera
// ═══════════════════════════════════════════════════════════════════════════════

struct CameraState {
  vec3 eye{0.f, 0.f, 0.f};
  vec3 dir{0.f, 0.f, 1.f};
  vec3 up{0.f, -1.f, 0.f};
  float aspect{16.f / 9.f};
};

enum class CameraMode { Orbit, Fly };

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
    vec3 eye = {center[0] + offset[0], center[1] + offset[1],
                center[2] + offset[2]};
    float len = std::sqrt(offset[0] * offset[0] + offset[1] * offset[1] +
                          offset[2] * offset[2]);
    vec3 dir = {-offset[0] / len, -offset[1] / len, -offset[2] / len};
    return {eye, dir, up, aspect};
  }

  void orbit(float dx, float dy) {
    yaw += dx;
    pitch = std::clamp(pitch + dy, -1.5f, 1.5f);
  }

  void pan(float dx, float dy) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    vec3 right = {-sy, 0.f, cy};
    vec3 camUp = {0.f, 1.f, 0.f};
    for (int i = 0; i < 3; i++)
      center[i] += right[i] * dx * distance * 0.002f +
                   camUp[i] * dy * distance * 0.002f;
  }

  void zoom(float delta) {
    distance *= (1.f - delta * 0.1f);
    if (distance < 0.01f) distance = 0.01f;
  }
};

struct FlyCamera {
  vec3 eye{0.f, 0.f, 0.f};
  float yaw{0.f};
  float pitch{0.f};
  float speed{1.f};
  vec3 up{0.f, -1.f, 0.f};

  CameraState state(float aspect) const {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    vec3 dir = {cy * cp, sp, sy * cp};
    return {eye, dir, up, aspect};
  }

  void look(float dx, float dy) {
    yaw += dx;
    pitch = std::clamp(pitch + dy, -1.5f, 1.5f);
  }

  void move(float forward, float right, float upAmount, float dt) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    vec3 fwd = {cy * cp, sp, sy * cp};
    vec3 r = {-sy, 0.f, cy};
    float s = speed * dt;
    for (int i = 0; i < 3; i++)
      eye[i] += (fwd[i] * forward + r[i] * right) * s;
    eye[1] += upAmount * s;
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Shared data structures between threads
// ═══════════════════════════════════════════════════════════════════════════════

struct FramePixels {
  uvec2 size{0, 0};
  std::vector<uint32_t> rgba;
};

struct RendererConfig {
  vec4 bgColor{0.1f, 0.1f, 0.1f, 1.f};
  float ambientRadiance{1.0f};
  int spp{1};
  bool dirty{true};
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Application
// ═══════════════════════════════════════════════════════════════════════════════

struct App {
  // ANARI objects (owned by background thread after init)
  anari::Device device{nullptr};
  anari::World world{nullptr};
  anari::Camera anariCamera{nullptr};
  anari::Renderer anariRenderer{nullptr};
  anari::Frame anariFrame{nullptr};

  // Double-buffered communication (same pattern as main_app.cpp)
  TransactionalValue<CameraState> camera_shared;
  TransactionalValue<uvec2> frame_size_shared{uvec2{0, 0}};
  TransactionalValue<FramePixels> frame_outputs;
  TransactionalValue<RendererConfig> renderer_config_shared;

  // Background thread
  AsyncLoop async_loop{std::bind(&App::render_background, this)};
  FPSCounter bg_fps;
  FPSCounter fg_fps;

  // GUI-thread local state
  GLFWwindow *window{nullptr};
  GLuint frame_texture{0};
  uvec2 fb_size{0, 0};

  CameraMode cam_mode{CameraMode::Orbit};
  OrbitCamera orbit_cam;
  FlyCamera fly_cam;
  bool camera_modified{true};

  bool gui_enabled{true};
  bool async_enabled{true};

  struct {
    float scaleFactor{1.0f};
    float bgColor[3]{0.1f, 0.1f, 0.1f};
    float ambientRadiance{1.0f};
    int spp{1};
    float lightPhi{225.f};
    float lightTheta{225.f};
    float lightIntensity{3.0f};
  } config;

  GaussianData gaussianData;
  float sceneDiagonal{1.f};
  vec3 sceneCenter{0.f, 0.f, 0.f};

  // Mouse state
  double lastMouseX{0}, lastMouseY{0};
  bool lmbDown{false}, rmbDown{false};

  // ─── Background thread ───────────────────────────────────────────────────

  void render_background() {
    bool size_changed = frame_size_shared.update();
    if (size_changed) {
      uvec2 sz = frame_size_shared.ref();
      if (sz[0] == 0 || sz[1] == 0)
        return;
      anari::setParameter(device, anariFrame, "size", sz);
      anari::commitParameters(device, anariFrame);
    }
    {
      uvec2 sz = frame_size_shared.ref();
      if (sz[0] == 0 || sz[1] == 0)
        return;
    }

    if (camera_shared.update()) {
      const auto &cam = camera_shared.ref();
      anari::setParameter(device, anariCamera, "position", cam.eye);
      anari::setParameter(device, anariCamera, "direction", cam.dir);
      anari::setParameter(device, anariCamera, "up", cam.up);
      anari::setParameter(device, anariCamera, "aspect", cam.aspect);
      anari::commitParameters(device, anariCamera);
    }

    if (renderer_config_shared.update()) {
      const auto &rc = renderer_config_shared.ref();
      anari::setParameter(device, anariRenderer, "background", rc.bgColor);
      anari::setParameter(device, anariRenderer, "ambientRadiance",
                          rc.ambientRadiance);
      anari::setParameter(device, anariRenderer, "pixelSamples", rc.spp);
      anari::commitParameters(device, anariRenderer);
    }

    anari::render(device, anariFrame);
    anari::wait(device, anariFrame);

    auto fb =
        anari::map<uint32_t>(device, anariFrame, "channel.color");

    FramePixels pixels;
    pixels.size = frame_size_shared.ref();
    size_t count = static_cast<size_t>(fb.width) * fb.height;
    pixels.rgba.assign(fb.data, fb.data + count);
    anari::unmap(device, anariFrame, "channel.color");

    frame_outputs = pixels;

    bg_fps.count();
  }

  // ─── GUI thread: push camera ─────────────────────────────────────────────

  void push_camera() {
    if (!camera_modified)
      return;
    camera_modified = false;

    float aspect =
        fb_size[1] > 0 ? float(fb_size[0]) / float(fb_size[1]) : 1.f;
    CameraState cs;
    if (cam_mode == CameraMode::Orbit)
      cs = orbit_cam.state(aspect);
    else
      cs = fly_cam.state(aspect);

    camera_shared = cs;

    if (!async_enabled)
      render_background();
  }

  // ─── GUI thread: draw ────────────────────────────────────────────────────

  void draw() {
    // Consume latest rendered pixels from background thread
    frame_outputs.update([&](const FramePixels &px) {
      if (px.size[0] == 0 || px.size[1] == 0)
        return;
      glBindTexture(GL_TEXTURE_2D, frame_texture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, px.size[0], px.size[1], 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, px.rgba.data());
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    });

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
      ImGui::SetNextWindowSizeConstraints(ImVec2(360, 300),
                                          ImVec2(FLT_MAX, FLT_MAX));
      if (ImGui::Begin("Control Panel", nullptr)) {
        ImGui::Text("Mode: %s",
                     cam_mode == CameraMode::Orbit ? "Orbit (I)" : "Fly (F)");
        ImGui::Separator();

        bool renderer_dirty = false;

        if (ImGui::ColorEdit3("Background", config.bgColor)) {
          renderer_dirty = true;
        }
        if (ImGui::SliderFloat("Ambient Radiance", &config.ambientRadiance,
                                0.f, 5.f, "%.2f")) {
          renderer_dirty = true;
        }
        if (ImGui::SliderInt("Samples Per Pixel", &config.spp, 1, 64)) {
          renderer_dirty = true;
        }

        if (renderer_dirty) {
          RendererConfig rc;
          rc.bgColor = {config.bgColor[0], config.bgColor[1],
                        config.bgColor[2], 1.f};
          rc.ambientRadiance = config.ambientRadiance;
          rc.spp = config.spp;
          renderer_config_shared = rc;
        }

        ImGui::Separator();
        ImGui::SliderFloat("Light Phi", &config.lightPhi, 0.f, 360.f,
                           "%.1f");
        ImGui::SliderFloat("Light Theta", &config.lightTheta, 0.f, 360.f,
                           "%.1f");
        ImGui::SliderFloat("Light Intensity", &config.lightIntensity, 0.f,
                           10.f, "%.2f");

        ImGui::Separator();
        if (ImGui::SliderFloat("Gaussian Scale", &config.scaleFactor, 0.01f,
                                10.f, "%.3f")) {
          rebuildScene();
        }

        ImGui::Separator();
        ImGui::Text("Gaussians: %zu", gaussianData.positions.size());
        ImGui::Text("BG FPS: %.1f", bg_fps.fps);
        ImGui::Text("FG FPS: %.1f", fg_fps.fps);
      }
      ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // FPS in title bar (same as reference)
    if (fg_fps.count()) {
      std::stringstream title;
      title << std::fixed << std::setprecision(1)
            << "Interactive Gaussian Viewer  |  fg=" << fg_fps.fps
            << " fps  bg=" << bg_fps.fps << " fps";
      glfwSetWindowTitle(window, title.str().c_str());
    }
  }

  // ─── Scene rebuild ───────────────────────────────────────────────────────

  void rebuildScene() {
    if (async_enabled)
      async_loop.stop();

    anari::release(device, world);
    world = buildScene(device, gaussianData, config.scaleFactor);
    anari::setParameter(device, anariFrame, "world", world);
    anari::commitParameters(device, anariFrame);
    camera_modified = true;

    if (async_enabled) {
      render_background();
      async_loop.start();
    }
  }

  // ─── Resize ──────────────────────────────────────────────────────────────

  void resize(int w, int h) {
    if (w <= 0 || h <= 0)
      return;
    fb_size = {(unsigned)w, (unsigned)h};
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
      const auto &px = frame_outputs.get();
      if (px.size[0] > 0 && px.size[1] > 0) {
        stbi_flip_vertically_on_write(1);
        stbi_write_png("screenshot.png", px.size[0], px.size[1], 4,
                       px.rgba.data(), 4 * px.size[0]);
        printf("Screenshot saved: screenshot.png\n");
      }
      break;
    }
    case GLFW_KEY_I:
      printf("Entering orbit (inspect) mode\n");
      cam_mode = CameraMode::Orbit;
      break;
    case GLFW_KEY_F:
      printf("Entering fly mode\n");
      if (cam_mode != CameraMode::Fly) {
        float aspect =
            fb_size[1] > 0 ? float(fb_size[0]) / float(fb_size[1]) : 1.f;
        auto cs = orbit_cam.state(aspect);
        fly_cam.eye = cs.eye;
        fly_cam.yaw = orbit_cam.yaw + 3.14159265f;
        fly_cam.pitch = -orbit_cam.pitch;
        fly_cam.speed = orbit_cam.distance;
        fly_cam.up = orbit_cam.up;
      }
      cam_mode = CameraMode::Fly;
      break;
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

    if (cam_mode == CameraMode::Orbit) {
      if (lmbDown) {
        orbit_cam.orbit(dx * 0.005f, dy * 0.005f);
        camera_modified = true;
      }
      if (rmbDown) {
        orbit_cam.pan(dx, dy);
        camera_modified = true;
      }
    } else {
      if (rmbDown) {
        fly_cam.look(dx * 0.003f, -dy * 0.003f);
        camera_modified = true;
      }
    }
  }

  void onScroll(double /*xoffset*/, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse)
      return;
    if (cam_mode == CameraMode::Orbit) {
      orbit_cam.zoom(float(yoffset));
      camera_modified = true;
    }
  }

  // ─── Fly-mode per-frame movement ────────────────────────────────────────

  void updateFlyMovement(float dt) {
    if (cam_mode != CameraMode::Fly)
      return;
    float fwd = 0.f, right = 0.f, up = 0.f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) fwd += 1.f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) fwd -= 1.f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) right += 1.f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) right -= 1.f;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) up += 1.f;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) up -= 1.f;
    if (fwd != 0.f || right != 0.f || up != 0.f) {
      fly_cam.move(fwd, right, up, dt);
      camera_modified = true;
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  GLFW callbacks (forward to App)
// ═══════════════════════════════════════════════════════════════════════════════

static App *g_app = nullptr;

static void glfwResizeCb(GLFWwindow *, int w, int h) { g_app->resize(w, h); }
static void glfwKeyCb(GLFWwindow *, int k, int sc, int a, int m) {
  if (!ImGui::GetIO().WantCaptureKeyboard)
    g_app->onKey(k, sc, a, m);
}
static void glfwMouseButtonCb(GLFWwindow *, int b, int a, int m) {
  g_app->onMouseButton(b, a, m);
}
static void glfwCursorPosCb(GLFWwindow *, double x, double y) {
  g_app->onCursorPos(x, y);
}
static void glfwScrollCb(GLFWwindow *, double x, double y) {
  g_app->onScroll(x, y);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

static void printUsage(const char *argv0)
{
  printf("Usage: %s <path.ply> [options]\n", argv0);
  printf("  --scale-factor F        Multiply Gaussian scales (default: 1.0)\n");
  printf("  --opacity-threshold T   Min opacity to keep (default: 0.05)\n");
  printf("  --resolution WxH        Window resolution (default: 1920x1080)\n");
  printf("\nControls:\n");
  printf("  LMB drag    Orbit\n");
  printf("  RMB drag    Pan (orbit) / Look (fly)\n");
  printf("  Scroll      Zoom\n");
  printf("  WASD/QE     Move (fly mode)\n");
  printf("  I           Orbit mode\n");
  printf("  F           Fly mode\n");
  printf("  G           Toggle GUI\n");
  printf("  S           Screenshot\n");
  printf("  Escape      Quit\n");
}

int main(int argc, char *argv[])
{
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string plyPath = argv[1];
  float scaleFactor = 1.0f;
  float opacityThreshold = 0.05f;
  uvec2 winSize = {1920, 1080};

  for (int i = 2; i < argc; i++) {
    if (std::strcmp(argv[i], "--scale-factor") == 0 && i + 1 < argc)
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

  // ── Load PLY ──────────────────────────────────────────────────────────────

  auto data = loadPLY(plyPath, opacityThreshold);
  if (data.positions.empty()) {
    fprintf(stderr, "No Gaussians survived filtering.\n");
    return 1;
  }

  vec3 center;
  float diagonal;
  {
    for (int ax = 0; ax < 3; ax++)
      center[ax] = (data.bboxMin[ax] + data.bboxMax[ax]) * 0.5f;
    float dx = data.bboxMax[0] - data.bboxMin[0];
    float dy = data.bboxMax[1] - data.bboxMin[1];
    float dz = data.bboxMax[2] - data.bboxMin[2];
    diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  printf("Scene center: (%.3f, %.3f, %.3f)  diagonal: %.3f\n",
         center[0], center[1], center[2], diagonal);

  // ── GLFW + OpenGL ─────────────────────────────────────────────────────────

  if (!glfwInit()) {
    fprintf(stderr, "Failed to init GLFW\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

  GLFWwindow *window = glfwCreateWindow(winSize[0], winSize[1],
                                         "Interactive Gaussian Viewer",
                                         nullptr, nullptr);
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
    glfwTerminate();
    return 1;
  }
  printf("OpenGL %d.%d loaded\n", GLAD_VERSION_MAJOR(version),
         GLAD_VERSION_MINOR(version));

  // ── Dear ImGui ────────────────────────────────────────────────────────────

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 130");

  // ── ANARI / VisRTX ────────────────────────────────────────────────────────

  auto device = makeVisRTXDevice(anariStatusFunc);

  auto anariWorld = buildScene(device, data, scaleFactor);

  auto anariCamera = anari::newObject<anari::Camera>(device, "perspective");
  anari::commitParameters(device, anariCamera);

  auto anariRenderer =
      anari::newObject<anari::Renderer>(device, "default");
  vec4 bgColor = {0.1f, 0.1f, 0.1f, 1.f};
  anari::setParameter(device, anariRenderer, "background", bgColor);
  anari::setParameter(device, anariRenderer, "ambientRadiance", 1.0f);
  anari::setParameter(device, anariRenderer, "pixelSamples", 1);
  anari::commitParameters(device, anariRenderer);

  auto anariFrame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, anariFrame, "size", winSize);
  anari::setParameter(device, anariFrame, "channel.color",
                      ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, anariFrame, "world", anariWorld);
  anari::setParameter(device, anariFrame, "camera", anariCamera);
  anari::setParameter(device, anariFrame, "renderer", anariRenderer);
  anari::commitParameters(device, anariFrame);

  // ── App state ─────────────────────────────────────────────────────────────

  App app;
  app.device = device;
  app.world = anariWorld;
  app.anariCamera = anariCamera;
  app.anariRenderer = anariRenderer;
  app.anariFrame = anariFrame;
  app.window = window;
  app.gaussianData = std::move(data);
  app.sceneDiagonal = diagonal;
  app.sceneCenter = center;
  app.config.scaleFactor = scaleFactor;

  app.orbit_cam.center = center;
  app.orbit_cam.distance = 0.3f * diagonal;
  app.orbit_cam.yaw = 0.f;
  app.orbit_cam.pitch = 0.f;

  app.fly_cam.speed = diagonal * 0.5f;

  g_app = &app;

  glGenTextures(1, &app.frame_texture);
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glfwSetFramebufferSizeCallback(window, glfwResizeCb);
  glfwSetKeyCallback(window, glfwKeyCb);
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
  app.render_background();
  app.async_loop.start();

  // ── Main loop ─────────────────────────────────────────────────────────────

  auto lastTime = std::chrono::high_resolution_clock::now();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    auto now = std::chrono::high_resolution_clock::now();
    float dt =
        std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;

    app.updateFlyMovement(dt);
    app.push_camera();
    app.draw();

    glfwSwapBuffers(window);
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  app.async_loop.stop();

  glDeleteTextures(1, &app.frame_texture);

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  anari::release(device, anariCamera);
  anari::release(device, anariRenderer);
  anari::release(device, anariWorld);
  anari::release(device, anariFrame);
  anari::release(device, device);

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
