
<img src="../assets/playground_glass.gif" align="center" style="display: block; margin: 0 auto;"/>

# The Playground  

**3D Gaussian Ray Tracing (3DGRT)** enables the rendering of secondary ray effects such as reflections, refractions, depth of field and others, as well as inserting ray-traced mesh assets with materials into the scene.

The **Playground** is an interactive demo app that showcases various effects in action.  See examples published with the paper on the 3DGRT [Project page](https://research.nvidia.com/labs/toronto-ai/3DGRT).

## 🔥 News

- ✅[2025/06] 3DGRUT's playground engine is featured in [Kaolin's CVPR 2025 Tutorial](https://kaolin.readthedocs.io/en/latest/notes/cvpr2025.html).
- ✅[2025/06] Playground v2.0 releaed: Path tracing of PBR meshes and environment maps added.
- ✅[2025/04] Headless mode added (Engine3DGRUT is now exposed as api).
- ✅[2025/03] Playground v1.0 released (3dgrt + Glass, Mirrors, Diffuse Meshes, Depth of Field)


## Contents

- [The Playground](#-the-playground)  
  - [🔥 News](#-news)
  - [Contents](#-contents)
  - [🔧 Installation](#-installation)
  - [🏃🏻 How to Run 🏃🏼‍♀️](#-how-to-run)
    - [Headless Mode](#-headless-mode) 
    - [👻 Add your own assets](#-add-your-own-assets)
    - [Additional args️](#-additional-args)
  - [How it works️](#-how-it-works)
  - [🎮 Features ️](#-features)
    - [🌐 Mesh Primitives](#-mesh-primitives)
      - [🫙 Glass](#-glass)
      - [🪞 Mirror](#-mirror)
      - [🔵 Diffused Mesh](#-diffused-mesh)
      - [💎 PBR Mesh](#-pbr-mesh)
    - [🖌️ Materials](#-materials)
    - [🌇️ Environment Maps](#-environment-maps)
    - [⚙️ Quick Settings](#-quick-settings)
    - [📸 Depth of Field](#-depth-of-field)
    - [✨ Antialiasing](#-antialiasing)
      - [🤖 Optix Denoiser](#-optix-denoiser)
    - [🛠️ Render Definitions](#-render-definitions)
    - [Other Features](#-other-features)
  - [🚀 Future Content](#-future-content)


## 🔧 Installation

1. [Install 3dgrut](../README.md), following instructions on main page.
2. Install additional Playground requirements:

```bash
conda install -c conda-forge mesa-libgl-devel-cos7-x86_64 # may be necessary for OpenGL headers
pip install -r threedgrut_playground/requirements.txt
```

3. Download a pack of interesting mesh assets and env maps:
```bash
chmod +x ./threedgrut_playground/download_assets.sh
./threedgrut_playground/download_assets.sh
```

## 🏃🏻 How to Run 🏃🏼‍♀️

1. Train a 3DGRT scene, for example:
```bash
python train.py --config-name apps/colmap_3dgrt.yaml path=data/mipnerf360/bonsai out_dir=runs experiment_name=bonsai dataset.downsample_factor=2
```

2. Run playground as follows:
```bash
python playground.py --gs_object runs/bonsai/ckpt_last.pt
```

The playground supports loading `.pt` checkpoints, and exported `.ingp` and `.ply` files.

### Headless Mode
If you're running on a remote machine without a screen, a minimal version of the Playground is
available through Jupyter notebook: `threedgrut_playground/headless.ipynyb`.

The playground functionality is exposed through the main engine file:
`threedgrut_playground/engine.py`.

A more mature example of the Jupyter GUI is available as an [NVIDIA-kaolin tutorial](https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/physics/simulatable_3dgrut.ipynb).

In addition, a simple viser based GUI is available as a contribution from the community (by @tangkangqi):
```bash
python threedgrut_playground/viser_gui.py --gs_object=<CHECKPOINT>
```
Please note that the viser version does not expose most feature controls of the playground.
Further contributions to `threedgrut_playground/visor_gui.py` are welcome!

### 👻 Add your own assets
3. If desired, gather your own additional mesh assets (`.obj`, `.glb`, `.gltf` formats), and place them under `threedgrut_playground/assets/`.
The playground will load them automatically as available *primitives* as soon as the app starts.
Some interesting shapes are [available here](https://github.com/alecjacobson/common-3d-test-models/tree/master).
A subset of those are downloaded with the `download_assets.sh` script.

As of June 2025, the script will also download some sample `.hdr` env-maps, which
can be used to provide global light to PBR meshes.

5. Have fun experimenting! 

#### Additional args

```
python playground.py --gs_object <ckpt_path>  
                     [--mesh_assets <mesh_folder_path>]
                     [--envmap_assets <hdr_folder_path>]
                     [--default_gs_config <config_name>]
                     [--buffer_mode <"host2device" | "device2device">]
```
The full run command includes the following optional args:
* `--mesh_assets`: Path to folder containing mesh assets of .obj or .glb format. 
  * Defaults to `threedgrut_playground/assets`.
* `--envmap_assets`: Path to folder containing environment maps of .hdr format.
  * Defaults to `threedgrut_playground/assets`.
* `--default_gs_config`: Name of default config to use for .ingp, .ply files, or .pt files not trained with 3dgrt.
  * Defaults to `apps/colmap_3dgrt.yaml`.
* `--device2device`: Buffering mode for passing rendered data from CUDA to OpenGL screen buffer. Using device2device is recommended.
  * Defaults to `device2device`.

## How it Works

The Playground performs hybrid rendering of Gaussian particles and surface mesh primitive via ray tracing.

For each rendered frame:
1. A ray is traced from ray origin $\mathbf{r}_o$ against a BVH containing all mesh primitives in the scene.
Upon closest hit at point $\mathbf{x} \in \mathbb{R}^3$ , the ray may get redirected or shaded, depending on the surface mesh 
properties. These changes are memorized but not applied yet.
2. A 3dgrt volumetric integration phase runs for the segment of $[mathbf{r}_o, \mathbf{x}]$, to accumulate radiance.
3. If any radiance was contributed by the mesh, it is now taken into consideration. 
4. Loop back to step 1: the ray continues tracing using $\mathbf{x}$ as the new ray origin,
depending on redirection computed in step 1.

The process continues until the ray misses or accumulates enough radiance.
When a ray misses, if env maps are enabled, they will also contribute radiance to the ray. 
That is - both env maps and gaussians are used to light PBR primitives in the scene.

For antialiasing and depth of field, multiple rendering passes are used with ray jittering.

## 🎮 Features 

We invite you to explore the Playground by experimenting around!
Alternatively, in the following we discuss various features in depth.

<img src="../assets/playground_menu.png" align="center" height="500" style="display: block; margin: 0 auto;"/>

### 🌐 Mesh Primitives

The *Primitives* subsection allows adding, removing and duplicating different geometries.

The available geometries always include the default Quad and Sphere, and optionally other meshe files
placed under `threedgrut_playground/assets`

The Playground is always loaded with a default Sphere primitive placed at the origin.

*Add Primitive* adds additional primitives at the origin. 

The *Transform* subsection allows translating,
rotating and scaling primitives to place and orient them around the scene.

Changing the primitive *Type* immediately modifies how rays interact with the mesh.

#### 🫙 Glass
Glass type primitives follow [Snell's law](https://en.wikipedia.org/wiki/Snell%27s_law) and refract rays that hit them,
essentially diverting their direction to show 3D Gaussians and meshes beyond the glass. 

Glass objects allow changing the *Refractive Index (IOR)* to manipulate the "thickness" of the mesh medium.

#### 🪞 Mirror
Mirror type primitives act as perfect mirrors that reflect the rays around the mesh normal.

#### 🔵 Diffused Mesh
Diffused Meshes use Lambertian shading, to render meshes with diffusive materials. As **3DGRT** uses baked light,
we set the incoming light intensity to $L_{I}=1.0$ and compute the ray color $C$ as:
$$C=\mathopen|\mathbb{n}\mathclose| \mathopen|\mathbb{r}_d\mathclose| * D$$

where $n$ is the hit mesh normal, $\mathbb{r}_d$ is the normalized ray direction and $D$ is the
diffuse mesh color.

Diffused Mesh primitives can assign different materials.

#### 💎 PBR Mesh
PBR Meshes use Cook-Torrance shading, to render meshes with Physically Based materials.
The path tracer supports BRDF and BTDF of meshes alongside volumetric radiance intergration of Gaussian fields.

The approach taken by this path tracer is pragmatic, 
rather than principled, as hybrid rendering is an open research problem:
Here, 3DGRT particles are treated as radiating particles, while meshes absorb and scatter light.
Note that 3DGRT uses baked light for radiance, and therefore Gaussian particles are not affected by light sources.
In addition, envmaps assume HDR, which may require some tuning to match the rest of the scene.

PBR Meshes rely on Monte Carlo for high quality rendering, and therefore require antialiasing toggled on.
Quality improves with the number of SPPs.

PBR Mesh primitives can assign different materials, and their properties may be edited in the materials section.

### 🖌️ Materials

The materials section includes a property editor for all loaded materials in the scene.

By default, a *solid* and *checkboard* materials should always be available (the latter using a texture for diffuse color).

If `.gltf` / `.glb` files are loaded with additional materials, these materials would appear under this menu.

### 🌇️ Environment Maps

Environment maps can be loaded to provide global light in the scene, and override the model background.

When the engine loads, a list of available env maps is populated. Then envmaps can be selected from the dropdown.

Note that env maps are only enabled when path tracing is enabled, and at least 1 mesh primitive is added to the scene.

Env maps provide light in High Dynamic Range (HDR). The IBL Intensity (Image Based Lighting Intensity) and Exposure
allow to scale the range up and down, to adjust the light according to the selected env map and scene.

IBL Intensity is a linear scalar multiplier applied to the env map before path tracing.

After the path tracer runs and an image is rendered, Exposure is applied with $$hdr=hdr*2^{exposure},$$
just before tone mapping takes place.

Tone mapping, if enabled, is applied next, followed by gamma correction.

The position of the env map can also be adjusted by the offset sliders.

### ⚙️ Quick Settings

At the top of the menu, the Playground includes presets that allow to quickly toggle between various configurations for tradeoff between speed and quality:

* *Fast*: Turns off antialiasing and the Optix Denoiser.
* *Balanced*: Uses 4x MSAA antialiasing and toggles on the Optix Denoiser.
* *High Quality*: Uses Sobol antialiasing with 64 SPP, and toggles on the Optix Denoiser.

### 📸 Depth of Field

[Depth of Field](https://en.wikipedia.org/wiki/Depth_of_field), demonstrates a lens effect that
renders with blur outside the focus region.

Depth of Field is applied to all 3D Gaussians and mesh primitives in the scene.

The *Samples per Pixel (SPP)*, *Aperture Size* and *Focus Z* control the strength of this effect.

### ✨ Antialiasing

Certain rendering effects such as refractions at extreme angles may introduce noisy pixel artifacts. 
The playground ships 4 antialiasing modes to sample additional samples per pixel and mitigate this effect:

* *4x MSAA, 8x MSAA, 16x MSAA* - use **Multisampling Antialiasing** with a predetermined pattern. Source: [Ray Tracing Gems II](https://link.springer.com/book/10.1007/978-1-4842-7185-8).
* *Sobol* - uses a low discrepancy sequence to sample an arbitrary number of samples.  

#### 🤖 Optix Denoiser

Optix includes a built-in learned denoiser that post-processes noisy images.
This option is exposed through the menu under *Render > Use Optix Denoiser*.

The Optix denoiser can be used with or without antialiasing toggled on.

### 🛠️ Render Definitions

The *Render* subsection includes the following additional definitions:
* *Style*: toggles between rendered *color* and *density* channels.
* *Camera*: toggles between *Pinhole* and *Fisheye* lens.
* *Gamma Correction*: applies gamma correction over the entire rendered scene (Gaussians + Mesh)
* *Max PBR Bounces*: limits the number of maximum redirections a mesh can have (i.e. due to refractions, reflections)
before it is terminated. This number controls a tradeoff of quality and speed.

### Other Features

* 🎥 *Record Trajectory Video* exposes options for placing key-cameras and rendering a continuous video of
the camera moving along a path.
* ✂️ The *Slice Planes* menu allows to enable / disable / position 6 slicing planes for trimming the edge of the scene.