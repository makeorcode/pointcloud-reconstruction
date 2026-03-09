# StaticScan

Offline point cloud reconstruction pipeline for a rotating-platform LiDAR (Velodyne VLP-16) using ROS 2 MCAP bag files. No ROS installation required.

The pipeline has three stages:

| Script | Purpose | Output |
|---|---|---|
| `offline_deskew.py` | Motion-deskew each scan frame using platform encoder angles | `.npz` |
| `icp_merge.py` | ICP-align angular slices into a single registered map | `.pcd` |
| `flythrough.py` | First-person OpenGL viewer for the resulting point cloud | — |

---

## Requirements

**Python 3.11 recommended** — Open3D does not support Python 3.13+.

### Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy mcap mcap-ros2-support zstandard open3d glfw PyOpenGL PyOpenGL_accelerate
```

| Package | Used by | Purpose |
|---|---|---|
| `numpy` | all | Array maths |
| `mcap` | `offline_deskew`, `icp_merge` | MCAP bag reading |
| `mcap-ros2-support` | `offline_deskew`, `icp_merge` | ROS 2 message decoding |
| `zstandard` | `offline_deskew`, `icp_merge` | Decompress `.mcap.zstd` bags |
| `open3d` | `icp_merge`, `flythrough` | ICP registration, PCD I/O |
| `glfw` | `flythrough` | OpenGL window and input |
| `PyOpenGL` | `flythrough` | OpenGL bindings |
| `PyOpenGL_accelerate` | `flythrough` | Optional C accelerators for PyOpenGL |

> **Note:** `PyOpenGL_accelerate` is optional but recommended for performance.

### Virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

---

## Expected bag topics

| ROS 2 topic | Message type | Description |
|---|---|---|
| `/velodyne_points` | `sensor_msgs/PointCloud2` | VLP-16 scan frames |
| `/rotating_platform/angle` | `std_msgs/Float64` | Platform encoder angle in degrees |

Compressed `.mcap.zstd` bags are automatically decompressed in-place on first use.

---

## Usage

### Quick start

```bash
# Reconstruct with ICP
python icp_merge.py /data/my_scan/

# View the result
python flythrough.py icp_merged.pcd
```

---

### 1. Deskew — `offline_deskew.py`

Corrects motion blur in each scan frame by interpolating the platform rotation angle at each point's acquisition time. Can be used standalone or is called automatically by `icp_merge.py`.

```
python offline_deskew.py <bag_path> [output.npz]
```

| Argument | Description |
|---|---|
| `bag_path` | Path to a ROS 2 bag directory or a `.mcap` / `.mcap.zstd` file |
| `output.npz` | Output file (default: `deskewed.npz`) |

**Examples**

```bash
# From a bag directory
python offline_deskew.py /data/my_scan/

# From a compressed MCAP file with a custom output name
python offline_deskew.py /data/my_scan/scan.mcap.zstd corrected.npz
```

The output `.npz` contains:

| Key | Description |
|---|---|
| `points` | Array of deskewed XYZ point arrays (one per scan frame) |
| `timestamps` | Nanosecond timestamps per frame |
| `angle_times` | Encoder sample timestamps |
| `angle_values` | Encoder angles in radians |

**Calibration parameters** — edit at the top of `offline_deskew.py`:

| Parameter | Default | Description |
|---|---|---|
| `ROTATION_AXIS` | `'x'` | Axis the platform rotates around (`'x'`, `'y'`, or `'z'`) |
| `ANGLE_OFFSET_DEG` | `184.0` | Zero-angle calibration offset in degrees |
| `ROTATION_CENTER` | `[0,0,0]` | Offset of LiDAR optical centre from rotation axis |
| `INVERT_ROTATION` | `False` | Flip rotation direction |
| `ENCODER_TIME_OFFSET_MS` | `0.0` | Time offset between encoder and LiDAR clocks |
| `MOUNT_RPY_DEG` | `[0,0,0]` | LiDAR mount roll / pitch / yaw in degrees |
| `MOUNT_AXES` | `['+x','+y','+z']` | Axis remapping for the LiDAR mount |

---

### 2. ICP Merge — `icp_merge.py`

Splits the full scan into overlapping angular slices, deskews each frame, then uses ICP to align all slices into a single registered point cloud. Uses a two-pass strategy (chain → refine → re-chain) to minimise drift.

ICP alignment is applied on top of the deskewing step to compensate for mechanical imperfections — shaft runout, bearing wobble, or other hardware tolerances that cause opposite 180° scan halves to not align perfectly even after motion correction. The angular-slice approach gives ICP overlapping geometry between adjacent slices, making registration robust where pure deskewing falls short.

```
python icp_merge.py <bag_path> [options]
```

| Argument / Flag | Default | Description |
|---|---|---|
| `bag_path` | — | Path to bag directory or `.mcap` file |
| `--slices N` | `12` | Number of angular slices |
| `--overlap F` | `2.0` | Slice window width as a multiple of the step width |
| `--fine` | off | Use finer voxel size (0.05 m) for ICP |
| `--min-fitness F` | `0.3` | Minimum ICP fitness score to accept a slice |
| `--no-chain` | off | Align every slice directly to the reference (no chaining) |
| `--map-voxel F` | `0.01` | Final output voxel downsample size in metres; `0` to disable |
| `--trim-end N` | `0` | Trim the last N seconds from the bag |
| `-o FILE` | `icp_merged.pcd` | Output PCD file path |

**Examples**

```bash
# Basic run with defaults
python icp_merge.py /data/my_scan/

# 16 slices, fine ICP, custom output
python icp_merge.py /data/my_scan/ --slices 16 --fine -o result.pcd

# Trim last 3 seconds (e.g. platform spin-down)
python icp_merge.py /data/my_scan/ --trim-end 3

# Disable final downsample to keep all points
python icp_merge.py /data/my_scan/ --map-voxel 0

# More overlap for difficult geometry
python icp_merge.py /data/my_scan/ --slices 12 --overlap 3.0
```

The pipeline prints four stages of progress:

1. Read bag — decoder encoder and point cloud messages
2. Deskew all scan frames
3. Build overlapping slices and ICP-align (chain → refine → re-chain)
4. Merge core points with aligned transforms and write PCD

---

### 3. Viewer — `flythrough.py`

First-person OpenGL viewer for `.pcd` or `.npz` point cloud files.

```
python flythrough.py <file1.pcd> [file2.pcd ...]
```

Pass a single file for normal viewing, or multiple files to enter **compare mode** — each cloud is rendered in a distinct colour (blue, orange, green, magenta) and the full stitching / alignment toolset becomes available.

**Example — compare and stitch two scans**

```bash
python flythrough.py scan_a.pcd scan_b.pcd
```

---

**Navigation controls**

| Key / Mouse | Action |
|---|---|
| W / S | Move forward / backward |
| A / D | Strafe left / right |
| Space / LCtrl | Move up / down |
| Left-click drag | Look around |
| Mid / Right-click drag | Pan |
| Scroll | Adjust move speed |
| + / - | Point size |
| C | Cycle color mode |
| E | Toggle Eye-Dome Lighting (EDL) |
| L / Shift+L | EDL strength up / down |
| R | Reset camera to centre |
| P | Print camera position |
| 1–5 | Speed presets |
| X | Statistical outlier filter |
| F12 | Save / overwrite original PCD |
| F5 / F6 | Rotate scene ±90° around X |
| F7 / F8 | Rotate scene ±90° around Y |
| F9 / F10 | Rotate scene ±90° around Z |
| F1 | Reset scene rotation |
| Esc | Quit |

---

**Compare mode — cloud visibility** *(2+ files)*

| Key | Action |
|---|---|
| F2 | Toggle cloud 1 visibility |
| F3 | Toggle cloud 2 visibility |
| F4 | Toggle cloud 3 visibility |

---

**Grab mode — interactive alignment** *(2+ files, Tab to enter/exit)*

Grab mode lets you manually position cloud 2 against cloud 1 before running ICP.

| Key / Mouse | Action |
|---|---|
| Tab | Toggle grab mode on / off (Esc to cancel and revert) |
| Left-click drag | Translate cloud 2 in the view plane |
| Shift + left-click drag | Rotate cloud 2 around its centroid |
| Scroll | Push / pull cloud 2 along camera forward axis |
| I | Run ICP to refine alignment (then exits grab mode) |

---

**Pick mode — correspondence-based alignment** *(2+ files)*

Pick at least 3 matching point pairs from each cloud. A Kabsch SVD initial transform is computed from the picks, then refined with ICP.

| Key / Mouse | Action |
|---|---|
| T | Toggle pick mode on / off |
| Left-click | Pick nearest point (alternates between cloud 1 and cloud 2) |
| U | Undo last pick |
| G | Compute alignment from current pairs (needs 3+) |
| M | Merge all clouds and save (prompts for filename) |

---

## Typical workflow

```bash
# 1. Reconstruct with ICP (deskewing is done automatically)
python icp_merge.py /data/my_scan/ --slices 12 -o map.pcd

# 2. View the result
python flythrough.py map.pcd

# 3. Optionally run deskew standalone for inspection
python offline_deskew.py /data/my_scan/ deskewed.npz
python flythrough.py deskewed.npz
```

---

## Platform notes

- Tested on **Windows 11** with Python 3.11.
- Should work on Linux and macOS. On Linux ensure system OpenGL and GLFW libraries are available (`libglfw3-dev`, `libgl1-mesa-dev`).
- Compressed `.mcap.zstd` bags are decompressed in-place beside the original file on first use.
