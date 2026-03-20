# Collecting Point Cloud Data with the Unitree Go2

Guide for recording Go2 L1 LiDAR data to MCAP using the
[go2_ros2_sdk](https://github.com/abizovnuralem/go2_ros2_sdk)
and viewing it with `mcap_viewer.py` and `mcap_viewer_tf.py`.

---

## Prerequisites

- Ubuntu 22.04 with ROS 2 Humble (or Iron/Rolling)
- Unitree Go2 PRO/EDU (has the L1 LiDAR)
- Python 3.11 (required by open3d)
- go2_ros2_sdk built and working

### 1. Build the Go2 ROS2 SDK

```bash
mkdir -p ~/ros2_ws
cd ~/ros2_ws
git clone --recurse-submodules https://github.com/abizovnuralem/go2_ros2_sdk.git src

sudo apt install ros-$ROS_DISTRO-image-tools ros-$ROS_DISTRO-vision-msgs
sudo apt install python3-pip clang portaudio19-dev

cd src && pip install -r requirements.txt && cd ..

source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

### 2. Install MCAP storage (Humble only)

Humble defaults to SQLite3 bags. Install the MCAP plugin so you can record
in MCAP format directly:

```bash
sudo apt install ros-humble-rosbag2-storage-mcap
```

Iron and later use MCAP by default — no extra install needed.

### 3. Install this viewer's dependencies

```bash
cd /path/to/pointcloud-reconstruction
pip install -r requirements.txt
```

---

## SDK topic and frame reference

The go2_ros2_sdk publishes these PointCloud2 topics:

| Topic | Frame ID | Source | Notes |
|-------|----------|--------|-------|
| `point_cloud2` | `odom` | go2_driver_node | Raw LiDAR data, already in odom frame |
| `/pointcloud/aggregated` | (preserved) | lidar_to_pointcloud | Accumulated points |
| `/pointcloud/filtered` | `base_link` | pointcloud_aggregator | Filtered + downsampled |

TF tree (from URDF):
```
map ─── odom ─── base_link ─┬─ Head_upper ─── front_camera
                             ├─ imu
                             ├─ radar
                             └─ [leg joints]
```

SLAM Toolbox config: `scan_topic: /scan`, `base_frame: base_link`,
`odom_frame: odom`, `map_frame: map`.

---

## Step 1: Launch the Go2

The SDK has two launch files. Either works for recording.

**Standard launch** (Python LiDAR processing + SLAM + Nav2 + RViz2):

```bash
cd ~/ros2_ws
source install/setup.bash

export ROBOT_IP="192.168.x.x"    # your Go2's IP (check mobile app)
export CONN_TYPE="webrtc"         # or "cyclonedds" for Ethernet

ros2 launch go2_robot_sdk robot.launch.py
```

**Mapping-optimized launch** (better for point cloud capture — no
downsampling, wider height range, 20 Hz publish rate):

```bash
ros2 launch go2_robot_sdk mapping.launch.py
```

> Close the Unitree mobile app before connecting via WebRTC.

RViz2 opens automatically with the robot model, point cloud, laser scan,
camera feed, and SLAM map.

---

## Step 2: Verify data is flowing

```bash
# In a new terminal
source ~/ros2_ws/install/setup.bash

# Check point cloud topics
ros2 topic list | grep -i point

# You should see:
#   /point_cloud2               (raw, frame: odom)
#   /pointcloud/aggregated      (accumulated)
#   /pointcloud/filtered        (filtered, frame: base_link)

# Check rate on the topic you want to record
ros2 topic hz /point_cloud2
```

### Which topic to record?

| Topic | Best for | Frame |
|-------|----------|-------|
| `point_cloud2` | Raw data, most complete | `odom` |
| `/pointcloud/filtered` | Pre-filtered, less noise | `base_link` |

**Recommendation:** Record `point_cloud2` for the most data. The viewer
scripts handle filtering on playback.

---

## Step 3: Record to MCAP

```bash
# Recommended — point cloud + TF for aligned viewing
ros2 bag record \
  point_cloud2 \
  /tf \
  /tf_static \
  --storage mcap \
  -o my_scan

# Full recording — includes everything for later analysis
ros2 bag record \
  point_cloud2 \
  /pointcloud/filtered \
  /scan \
  /map \
  /odom \
  /tf \
  /tf_static \
  --storage mcap \
  -o my_scan
```

> On Iron/Jazzy, you can omit `--storage mcap` (it's the default).

### While recording

- **Drive the Go2** with the joystick to explore the space
- Watch the SLAM map build in RViz2
- The point cloud accumulates in parallel
- Press `Ctrl+C` to stop recording

This creates a `my_scan/` directory containing the `.mcap` file.

### Optional: save the SLAM map

Before stopping, save the 2D map from RViz2's SlamToolboxPlugin:

1. Enter a filename in "Save Map"
2. Click "Save Map", then "Serialize Map"

This gives you `.yaml`, `.pgm`, `.data`, and `.posegraph` files in `~/ros2_ws`.

---

## Step 4: View the raw point cloud

Start with `mcap_viewer.py` to quickly check the data. This stacks all
frames as-is — no transform corrections applied.

```bash
cd /path/to/pointcloud-reconstruction
source .venv/bin/activate

# List all topics in the recording
python mcap_viewer.py /path/to/data/my_scan/ --list-topics

# View — auto-detects the PointCloud2 topic
python mcap_viewer.py /path/to/data/my_scan/

# Or specify the topic explicitly
python mcap_viewer.py /path/to/data/my_scan/ --topic /point_cloud2
```

> **Note on the `point_cloud2` topic:** The SDK publishes points already
> transformed into the `odom` frame, so the raw view may actually look
> partially aligned even without TF correction. However, the odom frame
> drifts over time, so longer recordings will show increasing misalignment.
> Step 5 corrects this using the full TF tree.

### Handling large recordings

At 7 Hz, a 5-minute recording = ~2,100 frames:

```bash
# Use every 3rd frame + 5cm voxel downsample
python mcap_viewer.py /path/to/data/my_scan/ --every 3 --voxel 0.05

# Limit to first 100 frames
python mcap_viewer.py /path/to/data/my_scan/ --max-frames 100
```

---

## Step 5: View the cleaned-up point cloud (TF-aligned)

`mcap_viewer_tf.py` reads the recorded `/tf` and `/tf_static` transforms
to place each frame in the correct world position, then applies statistical
outlier removal and voxel downsampling.

**This is why Step 3 records `/tf` and `/tf_static`** — they contain the
robot's pose at every timestamp.

```bash
# 1. Process and save to PCD (recommended: 2cm voxel for good density)
python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --voxel 0.02 --save /path/to/data/cleaned.pcd

# 2. View the result
python flythrough.py /path/to/data/cleaned.pcd
```

# USE THESE COMMANDS
```bash
source .venv/bin/activate && python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --voxel 0.02 --save /path/to/data/cleaned_fine.pcd

python flythrough.py /path/to/data/cleaned_fine.pcd
```

Other useful variations:

```bash
# Quick preview with fewer frames
python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --every 5 --voxel 0.02 --save /path/to/data/preview.pcd

# Maximum density (all 43M+ points, no filtering — may be slow)
python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --no-cleanup --save /path/to/data/full.pcd

# Use odom frame instead of map (no SLAM correction, but no drift jumps)
python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --fixed-frame odom --voxel 0.02 --save /path/to/data/cleaned_odom.pcd

# Tune cleanup parameters
python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --voxel 0.03 --sor-neighbors 30 --sor-std 1.5 --save /path/to/data/tuned.pcd

# View directly without saving (launches flythrough viewer immediately)
python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --voxel 0.02
```

### Fixed frame choices

| Frame | What it does | Best for |
|-------|-------------|----------|
| `map` | SLAM-corrected world position | Best alignment, loop-closed |
| `odom` | Dead-reckoning position | Smooth but drifts over time |

The script auto-detects `map` if available, falling back to `odom`.

### Cleanup options

| Flag | Default | Description |
|------|---------|-------------|
| `--fixed-frame` | auto (`map` > `odom`) | World frame to transform into |
| `--voxel` | `0.05` | Voxel downsample size in meters |
| `--sor-neighbors` | `20` | Statistical outlier removal: neighbor count |
| `--sor-std` | `2.0` | Statistical outlier removal: std deviation ratio |
| `--no-cleanup` | off | Skip voxel + outlier filtering |

### If the TF transform fails

If you see `Cannot transform 'odom' -> 'map'`, the TF tree might not
include a map frame (e.g., SLAM wasn't running):

```bash
# The script prints available frames automatically
# Try odom instead:
python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --fixed-frame odom --voxel 0.02 --save /path/to/data/cleaned.pcd
```

If no TF data was recorded (forgot to include `/tf` in the bag), fall back
to the raw viewer:

```bash
python mcap_viewer.py /path/to/data/my_scan/ --topic /point_cloud2
```

---

## Viewer controls (flythrough.py)

| Key / Mouse | Action |
|---|---|
| W / A / S / D | Move forward / backward / strafe |
| Space / LCtrl | Move up / down |
| Left-click drag | Look around |
| Mid / Right-click drag | Pan |
| Scroll | Adjust move speed |
| 1–5 | Speed presets |
| + / - | Point size (increase to fill gaps) |
| Q | Toggle point shape (circle / square) |
| C | Cycle color mode (height / distance / white) |
| E | Toggle Eye-Dome Lighting (EDL) |
| X | Statistical outlier filter (good for noisy data) |
| R | Reset camera to centre |
| F12 | Save / overwrite original PCD |
| Esc | Quit |

**Tips:**
- Press **Q** to switch to square points — fills gaps between points
- Press **+** a few times to increase point size until gaps disappear
- **C** cycles through color modes — height mode uses a turbo colormap
  that clearly shows elevation changes
- Press **X** if performance is slow — it removes noisy outlier points

---

## Quick reference

All viewer commands assume you have activated the venv first:

```bash
cd /path/to/pointcloud-reconstruction
source .venv/bin/activate
```

| What | Command |
|------|---------|
| Launch Go2 | `ros2 launch go2_robot_sdk robot.launch.py` |
| Launch (mapping mode) | `ros2 launch go2_robot_sdk mapping.launch.py` |
| Verify LiDAR | `ros2 topic hz /point_cloud2` |
| Record (full) | `ros2 bag record point_cloud2 /pointcloud/filtered /scan /map /odom /tf /tf_static --storage mcap -o my_scan` |
| List MCAP topics | `python mcap_viewer.py /path/to/data/my_scan/ --list-topics` |
| **Raw view** | `python mcap_viewer.py /path/to/data/my_scan/ --topic /point_cloud2` |
| **Save cleaned PCD** | `python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --voxel 0.02 --save /path/to/data/cleaned.pcd` |
| **View PCD** | `python flythrough.py /path/to/data/cleaned.pcd` |
| Save max density | `python mcap_viewer_tf.py /path/to/data/my_scan/ --topic /point_cloud2 --no-cleanup --save /path/to/data/full.pcd` |

---

## Troubleshooting

- **No PointCloud2 topic** — The L1 is only on Go2 PRO and EDU models.
  The AIR does not have LiDAR.

- **Camera shows but no LiDAR** — LiDAR data may take a few seconds to
  start. Check `ros2 topic hz point_cloud2`.

- **`--storage mcap` not recognized** — Install the MCAP plugin:
  `sudo apt install ros-humble-rosbag2-storage-mcap`

- **open3d import error** — open3d requires Python 3.11. Create a venv:
  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- **Smeared point cloud in raw viewer** — Expected if the robot was moving.
  Use `mcap_viewer_tf.py` for the aligned version, or try
  `--every 10 --voxel 0.1` for aggressive downsampling.

- **`Cannot transform` error in TF viewer** — The fixed frame doesn't exist
  in the recording. Try `--fixed-frame odom`. If no TF data at all, you
  forgot to record `/tf` — re-record including it.

- **Points in `point_cloud2` already look aligned** — That's because the
  SDK publishes them in the `odom` frame. But odom drifts, so use
  `mcap_viewer_tf.py --fixed-frame map` for SLAM-corrected alignment.
