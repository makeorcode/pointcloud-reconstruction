#!/usr/bin/env python3
"""
Offline deskewing of VLP-16 point clouds using platform encoder angles.
Reproduces the ROBIN point_cloud_reconstructor algorithm without ROS.

Requirements:
    pip install mcap mcap-ros2-support numpy zstandard

Usage:
    python offline_deskew.py <bag_directory_or_mcap_file> [output.npz]
"""

import sys
import numpy as np
from pathlib import Path


# ── Production parameters ──────────────────────────────────────────
ROTATION_AXIS = 'x'
ANGLE_OFFSET_DEG = 184.0
ROTATION_CENTER = np.array([0.0, 0.0, 0.0])
INVERT_ROTATION = False
ENCODER_TIME_OFFSET_MS = 0.0

MOUNT_RPY_DEG = [0.0, 0.0, 0.0]
MOUNT_AXES = ['+x', '+y', '+z']


# ── Mount rotation matrix (ZYX Euler) ─────────────────────────────
def build_mount_rotation(rpy_deg):
    r, p, y = np.deg2rad(rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr            ],
    ])
    return R


# ── Axis mapping ──────────────────────────────────────────────────
def parse_axis_mapping(axes):
    axis_idx = {'x': 0, 'y': 1, 'z': 2}
    src = []
    signs = []
    for a in axes:
        a = a.strip().lower()
        sign = -1.0 if a.startswith('-') else 1.0
        key = a.lstrip('+-')
        src.append(axis_idx[key])
        signs.append(sign)
    return np.array(src, dtype=int), np.array(signs)


def apply_fwd_map(xyz, src, signs):
    return signs[0]*xyz[:, src[0]], signs[1]*xyz[:, src[1]], signs[2]*xyz[:, src[2]]


def apply_inv_map(m0, m1, m2, src, signs):
    out = np.empty((len(m0), 3))
    mapped = [m0, m1, m2]
    for i in range(3):
        out[:, src[i]] = signs[i] * mapped[i]
    return out


# ── PointCloud2 binary parsing ────────────────────────────────────
def parse_pointcloud2(msg):
    ros_to_np = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    fields = []
    for f in msg.fields:
        np_type = ros_to_np.get(f.datatype, 'u1')
        fields.append((f.name, np_type, f.offset))

    fields.sort(key=lambda x: x[2])
    dt_fields = []
    for name, dtype, offset in fields:
        dt_fields.append((name, dtype))

    dt = np.dtype(dt_fields)

    if any(f[2] != dt.fields[f[0]][1] for f in fields):
        dt = np.dtype({'names': [f[0] for f in fields],
                       'formats': [f[1] for f in fields],
                       'offsets': [f[2] for f in fields],
                       'itemsize': msg.point_step})

    points = np.frombuffer(msg.data, dtype=dt)
    return points.copy()


# ── Core deskew function ──────────────────────────────────────────
def deskew_cloud(points, cloud_time_ns, angle_times_ns, angle_values_rad,
                 rotation_axis=ROTATION_AXIS,
                 angle_offset_deg=ANGLE_OFFSET_DEG,
                 rotation_center=ROTATION_CENTER,
                 invert_rotation=INVERT_ROTATION,
                 encoder_time_offset_ms=ENCODER_TIME_OFFSET_MS,
                 mount_rpy_deg=MOUNT_RPY_DEG,
                 mount_axes=MOUNT_AXES,
                 use_per_point_time=True):
    x = points['x'].astype(np.float64)
    y = points['y'].astype(np.float64)
    z = points['z'].astype(np.float64)

    angles_unwrapped = np.unwrap(angle_values_rad)

    time_offset_ns = encoder_time_offset_ms * 1e6
    if use_per_point_time and 'time' in points.dtype.names:
        pt_ns = cloud_time_ns + points['time'].astype(np.float64) * 1e9 + time_offset_ns
        pt_ns = np.clip(pt_ns, angle_times_ns[0], angle_times_ns[-1])
        platform_angles = np.interp(pt_ns, angle_times_ns, angles_unwrapped)
    else:
        t = np.clip(cloud_time_ns + time_offset_ns, angle_times_ns[0], angle_times_ns[-1])
        platform_angles = np.full(len(points), np.interp(t, angle_times_ns, angles_unwrapped))

    offset_rad = np.deg2rad(angle_offset_deg)
    if invert_rotation:
        rot = platform_angles - offset_rad
    else:
        rot = -platform_angles - offset_rad
    cos_a = np.cos(rot)
    sin_a = np.sin(rot)

    src, signs = parse_axis_mapping(mount_axes)
    mx, my, mz = apply_fwd_map(np.column_stack([x, y, z]), src, signs)

    R = build_mount_rotation(mount_rpy_deg)
    px = R[0, 0]*mx + R[0, 1]*my + R[0, 2]*mz
    py = R[1, 0]*mx + R[1, 1]*my + R[1, 2]*mz
    pz = R[2, 0]*mx + R[2, 1]*my + R[2, 2]*mz

    px -= rotation_center[0]
    py -= rotation_center[1]
    pz -= rotation_center[2]

    if rotation_axis == 'x':
        rx = px
        ry = py * cos_a - pz * sin_a
        rz = py * sin_a + pz * cos_a
    elif rotation_axis == 'y':
        rx = px * cos_a + pz * sin_a
        ry = py
        rz = -px * sin_a + pz * cos_a
    else:
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        rz = pz

    rx += rotation_center[0]
    ry += rotation_center[1]
    rz += rotation_center[2]

    R_inv = R.T
    mx2 = R_inv[0, 0]*rx + R_inv[0, 1]*ry + R_inv[0, 2]*rz
    my2 = R_inv[1, 0]*rx + R_inv[1, 1]*ry + R_inv[1, 2]*rz
    mz2 = R_inv[2, 0]*rx + R_inv[2, 1]*ry + R_inv[2, 2]*rz

    corrected = apply_inv_map(mx2, my2, mz2, src, signs)

    points['x'] = corrected[:, 0]
    points['y'] = corrected[:, 1]
    points['z'] = corrected[:, 2]

    return corrected


# ── Resolve bag path to .mcap file ────────────────────────────────
def resolve_mcap_path(bag_path):
    bag_path = Path(bag_path)
    if bag_path.is_file() and bag_path.suffix == '.mcap':
        return bag_path

    if bag_path.is_dir():
        # Look for plain .mcap first
        mcap_files = [f for f in bag_path.glob("*.mcap") if not str(f).endswith('.zstd')]
        if mcap_files:
            return mcap_files[0]

        # Decompress .mcap.zstd if needed
        zstd_files = list(bag_path.glob("*.mcap.zstd"))
        if zstd_files:
            import zstandard
            src = zstd_files[0]
            dst = src.with_suffix('')  # strip .zstd
            print(f"Decompressing {src.name} -> {dst.name} ...")
            dctx = zstandard.ZstdDecompressor()
            with open(src, 'rb') as ifh, open(dst, 'wb') as ofh:
                dctx.copy_stream(ifh, ofh)
            print(f"Done ({dst.stat().st_size / 1e6:.1f} MB)")
            return dst

    raise FileNotFoundError(f"No .mcap file found in {bag_path}")


# ── Bag reader ────────────────────────────────────────────────────
def read_bag(bag_path):
    from mcap_ros2.reader import read_ros2_messages

    mcap_file = resolve_mcap_path(bag_path)
    print(f"Reading: {mcap_file}")

    angle_times = []
    angle_values = []
    clouds = []

    for msg in read_ros2_messages(str(mcap_file)):
        if msg.channel.topic == '/rotating_platform/angle':
            stamp = msg.log_time_ns
            angle_deg = msg.ros_msg.data
            angle_times.append(stamp)
            angle_values.append(np.deg2rad(angle_deg))
        elif msg.channel.topic == '/velodyne_points':
            stamp_msg = msg.ros_msg.header.stamp
            stamp_ns = stamp_msg.sec * 1_000_000_000 + stamp_msg.nanosec
            clouds.append((stamp_ns, msg.ros_msg))

    angle_times = np.array(angle_times, dtype=np.float64)
    angle_values = np.array(angle_values, dtype=np.float64)

    print(f"Loaded {len(angle_times)} angle samples, {len(clouds)} point clouds")
    return angle_times, angle_values, clouds


# ── Main ──────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python offline_deskew.py <bag_path> [output.npz]")
        print("  bag_path: path to rosbag directory or .mcap file")
        print("  output:   optional output file (default: deskewed.npz)")
        sys.exit(1)

    bag_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else "deskewed.npz"

    angle_times, angle_values, clouds = read_bag(bag_path)

    if len(angle_times) < 2:
        print("ERROR: Need at least 2 angle samples for interpolation")
        sys.exit(1)

    all_points = []
    all_timestamps = []

    for i, (cloud_time_ns, cloud_msg) in enumerate(clouds):
        points = parse_pointcloud2(cloud_msg)
        if len(points) == 0:
            continue

        corrected = deskew_cloud(
            points, cloud_time_ns, angle_times, angle_values
        )

        all_points.append(corrected.astype(np.float32))
        all_timestamps.append(cloud_time_ns)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(clouds)} clouds")

    print(f"Processed {len(all_points)} clouds total")

    np.savez_compressed(
        output_path,
        points=np.array(all_points, dtype=object),
        timestamps=np.array(all_timestamps),
        angle_times=angle_times,
        angle_values=angle_values,
    )
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
