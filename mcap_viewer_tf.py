#!/usr/bin/env python3
"""
TF-aware MCAP point cloud viewer.

Like mcap_viewer.py but transforms each frame from the LiDAR frame into
a fixed frame (map/odom) using recorded TF data. This produces a clean,
aligned point cloud even when the robot was moving during recording.

Also applies statistical outlier removal and voxel downsampling for a
cleaned-up result.

Usage:
    python mcap_viewer_tf.py <bag_or_mcap>
    python mcap_viewer_tf.py <bag_or_mcap> --topic /point_cloud_map
    python mcap_viewer_tf.py <bag_or_mcap> --fixed-frame map
    python mcap_viewer_tf.py <bag_or_mcap> --no-cleanup
    python mcap_viewer_tf.py <bag_or_mcap> --voxel 0.03 --sor-neighbors 30
    python mcap_viewer_tf.py <bag_or_mcap> --save cleaned.pcd
"""

import sys
import argparse
from collections import defaultdict

import numpy as np
from pathlib import Path

from offline_deskew import resolve_mcap_path, parse_pointcloud2


# ── Quaternion / transform math ──────────────────────────────────────

def quat_to_matrix(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.dot(q, q)
    if n < 1e-12:
        return np.eye(3)
    q *= np.sqrt(2.0 / n)
    outer = np.outer(q, q)
    return np.array([
        [1.0 - outer[1,1] - outer[2,2],  outer[0,1] - outer[2,3],  outer[0,2] + outer[1,3]],
        [outer[0,1] + outer[2,3],  1.0 - outer[0,0] - outer[2,2],  outer[1,2] - outer[0,3]],
        [outer[0,2] - outer[1,3],  outer[1,2] + outer[0,3],  1.0 - outer[0,0] - outer[1,1]],
    ])


def make_transform(tx, ty, tz, qx, qy, qz, qw):
    """Build a 4x4 homogeneous transform from translation + quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_matrix(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def slerp_quat(q0, q1, t):
    """Spherical linear interpolation between two quaternions (x,y,z,w)."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    result = a * q0 + b * q1
    return result / np.linalg.norm(result)


def interpolate_transform(t_ns, stamps, transforms):
    """Interpolate a 4x4 transform at time t_ns from sorted (stamps, transforms)."""
    if t_ns <= stamps[0]:
        return transforms[0]
    if t_ns >= stamps[-1]:
        return transforms[-1]

    idx = np.searchsorted(stamps, t_ns, side='right') - 1
    t0, t1 = stamps[idx], stamps[idx + 1]
    alpha = (t_ns - t0) / (t1 - t0) if t1 != t0 else 0.0

    T0, T1 = transforms[idx], transforms[idx + 1]

    # Lerp translation
    trans = (1 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

    # Slerp rotation (extract quaternions from matrices)
    q0 = matrix_to_quat(T0[:3, :3])
    q1 = matrix_to_quat(T1[:3, :3])
    q = slerp_quat(q0, q1, alpha)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_matrix(q[0], q[1], q[2], q[3])
    T[:3, 3] = trans
    return T


def matrix_to_quat(R):
    """3x3 rotation matrix to quaternion (x,y,z,w)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


# ── Minimal TF buffer ────────────────────────────────────────────────

class TFBuffer:
    """Lightweight TF2-like buffer for offline MCAP playback."""

    def __init__(self):
        # dynamic[parent][child] = (sorted_stamps_ns, sorted_transforms)
        self.dynamic = defaultdict(lambda: defaultdict(lambda: ([], [])))
        # static[parent][child] = 4x4 transform
        self.static = {}
        self._finalized = False

    def add_transform(self, parent, child, stamp_ns, T, is_static=False):
        if is_static:
            key = (parent, child)
            self.static[key] = T
        else:
            entry = self.dynamic[parent][child]
            entry[0].append(stamp_ns)
            entry[1].append(T)

    def finalize(self):
        """Sort dynamic transforms by timestamp. Call after loading all TFs."""
        for parent in self.dynamic:
            for child in self.dynamic[parent]:
                stamps, transforms = self.dynamic[parent][child]
                order = np.argsort(stamps)
                self.dynamic[parent][child] = (
                    np.array(stamps, dtype=np.int64)[order],
                    [transforms[i] for i in order],
                )
        self._finalized = True

    def _lookup_direct(self, parent, child, stamp_ns):
        """Lookup a single-hop transform parent->child at stamp_ns."""
        key = (parent, child)
        if key in self.static:
            return self.static[key]

        if parent in self.dynamic and child in self.dynamic[parent]:
            stamps, transforms = self.dynamic[parent][child]
            if len(stamps) > 0:
                return interpolate_transform(stamp_ns, stamps, transforms)

        return None

    def _find_path(self, source, target):
        """BFS to find a frame chain from source to target."""
        # Collect all known edges
        edges = defaultdict(set)
        for (p, c) in self.static:
            edges[p].add(c)
            edges[c].add(p)
        for p in self.dynamic:
            for c in self.dynamic[p]:
                edges[p].add(c)
                edges[c].add(p)

        # BFS
        visited = {source}
        queue = [(source, [source])]
        while queue:
            node, path = queue.pop(0)
            if node == target:
                return path
            for neighbor in edges[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def lookup(self, target_frame, source_frame, stamp_ns):
        """
        Get the 4x4 transform that takes points in source_frame
        into target_frame at the given timestamp.

        Returns None if the chain cannot be resolved.
        """
        if not self._finalized:
            self.finalize()

        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)

        path = self._find_path(target_frame, source_frame)
        if path is None:
            return None

        # Walk the chain, accumulating transforms
        T = np.eye(4, dtype=np.float64)
        for i in range(len(path) - 1):
            parent, child = path[i], path[i + 1]
            fwd = self._lookup_direct(parent, child, stamp_ns)
            if fwd is not None:
                T = T @ fwd
            else:
                inv = self._lookup_direct(child, parent, stamp_ns)
                if inv is not None:
                    T = T @ np.linalg.inv(inv)
                else:
                    return None
        return T

    def available_frames(self):
        """Return all known frame IDs."""
        frames = set()
        for (p, c) in self.static:
            frames.add(p)
            frames.add(c)
        for p in self.dynamic:
            frames.add(p)
            for c in self.dynamic[p]:
                frames.add(c)
        return sorted(frames)


# ── MCAP reading ─────────────────────────────────────────────────────

def read_tf_and_clouds(mcap_file, cloud_topic, max_frames=None, every=1):
    """Read TF transforms and PointCloud2 messages in a single pass."""
    from mcap_ros2.reader import read_ros2_messages

    tf_buffer = TFBuffer()
    clouds = []    # (stamp_ns, frame_id, xyz_array)
    cloud_count = 0

    tf_topics = {'/tf', '/tf_static'}
    read_topics = tf_topics | {cloud_topic}

    print(f"Reading '{cloud_topic}' + TF from {mcap_file.name} ...")

    for msg in read_ros2_messages(str(mcap_file), topics=list(read_topics)):
        topic = msg.channel.topic

        if topic in tf_topics:
            is_static = (topic == '/tf_static')
            for tf in msg.ros_msg.transforms:
                stamp = tf.header.stamp
                stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                t = tf.transform.translation
                r = tf.transform.rotation
                T = make_transform(t.x, t.y, t.z, r.x, r.y, r.z, r.w)
                tf_buffer.add_transform(
                    tf.header.frame_id, tf.child_frame_id,
                    stamp_ns, T, is_static=is_static)

        elif topic == cloud_topic:
            cloud_count += 1
            if (cloud_count - 1) % every != 0:
                continue

            ros_msg = msg.ros_msg
            stamp = ros_msg.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            frame_id = ros_msg.header.frame_id

            points = parse_pointcloud2(ros_msg)
            xyz = np.column_stack([
                points['x'].astype(np.float32),
                points['y'].astype(np.float32),
                points['z'].astype(np.float32),
            ])

            valid = np.isfinite(xyz).all(axis=1)
            dist = np.linalg.norm(xyz, axis=1)
            mask = valid & (dist > 0.3) & (dist < 200.0)
            xyz = xyz[mask]

            if len(xyz) > 0:
                clouds.append((stamp_ns, frame_id, xyz))

            if max_frames and len(clouds) >= max_frames:
                break

            if len(clouds) % 50 == 0 and len(clouds) > 0:
                total = sum(len(c[2]) for c in clouds)
                print(f"  {len(clouds)} frames, {total:,} points ...")

    tf_buffer.finalize()
    total_pts = sum(len(c[2]) for c in clouds)
    print(f"Read {len(clouds)} frames ({total_pts:,} points) from {cloud_count} messages")

    frames = tf_buffer.available_frames()
    print(f"TF frames: {', '.join(frames)}")

    return tf_buffer, clouds


def transform_clouds(tf_buffer, clouds, fixed_frame):
    """Transform each cloud into fixed_frame using TF lookup."""
    transformed = []
    skipped = 0

    for i, (stamp_ns, frame_id, xyz) in enumerate(clouds):
        T = tf_buffer.lookup(fixed_frame, frame_id, stamp_ns)
        if T is None:
            skipped += 1
            continue

        # Apply 4x4 transform: points are Nx3, convert to homogeneous
        ones = np.ones((len(xyz), 1), dtype=xyz.dtype)
        pts_h = np.hstack([xyz, ones])  # Nx4
        pts_t = (T @ pts_h.T).T[:, :3].astype(np.float32)
        transformed.append(pts_t)

        if (len(transformed)) % 100 == 0:
            print(f"  Transformed {len(transformed)} frames ...")

    if skipped > 0:
        print(f"  Warning: skipped {skipped} frames (TF lookup failed)")
    print(f"Transformed {len(transformed)} frames into '{fixed_frame}'")
    return transformed


def cleanup_cloud(points, voxel_size=0.05, sor_neighbors=20, sor_std=2.0):
    """Voxel downsample + statistical outlier removal."""
    import open3d as o3d

    before = len(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Voxel downsample
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        after_voxel = len(pcd.points)
        print(f"  Voxel downsample ({voxel_size}m): {before:,} -> {after_voxel:,}")

    # Statistical outlier removal
    if sor_neighbors > 0:
        before_sor = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=sor_neighbors, std_ratio=sor_std)
        after_sor = len(pcd.points)
        removed = before_sor - after_sor
        print(f"  Outlier removal ({sor_neighbors} neighbors, {sor_std} std): "
              f"{before_sor:,} -> {after_sor:,} ({removed:,} removed)")

    return np.asarray(pcd.points, dtype=np.float32)


def guess_fixed_frame(tf_buffer):
    """Try common fixed frame names."""
    frames = tf_buffer.available_frames()
    for candidate in ['map', 'odom', 'world', 'earth']:
        if candidate in frames:
            return candidate
    return frames[0] if frames else None


def main():
    parser = argparse.ArgumentParser(
        description="TF-aware MCAP point cloud viewer with cleanup")
    parser.add_argument("bag_path", help="Path to MCAP file or bag directory")
    parser.add_argument("--topic", "-t", help="PointCloud2 topic to read")
    parser.add_argument("--fixed-frame", "-f", default=None,
                        help="Target frame to transform into (default: auto-detect)")
    parser.add_argument("--list-topics", action="store_true",
                        help="List all topics and exit")
    parser.add_argument("--max-frames", "-n", type=int, default=None,
                        help="Max number of frames to load")
    parser.add_argument("--every", "-e", type=int, default=1,
                        help="Use every Nth frame (default: 1 = all)")

    # Cleanup options
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip cleanup (just transform, no filter)")
    parser.add_argument("--voxel", "-v", type=float, default=0.05,
                        help="Voxel downsample size in meters (default: 0.05)")
    parser.add_argument("--sor-neighbors", type=int, default=20,
                        help="Statistical outlier removal: neighbor count (default: 20)")
    parser.add_argument("--sor-std", type=float, default=2.0,
                        help="Statistical outlier removal: std ratio (default: 2.0)")

    parser.add_argument("--save", "-s", type=str, default=None,
                        help="Save result to PCD file and exit")
    args = parser.parse_args()

    mcap_file = resolve_mcap_path(args.bag_path)

    if args.list_topics:
        from mcap_viewer import list_topics
        list_topics(mcap_file)
        return

    # Find the point cloud topic
    if args.topic:
        cloud_topic = args.topic
    else:
        from mcap_viewer import find_pointcloud_topic
        cloud_topic = find_pointcloud_topic(mcap_file)

    # Read everything in one pass
    tf_buffer, clouds = read_tf_and_clouds(
        mcap_file, cloud_topic, args.max_frames, args.every)

    if not clouds:
        print("ERROR: No point cloud data found.")
        sys.exit(1)

    # Determine fixed frame
    fixed_frame = args.fixed_frame or guess_fixed_frame(tf_buffer)
    if fixed_frame is None:
        print("ERROR: No TF frames found. Use mcap_viewer.py instead (no TF needed).")
        sys.exit(1)
    print(f"\nFixed frame: '{fixed_frame}'")

    # Check that at least one cloud can be transformed
    test_stamp, test_frame, _ = clouds[0]
    test_T = tf_buffer.lookup(fixed_frame, test_frame, test_stamp)
    if test_T is None:
        print(f"\nERROR: Cannot transform '{test_frame}' -> '{fixed_frame}'")
        print(f"Available frames: {', '.join(tf_buffer.available_frames())}")
        print(f"\nTry --fixed-frame with one of the above frames, or use "
              f"mcap_viewer.py for raw (untransformed) viewing.")
        sys.exit(1)

    # Transform all frames
    print(f"\nTransforming '{test_frame}' -> '{fixed_frame}' ...")
    transformed = transform_clouds(tf_buffer, clouds, fixed_frame)

    if not transformed:
        print("ERROR: No frames could be transformed.")
        sys.exit(1)

    points = np.vstack(transformed).astype(np.float32)
    print(f"\nAccumulated: {len(points):,} points")

    # Cleanup
    if not args.no_cleanup:
        print("\nCleaning up ...")
        points = cleanup_cloud(
            points,
            voxel_size=args.voxel,
            sor_neighbors=args.sor_neighbors,
            sor_std=args.sor_std,
        )
        print(f"Final: {len(points):,} points")

    # Save and exit if requested
    if args.save:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"\nSaved to {args.save}")
        return

    # Launch flythrough viewer
    from flythrough import FPSViewer
    label = f"{cloud_topic} in {fixed_frame}"
    viewer = FPSViewer([points], cloud_names=[label])
    viewer.run()


if __name__ == "__main__":
    main()
