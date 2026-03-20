#!/usr/bin/env python3
"""
Generic MCAP point cloud viewer.

Opens any MCAP file, lists available PointCloud2 topics, and displays
the accumulated point cloud in the flythrough viewer.

Works with any LiDAR that publishes sensor_msgs/PointCloud2
(Velodyne VLP-16, Unitree L1, Ouster, Livox, etc.).

Usage:
    python mcap_viewer.py <bag_or_mcap>                     # auto-detect topic
    python mcap_viewer.py <bag_or_mcap> --topic /cloud      # specify topic
    python mcap_viewer.py <bag_or_mcap> --list-topics       # list all topics
    python mcap_viewer.py <bag_or_mcap> --max-frames 200    # limit frames
    python mcap_viewer.py <bag_or_mcap> --every 5           # use every Nth frame
    python mcap_viewer.py <bag_or_mcap> --voxel 0.05        # downsample voxel size
"""

import sys
import argparse
import numpy as np
from pathlib import Path

from offline_deskew import resolve_mcap_path, parse_pointcloud2


def list_topics(mcap_file):
    """Print all topics and their message types from MCAP summary."""
    from mcap.reader import make_reader

    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

    if summary is None:
        print("Warning: no summary in MCAP file")
        return {}

    schemas_by_id = {s.id: s for s in summary.schemas.values()}
    # Count messages per channel from chunk indexes
    msg_counts = {}
    if summary.statistics:
        msg_counts = summary.statistics.channel_message_counts

    topics = {}
    for ch in summary.channels.values():
        schema = schemas_by_id.get(ch.schema_id)
        schema_name = schema.name if schema else "unknown"
        count = msg_counts.get(ch.id, 0)
        topics[ch.topic] = {"type": schema_name, "count": count}

    print(f"\nTopics in {mcap_file.name}:")
    print(f"{'Topic':<40} {'Type':<45} {'Messages':>8}")
    print("-" * 95)
    for topic, info in sorted(topics.items()):
        print(f"{topic:<40} {info['type']:<45} {info['count']:>8}")
    print()
    return topics


def find_pointcloud_topic(mcap_file):
    """Auto-detect a PointCloud2 topic in the MCAP file."""
    from mcap.reader import make_reader

    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

    if summary is None:
        print("Warning: no summary in MCAP, scanning messages...")
        topics = list_topics(mcap_file)
        pc_topics = [t for t, info in topics.items()
                     if "PointCloud2" in info["type"]]
    else:
        schemas_by_id = {s.id: s for s in summary.schemas.values()}
        pc_topics = []
        for ch in summary.channels.values():
            schema = schemas_by_id.get(ch.schema_id)
            if schema and "PointCloud2" in schema.name:
                pc_topics.append(ch.topic)

    if not pc_topics:
        print("ERROR: No PointCloud2 topics found in this MCAP file.")
        print("Use --list-topics to see what's available.")
        sys.exit(1)

    if len(pc_topics) == 1:
        print(f"Auto-detected topic: {pc_topics[0]}")
        return pc_topics[0]

    print(f"Multiple PointCloud2 topics found:")
    for i, t in enumerate(pc_topics):
        print(f"  [{i}] {t}")
    print(f"\nUsing: {pc_topics[0]}  (override with --topic)")
    return pc_topics[0]


def read_clouds(mcap_file, topic, max_frames=None, every=1):
    """Read PointCloud2 messages from the MCAP file."""
    from mcap_ros2.reader import read_ros2_messages

    clouds = []
    count = 0

    print(f"Reading '{topic}' from {mcap_file.name} ...")

    for msg in read_ros2_messages(str(mcap_file), topics=[topic]):
        count += 1
        if (count - 1) % every != 0:
            continue

        points = parse_pointcloud2(msg.ros_msg)
        xyz = np.column_stack([
            points['x'].astype(np.float32),
            points['y'].astype(np.float32),
            points['z'].astype(np.float32),
        ])

        # Filter invalid points
        valid = np.isfinite(xyz).all(axis=1)
        dist = np.linalg.norm(xyz, axis=1)
        mask = valid & (dist > 0.3) & (dist < 200.0)
        xyz = xyz[mask]

        if len(xyz) > 0:
            clouds.append(xyz)

        if max_frames and len(clouds) >= max_frames:
            break

        if len(clouds) % 50 == 0 and len(clouds) > 0:
            total = sum(len(c) for c in clouds)
            print(f"  {len(clouds)} frames, {total:,} points ...")

    print(f"Read {len(clouds)} frames from {count} messages "
          f"({sum(len(c) for c in clouds):,} points total)")
    return clouds


def main():
    parser = argparse.ArgumentParser(
        description="View PointCloud2 data from any MCAP file")
    parser.add_argument("bag_path", help="Path to MCAP file or bag directory")
    parser.add_argument("--topic", "-t", help="PointCloud2 topic to read")
    parser.add_argument("--list-topics", action="store_true",
                        help="List all topics and exit")
    parser.add_argument("--max-frames", "-n", type=int, default=None,
                        help="Max number of frames to load")
    parser.add_argument("--every", "-e", type=int, default=1,
                        help="Use every Nth frame (default: 1 = all)")
    parser.add_argument("--voxel", "-v", type=float, default=None,
                        help="Voxel downsample size in meters (e.g. 0.05)")
    parser.add_argument("--save", "-s", type=str, default=None,
                        help="Save accumulated cloud to PCD file and exit")
    args = parser.parse_args()

    mcap_file = resolve_mcap_path(args.bag_path)

    if args.list_topics:
        list_topics(mcap_file)
        return

    topic = args.topic or find_pointcloud_topic(mcap_file)
    clouds = read_clouds(mcap_file, topic, args.max_frames, args.every)

    if not clouds:
        print("ERROR: No point cloud data found.")
        sys.exit(1)

    points = np.vstack(clouds).astype(np.float32)
    print(f"Accumulated cloud: {len(points):,} points")

    # Optional voxel downsample
    if args.voxel:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd = pcd.voxel_down_sample(args.voxel)
        points = np.asarray(pcd.points, dtype=np.float32)
        print(f"After voxel downsample ({args.voxel}m): {len(points):,} points")

    # Save and exit if requested
    if args.save:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"Saved to {args.save}")
        return

    # Launch flythrough viewer
    from flythrough import FPSViewer
    viewer = FPSViewer([points], cloud_names=[topic])
    viewer.run()


if __name__ == "__main__":
    main()
