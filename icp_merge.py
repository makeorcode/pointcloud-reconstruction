#!/usr/bin/env python3
"""
ICP-based point cloud reconstruction from bag file.

Instead of calibrating mount parameters, this splits the scan into N angular
slices with overlapping windows, then uses ICP to align each slice to a
reference. Overlapping windows give ICP much more shared geometry to work with.

Each slice covers (360/N * overlap_factor) degrees but only contributes its
core (360/N) degrees to the final output. The overlap is used solely for ICP.

Usage:
    python icp_merge.py <bag_directory>
    python icp_merge.py <bag_directory> --slices 16
    python icp_merge.py <bag_directory> --slices 12 --fine
    python icp_merge.py <bag_directory> -o output.pcd
"""

import sys
import copy
import time
import argparse
import numpy as np

from offline_deskew import (
    read_bag, parse_pointcloud2, deskew_cloud,
    ROTATION_AXIS, ANGLE_OFFSET_DEG, ROTATION_CENTER,
    INVERT_ROTATION, ENCODER_TIME_OFFSET_MS,
    MOUNT_RPY_DEG, MOUNT_AXES,
)


def filter_points(pts, max_range=80.0, min_range=0.5):
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    dist = np.linalg.norm(pts, axis=1)
    mask = (dist < max_range) & (dist > min_range)
    return pts[mask]


def run_icp(pcd_source, pcd_target, voxel_size):
    """Align pcd_source onto pcd_target using coarse-to-fine ICP."""
    import open3d as o3d

    down_src = pcd_source.voxel_down_sample(voxel_size)
    down_tgt = pcd_target.voxel_down_sample(voxel_size)
    down_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 4, max_nn=30))
    down_tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 4, max_nn=30))

    # Coarse point-to-point
    result_coarse = o3d.pipelines.registration.registration_icp(
        down_src, down_tgt, voxel_size * 5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    # Fine point-to-plane
    result_fine = o3d.pipelines.registration.registration_icp(
        down_src, down_tgt, voxel_size * 2,
        result_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    return result_fine


def main():
    parser = argparse.ArgumentParser(description="ICP-based scan reconstruction")
    parser.add_argument("bag_path", help="Path to bag directory or .mcap file")
    parser.add_argument("--slices", type=int, default=12,
                        help="Number of angular slices (default: 12)")
    parser.add_argument("--overlap", type=float, default=2.0,
                        help="Overlap factor: each slice window is this many "
                             "times the step width (default: 2.0)")
    parser.add_argument("--fine", action="store_true",
                        help="Use finer voxel size for ICP")
    parser.add_argument("--min-fitness", type=float, default=0.3,
                        help="Reject slices with ICP fitness below this "
                             "(default: 0.3)")
    parser.add_argument("--no-chain", action="store_true",
                        help="Align every slice to the reference instead of "
                             "chaining neighbors (only works when all slices "
                             "share geometry with the reference)")
    parser.add_argument("--map-voxel", type=float, default=0.01,
                        help="Final downsample voxel size in metres, 0 to "
                             "disable (default: 0.01)")
    parser.add_argument("--trim-end", type=float, default=0,
                        help="Trim last N seconds from bag data (default: 0)")
    parser.add_argument("-o", "--output", default="icp_merged.pcd",
                        help="Output PCD file (default: icp_merged.pcd)")
    args = parser.parse_args()

    if not args.output.lower().endswith('.pcd'):
        args.output += '.pcd'

    n_slices = args.slices
    overlap = args.overlap
    voxel_size = 0.05 if args.fine else 0.10
    step_rad = 2 * np.pi / n_slices
    window_rad = step_rad * overlap  # wider window for ICP overlap

    step_deg = 360.0 / n_slices
    window_deg = step_deg * overlap

    print("=" * 60)
    print(f"  ICP Merge ({n_slices} slices, {window_deg:.0f}° windows, "
          f"{step_deg:.0f}° steps)")
    print("=" * 60)

    # Step 1: Read bag
    print("\n[1/4] Reading bag...")
    angle_times, angle_values, raw_clouds = read_bag(args.bag_path)

    # Trim end of bag if requested
    if args.trim_end > 0:
        cutoff_ns = angle_times[-1] - args.trim_end * 1e9
        angle_mask = angle_times <= cutoff_ns
        angle_times = angle_times[angle_mask]
        angle_values = angle_values[angle_mask]
        raw_clouds = [(t, msg) for t, msg in raw_clouds if t <= cutoff_ns]
        duration = (angle_times[-1] - angle_times[0]) / 1e9
        print(f"  Trimmed last {args.trim_end:.0f}s — {len(raw_clouds)} clouds, "
              f"{duration:.1f}s remaining")

    angles_unwrapped = np.unwrap(angle_values)

    # Step 2: Deskew all clouds, store with their phase angle
    print(f"\n[2/4] Deskewing all clouds...")
    all_deskewed = []  # list of (phase, points)

    for i, (cloud_time_ns, cloud_msg) in enumerate(raw_clouds):
        points = parse_pointcloud2(cloud_msg)
        if len(points) == 0:
            continue

        corrected = deskew_cloud(
            points, cloud_time_ns, angle_times, angle_values,
            rotation_axis=ROTATION_AXIS,
            angle_offset_deg=ANGLE_OFFSET_DEG,
            rotation_center=ROTATION_CENTER,
            invert_rotation=INVERT_ROTATION,
            encoder_time_offset_ms=ENCODER_TIME_OFFSET_MS,
            mount_rpy_deg=MOUNT_RPY_DEG,
            mount_axes=MOUNT_AXES,
        )
        pts = filter_points(corrected.astype(np.float32))
        if len(pts) == 0:
            continue

        t_clipped = np.clip(cloud_time_ns, angle_times[0], angle_times[-1])
        phase = np.interp(t_clipped, angle_times, angles_unwrapped) % (2 * np.pi)
        all_deskewed.append((phase, pts))

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(raw_clouds)}")

    print(f"  Total deskewed clouds: {len(all_deskewed)}")

    # Step 3: Build overlapping slices and ICP-align
    import open3d as o3d

    # For each slice, collect clouds in the wide window (for ICP)
    # and track which clouds are in the core (for final output)
    print(f"\n[3/4] Building overlapping slices and aligning...")

    slice_centers = [s * step_rad for s in range(n_slices)]

    def angle_in_window(phase, center, half_width):
        """Check if phase is within [center - half_width, center + half_width] mod 2pi."""
        diff = (phase - center + np.pi) % (2 * np.pi) - np.pi
        return abs(diff) < half_width

    # Build wide (for ICP) and core (for output) point clouds per slice
    wide_pcds = []
    core_pcds = []
    half_window = window_rad / 2
    half_step = step_rad / 2

    for s in range(n_slices):
        center = slice_centers[s]
        wide_pts = []
        core_pts = []

        for phase, pts in all_deskewed:
            if angle_in_window(phase, center, half_window):
                wide_pts.append(pts)
            if angle_in_window(phase, center, half_step):
                core_pts.append(pts)

        # Wide PCD (for ICP alignment)
        if wide_pts:
            w = np.vstack(wide_pts)
            pcd_w = o3d.geometry.PointCloud()
            pcd_w.points = o3d.utility.Vector3dVector(w.astype(np.float64))
        else:
            pcd_w = None

        # Core PCD (for final output, no overlap)
        if core_pts:
            c = np.vstack(core_pts)
            pcd_c = o3d.geometry.PointCloud()
            pcd_c.points = o3d.utility.Vector3dVector(c.astype(np.float64))
        else:
            pcd_c = None

        wide_pcds.append(pcd_w)
        core_pcds.append(pcd_c)

        deg_start = np.degrees(center - half_step)
        deg_end = np.degrees(center + half_step)
        wide_count = len(w) if wide_pts else 0
        core_count = len(c) if core_pts else 0
        print(f"  Slice {s:2d} (core {deg_start:5.1f}°-{deg_end:5.1f}°): "
              f"{core_count:,} core, {wide_count:,} wide")

    # Find reference (largest wide slice)
    valid = [(s, wide_pcds[s]) for s in range(n_slices) if wide_pcds[s] is not None]
    if len(valid) < 2:
        print("ERROR: Need at least 2 non-empty slices")
        sys.exit(1)

    ref_idx = max(valid, key=lambda x: len(x[1].points))[0]
    print(f"\n  Reference slice: {ref_idx}")

    transforms = {ref_idx: np.eye(4)}
    ordered = sorted([s for s, _ in valid])
    t0 = time.time()
    rejected = set()

    if not args.no_chain:
        # Two-pass alignment:
        # Pass 1: Chain neighbors to get initial transforms for all slices.
        # Pass 2: Refine each slice directly against the reference using
        #         the chain transform as ICP initial guess. This gives
        #         independent per-slice alignment (no drift accumulation)
        #         with good initialization (from chain).
        ref_pos = ordered.index(ref_idx)
        n_valid = len(ordered)

        # Build chain order: ref+1, ref+2, ..., wrapping around to ref-1
        chain_order = []
        for k in range(1, n_valid):
            chain_order.append(ordered[(ref_pos + k) % n_valid])

        # Pass 1: Chain neighbors for initial guesses
        print("  Pass 1: Chaining neighbors...")
        prev_idx = ref_idx
        for src_idx in chain_order:
            result = run_icp(wide_pcds[src_idx], wide_pcds[prev_idx], voxel_size)
            transforms[src_idx] = transforms[prev_idx] @ result.transformation
            print(f"    Slice {src_idx:2d} -> {prev_idx:2d}: "
                  f"fitness={result.fitness:.3f}  RMSE={result.inlier_rmse:.4f}m")
            prev_idx = src_idx

        # Pass 2: Refine each slice against reference using chain as init
        print(f"\n  Pass 2: Refining against reference (slice {ref_idx})...")
        import open3d as o3d

        refined = {ref_idx}  # slices with direct-to-ref transforms
        unrefined = []

        # Downsample reference once
        down_tgt = wide_pcds[ref_idx].voxel_down_sample(voxel_size)
        down_tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4, max_nn=30))

        for src_idx in chain_order:
            down_src = wide_pcds[src_idx].voxel_down_sample(voxel_size)
            down_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 4, max_nn=30))

            result = o3d.pipelines.registration.registration_icp(
                down_src, down_tgt, voxel_size * 3,
                transforms[src_idx],
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=200),
            )

            if result.fitness >= args.min_fitness:
                transforms[src_idx] = result.transformation
                refined.add(src_idx)
                print(f"    Slice {src_idx:2d} -> ref {ref_idx:2d}: "
                      f"fitness={result.fitness:.3f}  "
                      f"RMSE={result.inlier_rmse:.4f}m  (refined)")
            else:
                unrefined.append(src_idx)
                print(f"    Slice {src_idx:2d} -> ref {ref_idx:2d}: "
                      f"fitness={result.fitness:.3f}  "
                      f"RMSE={result.inlier_rmse:.4f}m  (needs re-chain)")

        # Pass 3: Re-chain unrefined slices from nearest refined neighbor
        if unrefined:
            print(f"\n  Pass 3: Re-chaining {len(unrefined)} slices from "
                  f"refined anchors...")

            # Build circular ordering for neighbor lookup
            circ = ordered  # already sorted 0..N-1
            n_circ = len(circ)

            # For each unrefined slice, find nearest refined neighbor
            # and chain outward from it
            changed = True
            while changed:
                changed = False
                for src_idx in list(unrefined):
                    pos = circ.index(src_idx)
                    # Look left and right for a refined neighbor
                    for step in range(1, n_circ):
                        left = circ[(pos - step) % n_circ]
                        right = circ[(pos + step) % n_circ]
                        anchor = None
                        if left in refined:
                            anchor = left
                        elif right in refined:
                            anchor = right
                        if anchor is not None:
                            result = run_icp(wide_pcds[src_idx],
                                             wide_pcds[anchor], voxel_size)
                            transforms[src_idx] = (
                                transforms[anchor] @ result.transformation)
                            refined.add(src_idx)
                            unrefined.remove(src_idx)
                            changed = True
                            print(f"    Slice {src_idx:2d} -> {anchor:2d}: "
                                  f"fitness={result.fitness:.3f}  "
                                  f"RMSE={result.inlier_rmse:.4f}m  "
                                  f"(re-chained)")
                            break
    else:
        # Direct-to-reference alignment (identity initial guess)
        for src_idx in ordered:
            if src_idx == ref_idx:
                continue
            result = run_icp(wide_pcds[src_idx], wide_pcds[ref_idx], voxel_size)
            transforms[src_idx] = result.transformation
            tag = ""
            if result.fitness < args.min_fitness:
                rejected.add(src_idx)
                tag = "  ** REJECTED **"
            print(f"  Slice {src_idx:2d} -> ref {ref_idx:2d}: "
                  f"fitness={result.fitness:.3f}  RMSE={result.inlier_rmse:.4f}m{tag}")

    elapsed = time.time() - t0
    if rejected:
        print(f"\n  Rejected {len(rejected)} slice(s) with fitness < "
              f"{args.min_fitness}: {sorted(rejected)}")
    print(f"  ICP alignment done in {elapsed:.1f}s")

    # Step 4: Merge using CORE points only (no overlap in output)
    print(f"\n[4/4] Merging core slices with ICP transforms...")
    merged = o3d.geometry.PointCloud()
    for s in ordered:
        if core_pcds[s] is not None and s not in rejected:
            aligned = copy.deepcopy(core_pcds[s]).transform(transforms[s])
            merged += aligned

    print(f"  Total: {len(merged.points):,} points")

    if args.map_voxel > 0:
        merged_down = merged.voxel_down_sample(args.map_voxel)
        print(f"  After downsample ({args.map_voxel:.3f}m): "
              f"{len(merged_down.points):,} points")
    else:
        merged_down = merged
        print(f"  No downsample applied")

    o3d.io.write_point_cloud(args.output, merged_down)
    print(f"\n  Saved to {args.output}")
    print(f"  View: .venv\\Scripts\\python.exe flythrough.py {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
