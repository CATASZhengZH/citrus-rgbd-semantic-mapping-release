#!/usr/bin/env python3
"""Merge measured trajectory and semantic-map metrics for SLAM controls."""

from __future__ import annotations

import os
import argparse
import csv
from pathlib import Path


BACKENDS = ("vins_rgbd", "rtabmap", "orbslam3")


def read_single_row(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return next(csv.DictReader(stream), None)


def value(row, key, default="NA"):
    if row is None:
        return default
    result = row.get(key, default)
    return result if result not in (None, "") else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--best-segmentation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for backend in BACKENDS:
        backend_root = args.root / "slam" / backend
        trajectory = read_single_row(backend_root / "trajectory_statistics.csv")
        mapping = read_single_row(backend_root / "mapping" / "semantic_mapping_metrics.csv")
        details = read_single_row(backend_root / f"{backend}_metrics.csv")
        if backend == "orbslam3":
            details = read_single_row(backend_root / "orbslam3_metrics.csv")

        if (backend_root / "STATUS_BLOCKED.md").exists():
            status = "BLOCKED"
        else:
            status = "SUCCESS" if trajectory and mapping else "PARTIAL"
        rows.append(
            {
                "localization_backend": backend,
                "segmentation_model": args.best_segmentation,
                "trajectory_pose_count": value(trajectory, "pose_count"),
                "trajectory_duration_s": value(trajectory, "trajectory_duration_s"),
                "trajectory_coverage": value(trajectory, "valid_pose_rate"),
                "valid_pose_count": value(trajectory, "valid_pose_count"),
                "tracking_lost_count": value(details, "tracking_lost_count"),
                "relocalization_count": value(details, "relocalization_count"),
                "path_length_m": value(trajectory, "path_length_m"),
                "largest_pose_jump_m": value(details, "largest_pose_jump_m"),
                "runtime_fps": value(details, "runtime_fps"),
                "cpu_usage_percent": value(details, "cpu_usage_percent"),
                "peak_ram_mb": (
                    str(float(value(details, "max_rss_kb")) / 1024.0)
                    if value(details, "max_rss_kb") != "NA"
                    else "NA"
                ),
                "semantic_conflict_rate": value(mapping, "semantic_conflict_rate"),
                "mean_semantic_entropy": value(mapping, "mean_semantic_entropy"),
                "mean_normalized_semantic_entropy": value(mapping, "mean_normalized_semantic_entropy"),
                "valid_semantic_voxels": value(mapping, "valid_semantic_voxel_count"),
                "outlier_ratio_26n": value(mapping, "isolated_semantic_voxel_ratio"),
                "status": status,
                "notes": "No ATE/RPE: no independent trajectory ground truth; no scale correction or ICP.",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
