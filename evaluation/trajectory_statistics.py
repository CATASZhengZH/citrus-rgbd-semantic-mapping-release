#!/usr/bin/env python3
"""Compute backend-neutral trajectory coverage statistics."""

import os
import argparse
import csv
import math
from pathlib import Path

import numpy as np


def load_trajectory(path: Path):
    first = path.read_text().splitlines()[0].strip()
    if first.startswith("timestamp"):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        stamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
        xyz = np.asarray(
            [[float(row[axis]) for axis in ("tx", "ty", "tz")] for row in rows],
            dtype=np.float64,
        )
    else:
        values = np.loadtxt(path, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 4:
            raise ValueError(f"Invalid headerless trajectory: {path}")
        stamps, xyz = values[:, 0], values[:, 1:4]
    if np.any(np.diff(stamps) < 0):
        raise ValueError("Trajectory timestamps are not monotonic")
    return stamps, xyz


def load_frame_stamps(path: Path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No frame entries in {path}")
    key = next(
        (candidate for candidate in ("timestamp", "rgb_timestamp", "t_rgb") if candidate in rows[0]),
        None,
    )
    if key is None:
        raise ValueError(f"No RGB timestamp column in {path}")
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--max-dt", type=float, default=0.05)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    stamps, xyz = load_trajectory(args.trajectory)
    frame_stamps = load_frame_stamps(args.frames)
    indices = np.searchsorted(stamps, frame_stamps)
    right = np.clip(indices, 0, len(stamps) - 1)
    left = np.clip(indices - 1, 0, len(stamps) - 1)
    use_left = np.abs(frame_stamps - stamps[left]) <= np.abs(frame_stamps - stamps[right])
    nearest = np.where(use_left, left, right)
    errors = np.abs(frame_stamps - stamps[nearest])
    matched = errors <= args.max_dt

    duration = float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    path_length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    frequency = (len(stamps) - 1) / duration if duration > 0 else math.nan
    values = {
        "backend": args.backend,
        "trajectory_path": str(args.trajectory),
        "pose_count": len(stamps),
        "trajectory_duration_s": duration,
        "pose_frequency_hz": frequency,
        "path_length_m": path_length,
        "frame_count": len(frame_stamps),
        "valid_pose_count": int(matched.sum()),
        "valid_pose_rate": float(matched.mean()),
        "mean_match_error_s": float(errors[matched].mean()) if matched.any() else math.nan,
        "max_match_error_s": float(errors[matched].max()) if matched.any() else math.nan,
        "max_allowed_error_s": args.max_dt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=values.keys())
        writer.writeheader()
        writer.writerow(values)
    print(values)


if __name__ == "__main__":
    main()
