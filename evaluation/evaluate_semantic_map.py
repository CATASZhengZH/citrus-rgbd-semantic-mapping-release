#!/usr/bin/env python3
"""Compute frame-balanced multi-view semantic consistency metrics."""

from __future__ import annotations

import os
import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml


NEIGHBORS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


def parse_run(value: str):
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Run must be NAME=OBSERVATIONS_CSV")
    return parts[0], Path(parts[1])


def evaluate(path: Path, class_ids, class_names):
    votes = defaultdict(Counter)
    observed_frames = set()
    with path.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            observed_frames.add(row["frame"])
            voxel = (int(row["voxel_x"]), int(row["voxel_y"]), int(row["voxel_z"]))
            votes[voxel][int(row["class_id"])] += 1
    if not votes:
        raise ValueError(f"No semantic observations in {path}")

    final_class = {}
    observation_counts = []
    multi_view_counts = []
    conflicts = 0
    entropy_values = []
    normalized_entropy_values = []
    for voxel, class_votes in votes.items():
        final_class[voxel] = max(class_votes.items(), key=lambda item: (item[1], -item[0]))[0]
        count = sum(class_votes.values())
        observation_counts.append(count)
        if count < 2:
            continue
        multi_view_counts.append(count)
        if len(class_votes) > 1:
            conflicts += 1
        entropy = -sum((value / count) * math.log(value / count) for value in class_votes.values())
        entropy_values.append(entropy)
        normalized_entropy_values.append(entropy / math.log(len(class_ids)) if len(class_ids) > 1 else 0.0)

    occupied = set(final_class)
    isolated = sum(
        1
        for x, y, z in occupied
        if not any((x + dx, y + dy, z + dz) in occupied for dx, dy, dz in NEIGHBORS_26)
    )
    per_class = Counter(final_class.values())
    per_class_rows = []
    for class_id in class_ids:
        relevant = [(voxel, class_votes) for voxel, class_votes in votes.items() if class_id in class_votes]
        class_multi = [(voxel, class_votes) for voxel, class_votes in relevant if sum(class_votes.values()) >= 2]
        class_conflicts = sum(1 for _, class_votes in class_multi if len(class_votes) > 1)
        class_frame_votes = sum(class_votes[class_id] for _, class_votes in relevant)
        per_class_rows.append({
            "class_id": class_id,
            "class_name": class_names[class_id],
            "final_voxel_count": per_class[class_id],
            "frame_vote_count": class_frame_votes,
            "candidate_voxel_count": len(relevant),
            "multi_view_candidate_voxel_count": len(class_multi),
            "conflict_rate_when_observed": class_conflicts / len(class_multi) if class_multi else 0.0,
            "final_voxel_fraction": per_class[class_id] / len(occupied) if occupied else 0.0,
        })
    multi_view_voxels = len(multi_view_counts)
    valid_frame_count = 0
    summary_path = path.parent / "mapping_summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as stream:
            summary_rows = list(csv.DictReader(stream))
        if summary_rows:
            valid_frame_count = int(summary_rows[0].get("valid_frame_count", 0))
    row = {
        "valid_semantic_voxel_count": len(occupied),
        "multi_view_semantic_voxel_count": multi_view_voxels,
        "semantic_conflict_rate": conflicts / multi_view_voxels if multi_view_voxels else 0.0,
        "mean_semantic_entropy": statistics.fmean(entropy_values) if entropy_values else 0.0,
        "mean_normalized_semantic_entropy": statistics.fmean(normalized_entropy_values) if normalized_entropy_values else 0.0,
        "isolated_semantic_voxel_ratio": isolated / len(occupied),
        "mean_observations_per_voxel": statistics.fmean(observation_counts),
        "median_observations_per_voxel": statistics.median(observation_counts),
        "mean_observations_per_multiview_voxel": statistics.fmean(multi_view_counts) if multi_view_counts else 0.0,
        "median_observations_per_multiview_voxel": statistics.median(multi_view_counts) if multi_view_counts else 0.0,
        "observed_class_count": sum(per_class[class_id] > 0 for class_id in class_ids),
        "class_coverage": sum(per_class[class_id] > 0 for class_id in class_ids) / len(class_ids),
        "frames_with_semantic_observation": len(observed_frames),
        "non_background_frame_coverage": len(observed_frames) / valid_frame_count if valid_frame_count else 0.0,
        "outlier_ratio": isolated / len(occupied),
    }
    for class_id in class_ids:
        row[f"{class_names[class_id]}_voxel_count"] = per_class[class_id]
    collapse_reasons = []
    if row["class_coverage"] < 1.0:
        collapse_reasons.append("missing_predicted_class")
    if row["non_background_frame_coverage"] < 0.5:
        collapse_reasons.append("low_non_background_frame_coverage")
    row["collapse_sanity_flag"] = ";".join(collapse_reasons) if collapse_reasons else "ok"
    return row, per_class_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class-output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    class_ids = [int(item["id"]) for item in config["classes"]]
    class_names = {int(item["id"]): item["name"] for item in config["classes"]}
    rows = []
    per_class_output_rows = []
    for name, path in args.run:
        row = {"run": name, "observations": str(path), "voxel_size_m": config["reconstruction"]["semantic_voxel_size_m"]}
        metrics, class_rows = evaluate(path, class_ids, class_names)
        row.update(metrics)
        rows.append(row)
        for class_row in class_rows:
            per_class_output_rows.append({"run": name, "observations": str(path), **class_row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.per_class_output:
        args.per_class_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_class_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_class_output_rows[0]))
            writer.writeheader()
            writer.writerows(per_class_output_rows)


if __name__ == "__main__":
    main()
