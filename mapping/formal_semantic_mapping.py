#!/usr/bin/env python3
"""Formal Orchard1 semantic mapping with frame-level voxel votes."""

from __future__ import annotations

import os
import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(os.environ.get("CITRUS_PAPER_ROOT", Path(__file__).resolve().parents[2]))
METRICS = ROOT / "metrics"
STAGE7_SCRIPTS = Path(os.environ.get("CITRUS_STAGE7_SCRIPTS", Path(__file__).resolve().parents[1] / "mapping"))
sys.path.insert(0, str(METRICS))
sys.path.insert(0, str(STAGE7_SCRIPTS))

from formal_semantic_sanity_check import BiSeNetPredictor, YoloPredictor, draw_orthographic  # noqa: E402
from full_sequence_world_mapping import read_frame_index, read_poses  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path):
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = protocol["classes"]
    ids = [int(item["id"]) for item in classes]
    if ids != list(range(len(ids))):
        raise ValueError(f"Formal classes must use contiguous IDs from zero: {ids}")
    if protocol["mapping"].get("per_class_point_cap") is not None:
        raise ValueError("Formal protocol prohibits per-class point caps")
    if not protocol["mapping"].get("frame_level_voxelization"):
        raise ValueError("Frame-level voxelization must be enabled")
    return protocol, classes


def camera_points(image_shape, depth_raw, mask, camera):
    height, width = image_shape[:2]
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    if depth_raw.shape[:2] != (height, width):
        depth_raw = cv2.resize(depth_raw, (width, height), interpolation=cv2.INTER_NEAREST)
    depth = depth_raw.astype(np.float32) * float(camera["depth_scale"])
    valid_depth = np.isfinite(depth) & (depth >= float(camera["min_depth_m"])) & (depth <= float(camera["max_depth_m"]))
    valid = valid_depth & (mask > 0)
    rows, columns = np.where(valid)
    if not len(columns):
        return np.empty((0, 3), np.float32), rows, columns, valid_depth
    z = depth[rows, columns]
    x = (columns.astype(np.float32) - float(camera["cx"])) * z / float(camera["fx"])
    y = (rows.astype(np.float32) - float(camera["cy"])) * z / float(camera["fy"])
    return np.column_stack((x, y, z)).astype(np.float32), rows, columns, valid_depth


def transform_world(camera_xyz, transform):
    if not len(camera_xyz):
        return camera_xyz
    homogeneous = np.column_stack((camera_xyz, np.ones(len(camera_xyz), np.float32)))
    return (transform @ homogeneous.T).T[:, :3].astype(np.float32)


def frame_voxel_votes(points_by_class, confidence_by_class, voxel_size):
    candidates = defaultdict(list)
    for class_id, points in points_by_class.items():
        if not len(points):
            continue
        voxel = np.floor(points / voxel_size).astype(np.int32)
        unique, inverse = np.unique(voxel, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        confidence_sum = np.bincount(inverse, weights=confidence_by_class[class_id])
        for index, coordinate in enumerate(unique):
            candidates[tuple(int(value) for value in coordinate)].append(
                (int(class_id), int(counts[index]), float(confidence_sum[index] / counts[index]))
            )
    return {
        coordinate: max(entries, key=lambda item: (item[1], item[2], -item[0]))[0]
        for coordinate, entries in candidates.items()
    }


class ObservationChunks:
    def __init__(self, directory: Path, chunk_frames=100):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk_frames = chunk_frames
        self.frame_indices, self.voxels, self.classes = [], [], []
        self.first_frame = None
        self.last_frame = None
        self.chunk_index = 0

    def append(self, frame_index, votes):
        if self.first_frame is None:
            self.first_frame = frame_index
        self.last_frame = frame_index
        for coordinate, class_id in votes.items():
            self.frame_indices.append(frame_index)
            self.voxels.append(coordinate)
            self.classes.append(class_id)
        if (frame_index - self.first_frame + 1) >= self.chunk_frames:
            self.flush()

    def flush(self):
        if self.first_frame is None:
            return
        path = self.directory / f"chunk_{self.chunk_index:04d}_{self.first_frame:06d}_{self.last_frame:06d}.npz"
        voxel_array = np.asarray(self.voxels, dtype=np.int32).reshape(-1, 3)
        np.savez_compressed(
            path,
            frame_index=np.asarray(self.frame_indices, dtype=np.int32),
            voxel_xyz=voxel_array,
            class_id=np.asarray(self.classes, dtype=np.uint8),
        )
        self.chunk_index += 1
        self.frame_indices, self.voxels, self.classes = [], [], []
        self.first_frame = None
        self.last_frame = None

    def close(self):
        self.flush()


def write_pcd(path: Path, xyz, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    bgr = np.asarray(bgr, dtype=np.uint8).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n")
        stream.write("FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
        stream.write(f"WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA ascii\n")
        for point, color in zip(xyz, bgr):
            packed = (int(color[2]) << 16) | (int(color[1]) << 8) | int(color[0])
            stream.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {packed}\n")


def summarize_metrics(histories, classes, total_valid_depth, total_semantic_depth):
    class_count = len(classes)
    observations = {voxel: int(sum(counts)) for voxel, counts in histories.items()}
    multi = [voxel for voxel, count in observations.items() if count >= 2]
    conflict = [voxel for voxel in multi if np.count_nonzero(histories[voxel]) > 1]
    entropy = []
    normalized = []
    for voxel in multi:
        counts = histories[voxel]
        probability = counts[counts > 0] / counts.sum()
        value = float(-(probability * np.log(probability)).sum())
        entropy.append(value)
        normalized.append(value / math.log(class_count))
    fused = {voxel: int(np.argmax(counts)) for voxel, counts in histories.items()}
    occupied = set(fused)
    isolated = 0
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    for x, y, z in occupied:
        if not any((x + dx, y + dy, z + dz) in occupied for dx, dy, dz in offsets):
            isolated += 1
    final_counts = Counter(fused.values())
    row = {
        "semantic_conflict_rate": len(conflict) / len(multi) if multi else 0.0,
        "mean_semantic_entropy": float(np.mean(entropy)) if entropy else 0.0,
        "normalized_semantic_entropy": float(np.mean(normalized)) if normalized else 0.0,
        "valid_semantic_voxel_count": len(fused),
        "multi_view_semantic_voxel_count": len(multi),
        "isolated_semantic_voxel_ratio": isolated / len(fused) if fused else 0.0,
        "non_background_coverage": total_semantic_depth / total_valid_depth if total_valid_depth else 0.0,
        "mean_observations_per_voxel": float(np.mean(list(observations.values()))) if observations else 0.0,
        "median_observations_per_voxel": float(np.median(list(observations.values()))) if observations else 0.0,
    }
    for item in classes:
        class_id, name = int(item["id"]), item["name"]
        row[f"{name}_voxel_count"] = final_counts[class_id]
        row[f"{name}_class_coverage"] = final_counts[class_id] / len(fused) if fused else 0.0
    return row, fused


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    protocol_path = args.protocol.resolve()
    protocol, classes = load_protocol(protocol_path)
    model_spec = protocol["models"][args.model]
    weight = Path(model_spec["weight"])
    if not weight.is_file():
        raise FileNotFoundError(weight)
    frames = read_frame_index(protocol["paths"]["frames_index"])
    poses = read_poses(protocol["paths"]["vins_poses"])
    if args.mode == "pilot":
        start, end = int(protocol["pilot"]["start_index"]), int(protocol["pilot"]["end_index"])
    else:
        start, end = int(protocol["full_sequence"]["start_index"]), int(protocol["full_sequence"]["end_index"])
    end = min(end, len(frames) - 1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    observation_dir = output / "frame_semantic_observations"
    if observation_dir.exists():
        shutil.rmtree(observation_dir)
    chunks = ObservationChunks(observation_dir)

    segmentation = protocol["segmentation"]
    if model_spec["kind"] == "bisenet":
        predictor = BiSeNetPredictor(weight, args.device)
    else:
        predictor = YoloPredictor(weight, args.device, float(segmentation["confidence_threshold"]), len(classes))
        predictor.iou_threshold = float(segmentation["iou_threshold"])

    # Formal YOLO calls are explicit and identical for every YOLO generation.
    if model_spec["kind"] == "yolo":
        original_call = predictor.__call__
        def predict_with_iou(image):
            # YoloPredictor owns all conversion logic; set the shared model-level override.
            predictor.model.overrides["iou"] = predictor.iou_threshold
            return original_call(image)
        predict = predict_with_iou
    else:
        predict = predictor

    histories = {}
    frame_rows = []
    ratio_rows = []
    total_valid_depth = 0
    total_semantic_depth = 0
    valid_pose_count = 0
    processed_count = 0
    start_time = time.perf_counter()
    voxel_size = float(protocol["mapping"]["voxel_size_m"])

    for ordinal, frame_index in enumerate(range(start, end + 1), 1):
        record = frames[frame_index]
        pose = poses.get(record["frame"])
        base = {
            "frame_index": frame_index, "frame": record["frame"], "timestamp": record["t_rgb"],
            "pose_valid": int(bool(pose and pose["valid"])), "pose_dt": pose["pose_dt"] if pose else "NA",
        }
        if not pose or not pose["valid"]:
            frame_rows.append({**base, "processed": 0, "semantic_voxel_count": 0, "citrus_voxels": 0, "tree_voxels": 0, "trunk_voxels": 0, "inference_ms": "NA", "raw_overlap_ratio": "NA"})
            continue
        valid_pose_count += 1
        image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        depth = cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:
            frame_rows.append({**base, "processed": 0, "semantic_voxel_count": 0, "citrus_voxels": 0, "tree_voxels": 0, "trunk_voxels": 0, "inference_ms": "NA", "raw_overlap_ratio": "NA"})
            continue
        before = time.perf_counter()
        semantic, masks, score_maps, overlap = predict(image)
        inference_ms = (time.perf_counter() - before) * 1000.0
        if semantic.shape != image.shape[:2]:
            raise RuntimeError(f"Semantic size mismatch at {record['frame']}: {semantic.shape} vs {image.shape[:2]}")

        points_by_class, confidence_by_class = {}, {}
        valid_depth_mask = None
        for item in classes:
            class_id = int(item["id"])
            camera_xyz, rows, columns, valid_depth = camera_points(image.shape, depth, masks[class_id], protocol["camera"])
            valid_depth_mask = valid_depth
            points_by_class[class_id] = transform_world(camera_xyz, pose["T"])
            confidence_by_class[class_id] = score_maps[class_id][rows, columns].astype(np.float32)
        votes = frame_voxel_votes(points_by_class, confidence_by_class, voxel_size)
        chunks.append(frame_index, votes)
        for coordinate, class_id in votes.items():
            counts = histories.get(coordinate)
            if counts is None:
                counts = np.zeros(len(classes), dtype=np.uint16)
                histories[coordinate] = counts
            if counts[class_id] < np.iinfo(np.uint16).max:
                counts[class_id] += 1
        semantic_valid = (semantic > 0) & valid_depth_mask
        total_valid_depth += int(valid_depth_mask.sum())
        total_semantic_depth += int(semantic_valid.sum())
        ratios = {item["name"]: float((semantic == int(item["id"]) + 1).mean()) for item in classes}
        ratios["background"] = float((semantic == 0).mean())
        ratio_rows.append({"model": args.model, "frame_index": frame_index, "frame": record["frame"], **{f"{name}_pixel_ratio": value for name, value in ratios.items()}})
        frame_class_counts = Counter(votes.values())
        frame_rows.append({
            **base, "processed": 1, "semantic_voxel_count": len(votes),
            "citrus_voxels": frame_class_counts[0], "tree_voxels": frame_class_counts[1], "trunk_voxels": frame_class_counts[2],
            "inference_ms": f"{inference_ms:.3f}", "raw_overlap_ratio": f"{overlap:.9f}",
        })
        processed_count += 1
        if ordinal % 100 == 0:
            print(f"{args.model} {args.mode}: {ordinal}/{end-start+1} processed={processed_count} global_voxels={len(histories)}", flush=True)

    chunks.close()
    metrics, fused = summarize_metrics(histories, classes, total_valid_depth, total_semantic_depth)
    class_xyz, class_colors = defaultdict(list), defaultdict(list)
    all_xyz, all_colors = [], []
    for coordinate, class_id in fused.items():
        center = (np.asarray(coordinate, np.float32) + 0.5) * voxel_size
        color = protocol["classes"][class_id]["bgr"]
        class_xyz[class_id].append(center)
        class_colors[class_id].append(color)
        all_xyz.append(center); all_colors.append(color)
    for item in classes:
        class_id, name = int(item["id"]), item["name"]
        write_pcd(output / f"global_world_{name}_cloud.pcd", class_xyz[class_id], class_colors[class_id])
    write_pcd(output / "global_world_all_class_semantic_cloud.pcd", all_xyz, all_colors)
    write_csv(output / "frame_pose_mapping_stats.csv", frame_rows)
    write_csv(output / "frame_class_ratios.csv", ratio_rows)
    elapsed = time.perf_counter() - start_time
    metrics.update({
        "model": args.model, "mode": args.mode, "start_index": start, "end_index": end,
        "selected_frame_count": end - start + 1, "valid_pose_count": valid_pose_count,
        "processed_frame_count": processed_count, "pose_match_rate": valid_pose_count / (end - start + 1),
        "voxel_size_m": voxel_size, "runtime_seconds": elapsed, "weight": str(weight), "weight_sha256": sha256(weight),
        "fallback_used": False, "per_class_point_cap": "none",
    })
    write_csv(output / "mapping_metrics.csv", [metrics])
    run_config = {
        "protocol": str(protocol_path), "model": args.model, "weight": str(weight), "weight_sha256": metrics["weight_sha256"],
        "device": args.device, "mode": args.mode, "frame_range": [start, end], "fallback_used": False,
        "mapping": protocol["mapping"], "segmentation": protocol["segmentation"], "classes": classes,
    }
    (output / "run_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    (output / "runtime_log.json").write_text(json.dumps({"runtime_seconds": elapsed, "processed_fps": processed_count / elapsed if elapsed else 0.0}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/FORMAL_MAPPING_PROTOCOL.yaml")
    parser.add_argument("--model", choices=("bisenet", "yolov8n", "yolo11n", "yolo26n"), required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
