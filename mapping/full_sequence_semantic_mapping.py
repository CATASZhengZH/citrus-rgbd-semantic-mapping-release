#!/usr/bin/env python3
"""Reuse Stage7 projection primitives for class-generic Orchard1 mapping."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import yaml


STAGE7_SCRIPTS = Path(os.environ.get("CITRUS_STAGE7_SCRIPTS", Path(__file__).resolve().parents[1] / "mapping"))
sys.path.insert(0, str(STAGE7_SCRIPTS))
from full_sequence_world_mapping import PcdStream, read_frame_index, read_poses, reconstruct_points, yolo_masks  # noqa: E402


class YoloBackend:
    def __init__(self, model_path, class_ids, confidence, device):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.class_ids = class_ids
        self.confidence = confidence

    def __call__(self, image):
        return yolo_masks(image, self.model, self.confidence, self.class_ids)


class BiSeNetBackend:
    def __init__(self, model_path, class_ids, repository, device):
        sys.path.insert(0, str(repository))
        from lib.models import model_factory

        self.class_ids = class_ids
        self.device = torch.device(device)
        self.model = model_factory["bisenetv1"](len(class_ids) + 1)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        self.model.aux_mode = "eval"
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.4, 0.4, 0.4], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2, 0.2, 0.2], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        logits = self.model((tensor - self.mean) / self.std)[0]
        if logits.shape[-2:] != image.shape[:2]:
            logits = functional.interpolate(logits, image.shape[:2], mode="bilinear", align_corners=False)
        probabilities = logits.softmax(1)[0]
        semantic = probabilities.argmax(0).byte().cpu().numpy()
        confidence = probabilities.max(0).values.float().cpu().numpy()
        masks = {}
        scores = {}
        for class_id in self.class_ids:
            mask = (semantic == class_id + 1).astype(np.uint8)
            masks[class_id] = mask
            scores[class_id] = np.where(mask > 0, confidence, 0.0).astype(np.float32)
        return masks, scores


def frame_voxel_votes(points_by_class, scores_by_class, voxel_size):
    votes = {}
    for class_id, points in points_by_class.items():
        if len(points) == 0:
            continue
        voxels = np.floor(points / voxel_size).astype(np.int64)
        unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        score_sums = np.bincount(inverse, weights=scores_by_class[class_id])
        for index, voxel in enumerate(unique):
            key = tuple(int(value) for value in voxel)
            votes.setdefault(key, []).append((class_id, int(counts[index]), float(score_sums[index] / counts[index])))
    selected = []
    for key, candidates in votes.items():
        # One frame contributes exactly one semantic vote to a voxel.
        class_id, point_count, mean_confidence = max(candidates, key=lambda item: (item[1], item[2], -item[0]))
        selected.append((*key, class_id, point_count, mean_confidence))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames-index", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-type", choices=("yolo", "bisenet"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--save-overlays-every", type=int, default=500)
    parser.add_argument("--bisenet-repository", type=Path, default=Path(os.environ.get("BISENET_REPOSITORY", "third_party/BiSeNet-model-conversion")))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frames_index = args.frames_index or Path(config["paths"]["frames_index"])
    poses_path = args.poses or Path(config["paths"]["vins_poses"])
    frames = read_frame_index(frames_index)
    if args.limit_frames:
        frames = frames[: args.limit_frames]
    poses = read_poses(poses_path)
    classes = config["classes"]
    class_ids = [int(item["id"]) for item in classes]
    class_by_id = {int(item["id"]): item for item in classes}
    confidence = float(config["reconstruction"]["segmentation_confidence"])
    voxel_size = float(config["reconstruction"]["semantic_voxel_size_m"])
    max_points = int(config["reconstruction"]["max_points_per_class_per_frame"])

    if args.model_type == "yolo":
        backend = YoloBackend(args.model, class_ids, confidence, args.device)
    else:
        backend = BiSeNetBackend(args.model, class_ids, args.bisenet_repository, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output_dir / "run_config.yaml")
    overlay_dir = args.output_dir / "overlays"
    mask_dir = args.output_dir / "semantic_masks"
    if args.save_overlays_every > 0:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
    global_pcd = PcdStream(args.output_dir / "global_semantic_cloud.pcd")
    class_pcd = {
        class_id: PcdStream(args.output_dir / f"global_world_{class_by_id[class_id]['name']}_cloud.pcd")
        for class_id in class_ids
    }

    stats_path = args.output_dir / "frame_pose_mapping_stats.csv"
    observations_path = args.output_dir / "frame_semantic_voxel_observations.csv"
    start_time = time.perf_counter()
    valid_pose_count = 0
    valid_frame_count = 0
    total_points = {class_id: 0 for class_id in class_ids}
    observation_rows = 0

    with stats_path.open("w", newline="", encoding="utf-8") as stats_stream, observations_path.open(
        "w", newline="", encoding="utf-8"
    ) as observation_stream:
        stats_writer = csv.writer(stats_stream)
        observation_writer = csv.writer(observation_stream)
        stats_writer.writerow(
            ["frame", "timestamp", "pose_valid", "pose_dt", "image_valid"]
            + [f"{class_by_id[class_id]['name']}_points" for class_id in class_ids]
            + ["rgb_path", "depth_path"]
        )
        observation_writer.writerow(["frame", "timestamp", "voxel_x", "voxel_y", "voxel_z", "class_id", "point_count", "mean_confidence"])

        for frame_index, record in enumerate(frames):
            frame_name = record["frame"]
            timestamp = float(record["t_rgb"])
            pose = poses.get(frame_name)
            zeros = [0] * len(class_ids)
            if pose is None or not pose["valid"]:
                stats_writer.writerow([frame_name, f"{timestamp:.9f}", 0, "nan", 0, *zeros, record["rgb_path"], record["depth_path"]])
                continue
            valid_pose_count += 1
            image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
            depth = cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED)
            if image is None or depth is None:
                stats_writer.writerow([frame_name, f"{timestamp:.9f}", 1, f"{pose['pose_dt']:.9f}", 0, *zeros, record["rgb_path"], record["depth_path"]])
                continue
            masks, score_maps = backend(image)
            points_by_class = {}
            scores_by_class = {}
            frame_counts = []
            for class_id in class_ids:
                points, scores = reconstruct_points(
                    image.shape,
                    depth,
                    masks[class_id],
                    score_maps[class_id],
                    pose["T"],
                    {"camera": config["camera"]},
                    class_id,
                    max_points,
                )
                points_by_class[class_id] = points
                scores_by_class[class_id] = scores
                color = class_by_id[class_id]["bgr"]
                class_pcd[class_id].write(points, color)
                global_pcd.write(points, color)
                total_points[class_id] += len(points)
                frame_counts.append(len(points))

            for vx, vy, vz, class_id, point_count, mean_confidence in frame_voxel_votes(points_by_class, scores_by_class, voxel_size):
                observation_writer.writerow(
                    [frame_name, f"{timestamp:.9f}", vx, vy, vz, class_id, point_count, f"{mean_confidence:.6f}"]
                )
                observation_rows += 1
            valid_frame_count += 1
            stats_writer.writerow(
                [frame_name, f"{timestamp:.9f}", 1, f"{pose['pose_dt']:.9f}", 1, *frame_counts, record["rgb_path"], record["depth_path"]]
            )

            if args.save_overlays_every > 0 and frame_index % args.save_overlays_every == 0:
                color_mask = np.zeros_like(image)
                for class_id in class_ids:
                    color_mask[masks[class_id] > 0] = class_by_id[class_id]["bgr"]
                overlay = cv2.addWeighted(image, 0.70, color_mask, 0.30, 0.0)
                cv2.imwrite(str(mask_dir / f"{frame_name}_mask.png"), color_mask)
                cv2.imwrite(str(overlay_dir / f"{frame_name}_overlay.jpg"), overlay)
            if (frame_index + 1) % 100 == 0:
                counts_text = " ".join(f"{class_by_id[c]['name']}={total_points[c]}" for c in class_ids)
                print(f"frames={frame_index + 1}/{len(frames)} valid={valid_frame_count} {counts_text}", flush=True)

    global_pcd.close()
    for stream in class_pcd.values():
        stream.close()
    elapsed = time.perf_counter() - start_time
    summary_path = args.output_dir / "mapping_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "frame_count", "valid_pose_count", "valid_frame_count", "pose_match_rate", "runtime_seconds",
            "semantic_voxel_size_m", "frame_voxel_observation_rows",
        ] + [f"{class_by_id[class_id]['name']}_point_count" for class_id in class_ids]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        row = {
            "frame_count": len(frames),
            "valid_pose_count": valid_pose_count,
            "valid_frame_count": valid_frame_count,
            "pose_match_rate": valid_pose_count / len(frames) if frames else 0.0,
            "runtime_seconds": elapsed,
            "semantic_voxel_size_m": voxel_size,
            "frame_voxel_observation_rows": observation_rows,
        }
        row.update({f"{class_by_id[class_id]['name']}_point_count": total_points[class_id] for class_id in class_ids})
        writer.writerow(row)
    with (args.output_dir / "runtime.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "model_type", "model", "device", "runtime_seconds", "processed_frame_count",
            "segmentation_success_count", "valid_pose_count", "pose_match_rate",
        ])
        writer.writeheader()
        writer.writerow({
            "model_type": args.model_type,
            "model": str(args.model),
            "device": args.device,
            "runtime_seconds": elapsed,
            "processed_frame_count": valid_frame_count,
            "segmentation_success_count": valid_frame_count,
            "valid_pose_count": valid_pose_count,
            "pose_match_rate": valid_pose_count / len(frames) if frames else 0.0,
        })
    observation_alias = args.output_dir / "frame_semantic_observations.csv"
    if not observation_alias.exists():
        os.symlink(observations_path.name, observation_alias)
    print(f"output={args.output_dir} runtime_seconds={elapsed:.3f} valid_frames={valid_frame_count}")


if __name__ == "__main__":
    main()
