#!/usr/bin/env python3
"""Read-only formal-model semantic sanity check on fixed RGB-D frames."""

from __future__ import annotations

import os
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("CITRUS_PAPER_ROOT", Path(__file__).resolve().parents[2]))
STAGE7 = Path(os.environ.get("CITRUS_STAGE7_ROOT", Path(__file__).resolve().parents[2] / "external/stage7_yield_estimation"))
STAGE7_SCRIPTS = STAGE7 / "scripts"
sys.path.insert(0, str(STAGE7_SCRIPTS))
from full_sequence_world_mapping import PcdStream, read_frame_index, read_poses, reconstruct_points  # noqa: E402


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DRAW_ORDER = [1, 2, 0]  # tree, trunk, citrus; details overwrite canopy.


def pil_font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def load_formal_config():
    mapping = yaml.safe_load((ROOT / "configs/class_mapping.yaml").read_text(encoding="utf-8"))
    colors = yaml.safe_load((ROOT / "configs/segmentation_colors.yaml").read_text(encoding="utf-8"))
    world = yaml.safe_load((ROOT / "configs/world_mapping.yaml").read_text(encoding="utf-8"))
    classes = []
    for item in mapping["classes"]:
        entry = colors[item["name"]]
        if int(entry["id"]) != int(item["id"]):
            raise ValueError(f"Class/color ID mismatch for {item['name']}")
        classes.append({"id": int(item["id"]), "name": item["name"], "bgr": tuple(int(x) for x in entry["bgr"])})
    return classes, colors["background"]["bgr"], world


class BiSeNetPredictor:
    def __init__(self, weight, device):
        repository = Path(os.environ.get("BISENET_REPOSITORY", "third_party/BiSeNet-model-conversion"))
        sys.path.insert(0, str(repository))
        from lib.models import model_factory

        self.device = torch.device(device)
        self.model = model_factory["bisenetv1"](4)
        self.model.load_state_dict(torch.load(weight, map_location="cpu"), strict=True)
        self.model.aux_mode = "eval"
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.4, 0.4, 0.4], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2, 0.2, 0.2], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(self.device)
        logits = self.model((tensor - self.mean) / self.std)[0]
        if logits.shape[-2:] != image.shape[:2]:
            logits = functional.interpolate(logits, image.shape[:2], mode="bilinear", align_corners=False)
        probabilities = logits.softmax(1)[0]
        semantic = probabilities.argmax(0).byte().cpu().numpy()
        confidence = probabilities.max(0).values.float().cpu().numpy()
        masks = {cid: (semantic == cid + 1).astype(np.uint8) for cid in range(3)}
        scores = {cid: np.where(masks[cid] > 0, confidence, 0).astype(np.float32) for cid in range(3)}
        return semantic, masks, scores, 0.0


class YoloPredictor:
    def __init__(self, weight, device, confidence=0.25, expected_classes=3):
        from ultralytics import YOLO

        self.model = YOLO(str(weight))
        self.device = device
        self.confidence = confidence
        self.expected_classes = expected_classes
        self.names = dict(self.model.names)

    def __call__(self, image):
        height, width = image.shape[:2]
        result = self.model.predict(
            image, imgsz=640, conf=self.confidence, device=self.device,
            retina_masks=True, verbose=False,
        )[0]
        masks = {cid: np.zeros((height, width), np.uint8) for cid in range(self.expected_classes)}
        scores = {cid: np.zeros((height, width), np.float32) for cid in range(self.expected_classes)}
        if result.masks is not None and result.boxes is not None:
            raw_masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for raw, class_id, score in zip(raw_masks, classes, confidences):
                class_id = int(class_id)
                if class_id not in masks:
                    continue
                if raw.shape != (height, width):
                    raw = cv2.resize(raw.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
                region = raw > 0.5
                update = region & (score >= scores[class_id])
                masks[class_id][update] = 1
                scores[class_id][update] = float(score)
        overlap = sum(masks.values()) > 1
        semantic = np.zeros((height, width), np.uint8)
        for class_id in (cid for cid in DRAW_ORDER if cid < self.expected_classes):
            semantic[masks[class_id] > 0] = class_id + 1
        final_masks = {class_id: (semantic == class_id + 1).astype(np.uint8) for class_id in masks}
        final_scores = {class_id: np.where(final_masks[class_id], scores[class_id], 0).astype(np.float32) for class_id in masks}
        return semantic, final_masks, final_scores, float(overlap.mean())


def image_scene_score(image, depth, pose_valid):
    small = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    orange = ((h <= 28) & (s >= 70) & (v >= 65)).mean()
    green = ((h >= 30) & (h <= 95) & (s >= 40) & (v >= 35)).mean()
    brown = ((h <= 28) & (s >= 35) & (v >= 25) & (v <= 185)).mean()
    valid_depth = ((depth > 300) & (depth < 2500)).mean() if depth is not None else 0.0
    return 4.0 * orange + green + 0.7 * brown + 0.25 * valid_depth + 0.15 * float(pose_valid), orange, green, brown, valid_depth


def score_frames(frames, poses):
    rows = []
    for index, record in enumerate(frames):
        image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        depth = cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED)
        pose = poses.get(record["frame"])
        if image is None or depth is None:
            rows.append((index, -1.0, 0, 0, 0, 0, False))
            continue
        score, orange, green, brown, valid_depth = image_scene_score(image, depth, bool(pose and pose["valid"]))
        rows.append((index, score, orange, green, brown, valid_depth, bool(pose and pose["valid"])))
    return rows


def choose_spaced(candidates, count, spacing=35):
    selected = []
    for row in sorted(candidates, key=lambda item: item[1], reverse=True):
        if row[6] and all(abs(row[0] - other[0]) >= spacing for other in selected):
            selected.append(row)
            if len(selected) == count:
                break
    if len(selected) < count:
        for row in sorted(candidates, key=lambda item: item[1], reverse=True):
            if row[6] and row not in selected:
                selected.append(row)
                if len(selected) == count:
                    break
    return sorted(selected, key=lambda item: item[0])


def select_frames(frames, scores):
    n = len(frames)
    thirds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    selected = []
    for begin, end in thirds:
        selected.extend(choose_spaced([row for row in scores if begin <= row[0] < end], 10))

    values = np.array([row[1] if row[6] else -2.0 for row in scores], dtype=np.float64)
    best_start, best_value = 0, -np.inf
    for start in range(0, n - 200 + 1, 5):
        window = values[start : start + 200]
        valid = window > -1.0
        if valid.mean() < 0.97:
            continue
        value = window[valid].mean() + 0.2 * np.percentile(window[valid], 80)
        if value > best_value:
            best_value, best_start = value, start
    return selected, list(range(best_start, best_start + 200)), best_value


def colorize_semantic(semantic, classes):
    color = np.zeros((*semantic.shape, 3), np.uint8)
    for item in classes:
        color[semantic == item["id"] + 1] = item["bgr"]
    return color


def append_legend(image, classes, ratios=None, warning=""):
    bar_height = 64 if ratios is None else 92
    canvas = np.full((image.shape[0] + bar_height, image.shape[1], 3), 255, np.uint8)
    canvas[: image.shape[0]] = image
    entries = [("background", (0, 0, 0))] + [(item["name"], item["bgr"]) for item in classes]
    x, y = 10, image.shape[0] + 12
    for name, bgr in entries:
        cv2.rectangle(canvas, (x, y), (x + 18, y + 18), bgr, -1)
        cv2.rectangle(canvas, (x, y), (x + 18, y + 18), (80, 80, 80), 1)
        cv2.putText(canvas, name, (x + 24, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
        x += 118
    if ratios is not None:
        text = "  ".join(f"{key}={value:.1%}" for key, value in ratios.items())
        cv2.putText(canvas, text, (10, image.shape[0] + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (35, 35, 35), 1, cv2.LINE_AA)
        if warning:
            cv2.putText(canvas, warning[:95], (10, image.shape[0] + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 80, 220), 1, cv2.LINE_AA)
    return canvas


def depth_preview(depth, world):
    scale = float(world["camera"]["depth_scale"])
    meters = depth.astype(np.float32) * scale
    valid = (meters >= float(world["camera"]["min_depth_m"])) & (meters <= float(world["camera"]["max_depth_m"]))
    normalized = np.zeros_like(depth, dtype=np.uint8)
    lo, hi = float(world["camera"]["min_depth_m"]), float(world["camera"]["max_depth_m"])
    normalized[valid] = np.clip((meters[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color, valid


def warning_flags(ratios, green_ratio, overlap):
    flags = []
    if max(ratios.values()) > 0.90:
        flags.append(f"single_class_gt90:{max(ratios, key=ratios.get)}")
    if ratios["citrus"] > 0.40 and green_ratio > 0.25:
        flags.append("citrus_canopy_like_area")
    if ratios["trunk"] == 0:
        flags.append("trunk_zero")
    if ratios["background"] > 0.999:
        flags.append("all_background")
    if overlap > 0.01:
        flags.append("raw_class_overlap_gt1pct")
    return flags


def save_30frame_outputs(model_name, predictor, selected, frames, scores_by_index, classes, world, output_root, stats):
    model_dir = output_root / "30frames" / model_name
    for index in selected:
        record = frames[index]
        image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        depth = cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED)
        semantic, masks, score_maps, overlap = predictor(image)
        ratios = {item["name"]: float((semantic == item["id"] + 1).mean()) for item in classes}
        ratios["background"] = float((semantic == 0).mean())
        green_ratio = scores_by_index[index][3]
        flags = warning_flags(ratios, green_ratio, overlap)
        color_mask = colorize_semantic(semantic, classes)
        overlay = cv2.addWeighted(image, 0.68, color_mask, 0.32, 0)
        depth_color, valid_depth = depth_preview(depth, world)
        depth_semantic = color_mask.copy()
        depth_semantic[~valid_depth] = 0
        frame_dir = model_dir / record["frame"]
        frame_dir.mkdir(parents=True, exist_ok=True)
        warning = ";".join(flags)
        products = {
            "rgb": append_legend(image, classes),
            "semantic_mask": append_legend(color_mask, classes, ratios, warning),
            "overlay": append_legend(overlay, classes, ratios, warning),
            "depth": append_legend(depth_color, classes),
            "valid_depth_semantic": append_legend(depth_semantic, classes, ratios, warning),
        }
        for suffix, product in products.items():
            cv2.imwrite(str(frame_dir / f"{record['frame']}_{suffix}.png"), product)
        stats.append({
            "model": model_name, "frame": record["frame"], "frame_index": index,
            "citrus_pixel_ratio": ratios["citrus"], "tree_pixel_ratio": ratios["tree"],
            "trunk_pixel_ratio": ratios["trunk"], "background_ratio": ratios["background"],
            "valid_depth_ratio": float(valid_depth.mean()), "raw_cross_class_overlap_ratio": overlap,
            "warnings": warning or "OK",
        })


def frame_voxel_votes(points_by_class, scores_by_class, voxel_size):
    candidates = defaultdict(list)
    for class_id, points in points_by_class.items():
        if not len(points):
            continue
        voxels = np.floor(points / voxel_size).astype(np.int64)
        unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=scores_by_class[class_id])
        for idx, voxel in enumerate(unique):
            candidates[tuple(int(v) for v in voxel)].append((class_id, int(counts[idx]), float(sums[idx] / counts[idx])))
    return {voxel: max(items, key=lambda item: (item[1], item[2], -item[0]))[0] for voxel, items in candidates.items()}


def map_metrics(frame_votes, class_voxels, point_counts, class_names):
    histories = defaultdict(Counter)
    observations = defaultdict(int)
    for votes in frame_votes:
        for voxel, class_id in votes.items():
            histories[voxel][class_id] += 1
            observations[voxel] += 1
    multi = [voxel for voxel, count in observations.items() if count >= 2]
    conflicts = [voxel for voxel in multi if len(histories[voxel]) > 1]
    entropies = []
    for voxel in multi:
        total = sum(histories[voxel].values())
        entropy = -sum((count / total) * math.log(count / total) for count in histories[voxel].values())
        entropies.append(entropy / math.log(len(class_names)))
    citrus, tree = 0, 1
    return {
        "semantic_conflict_rate": len(conflicts) / len(multi) if multi else 0.0,
        "normalized_semantic_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "multi_view_voxel_count": len(multi),
        "all_semantic_voxel_count": len(histories),
        "citrus_tree_point_ratio": point_counts[citrus] / point_counts[tree] if point_counts[tree] else "NA",
        "citrus_tree_voxel_ratio": len(class_voxels[citrus]) / len(class_voxels[tree]) if class_voxels[tree] else "NA",
        **{f"{class_names[cid]}_world_point_count": point_counts[cid] for cid in point_counts},
        **{f"{class_names[cid]}_world_voxel_count": len(class_voxels[cid]) for cid in class_voxels},
    }


def draw_orthographic(points, colors, title, axes, size=(900, 620)):
    width, height = size
    canvas = np.full((height, width, 3), 255, np.uint8)
    if not len(points):
        cv2.putText(canvas, "EMPTY", (width // 2 - 80, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)
    else:
        values = points[:, axes]
        low, high = np.percentile(values, 0.2, axis=0), np.percentile(values, 99.8, axis=0)
        span = np.maximum(high - low, 1e-6)
        scale = min((width - 70) / span[0], (height - 100) / span[1])
        x = ((values[:, 0] - low[0]) * scale + 35).astype(int)
        y = (height - 45 - (values[:, 1] - low[1]) * scale).astype(int)
        valid = (x >= 1) & (x < width - 1) & (y >= 55) & (y < height - 1)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            canvas[y[valid] + dy, x[valid] + dx] = colors[valid]
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (35, 35, 35), 2, cv2.LINE_AA)
    return canvas


def save_200frame_map(model_name, predictor, indices, frames, poses, classes, world, output_root):
    map_dir = output_root / "200frames" / model_name
    map_dir.mkdir(parents=True, exist_ok=True)
    class_ids = [item["id"] for item in classes]
    class_names = {item["id"]: item["name"] for item in classes}
    pcds = {cid: PcdStream(map_dir / f"{class_names[cid]}_only.pcd") for cid in class_ids}
    all_pcd = PcdStream(map_dir / "all_class.pcd")
    preview_points = defaultdict(list)
    frame_votes = []
    class_voxels = {cid: set() for cid in class_ids}
    point_counts = Counter()
    processed = 0
    voxel_size = float(world["reconstruction"]["semantic_voxel_size_m"])
    max_points = int(world["reconstruction"]["max_points_per_class_per_frame"])
    for sequence_index, index in enumerate(indices, 1):
        record = frames[index]
        pose = poses.get(record["frame"])
        if not pose or not pose["valid"]:
            continue
        image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        depth = cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:
            continue
        semantic, masks, score_maps, overlap = predictor(image)
        points_by_class, scores_by_class = {}, {}
        for item in classes:
            cid = item["id"]
            points, scores = reconstruct_points(
                image.shape, depth, masks[cid], score_maps[cid], pose["T"], {"camera": world["camera"]}, cid, max_points,
            )
            points_by_class[cid], scores_by_class[cid] = points, scores
            pcds[cid].write(points, item["bgr"])
            all_pcd.write(points, item["bgr"])
            point_counts[cid] += len(points)
            if len(points):
                preview_points[cid].append(points)
                class_voxels[cid].update(map(tuple, np.floor(points / voxel_size).astype(np.int64)))
        frame_votes.append(frame_voxel_votes(points_by_class, scores_by_class, voxel_size))
        processed += 1
        if sequence_index % 50 == 0:
            print(f"{model_name}: map {sequence_index}/{len(indices)}", flush=True)
    for stream in pcds.values():
        stream.close()
    all_pcd.close()

    combined_points, combined_colors = [], []
    triptychs = {}
    for item in classes:
        cid = item["id"]
        points = np.concatenate(preview_points[cid]) if preview_points[cid] else np.empty((0, 3), np.float32)
        if len(points) > 300_000:
            points = points[:: math.ceil(len(points) / 300_000)]
        colors = np.tile(np.array(item["bgr"], np.uint8), (len(points), 1))
        combined_points.append(points)
        combined_colors.append(colors)
        triptychs[item["name"]] = save_map_views(map_dir, model_name, item["name"], points, colors)
    all_points = np.concatenate(combined_points) if combined_points else np.empty((0, 3), np.float32)
    all_colors = np.concatenate(combined_colors) if combined_colors else np.empty((0, 3), np.uint8)
    if len(all_points) > 450_000:
        stride = math.ceil(len(all_points) / 450_000)
        all_points, all_colors = all_points[::stride], all_colors[::stride]
    triptychs["all_class"] = save_map_views(map_dir, model_name, "all_class", all_points, all_colors)
    metrics = map_metrics(frame_votes, class_voxels, point_counts, class_names)
    metrics.update({"model": model_name, "selected_frame_count": len(indices), "processed_frame_count": processed})
    return metrics, triptychs


def save_map_views(map_dir, model_name, class_name, points, colors_rgb):
    files = []
    for suffix, axes in (("bev", [0, 1]), ("side", [0, 2]), ("front", [1, 2])):
        image = draw_orthographic(points, colors_rgb, f"{model_name} | {class_name} | {suffix}", axes)
        path = map_dir / f"{class_name}_{suffix}.png"
        cv2.imwrite(str(path), image)
        files.append(path)
    canvas = Image.new("RGB", (900, 620 * 3), "white")
    for index, path in enumerate(files):
        canvas.paste(Image.open(path).convert("RGB"), (0, index * 620))
    triptych = map_dir / f"{class_name}_triptych.png"
    canvas.save(triptych, optimize=True)
    return triptych


def create_30_contact(selected_indices, frames, model_names, output_root):
    tile_w, tile_h = 350, 265
    label_w = 120
    sheet = Image.new("RGB", (label_w + tile_w * len(model_names), 70 + tile_h * len(selected_indices)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 14), "Fixed 30-frame formal-model semantic overlays", fill=(25, 30, 35), font=pil_font(25, True))
    for column, model in enumerate(model_names):
        draw.text((label_w + column * tile_w + 8, 45), model, fill=(30, 35, 40), font=pil_font(17, True))
    for row, index in enumerate(selected_indices):
        frame = frames[index]["frame"]
        y = 70 + row * tile_h
        draw.text((8, y + 90), f"{index}\n{frame}", fill=(45, 50, 55), font=pil_font(14, True), spacing=4)
        for column, model in enumerate(model_names):
            path = output_root / "30frames" / model / frame / f"{frame}_overlay.png"
            tile = Image.open(path).convert("RGB")
            tile.thumbnail((tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
            sheet.paste(tile, (label_w + column * tile_w + 4, y + 4))
    sheet.save(output_root / "formal_models_30frame_contact_sheet.png", optimize=True)


def create_200_contact(model_triptychs, model_names, output_root):
    class_names = ["citrus", "tree", "trunk", "all_class"]
    tile_w, tile_h, label_w, header_h = 430, 760, 135, 80
    sheet = Image.new("RGB", (label_w + tile_w * 4, header_h + tile_h * len(model_names)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), "Fixed 200-frame world semantic maps", fill=(25, 30, 35), font=pil_font(25, True))
    for col, name in enumerate(class_names):
        draw.text((label_w + col * tile_w + 8, 48), name.replace("_", " "), fill=(35, 40, 45), font=pil_font(18, True))
    for row, model in enumerate(model_names):
        y = header_h + row * tile_h
        draw.text((8, y + 330), model, fill=(35, 40, 45), font=pil_font(17, True))
        for col, class_name in enumerate(class_names):
            tile = Image.open(model_triptychs[model][class_name]).convert("RGB")
            tile.thumbnail((tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
            sheet.paste(tile, (label_w + col * tile_w + 4, y + 4))
    sheet.save(output_root / "formal_models_200frame_world_map_contact_sheet.png", optimize=True)


def legacy_stage7_diagnosis(selected_indices, frames, output_root):
    old_weight = STAGE7.parent / "stage5_training/yolo_runs/yolov8n_seg/weights/best.pt"
    old_config = yaml.safe_load((STAGE7 / "configs/yield.yaml").read_text(encoding="utf-8"))
    predictor = YoloPredictor(old_weight, "cuda:0", 0.25, expected_classes=2)
    rows = []
    for index in selected_indices:
        image = cv2.imread(frames[index]["rgb_path"], cv2.IMREAD_COLOR)
        semantic, masks, scores, overlap = predictor(image)
        rows.append({
            "frame": frames[index]["frame"], "frame_index": index,
            "old_class0_ratio": float((semantic == 1).mean()),
            "old_class1_ratio": float((semantic == 2).mean()),
            "background_ratio": float((semantic == 0).mean()),
            "raw_cross_class_overlap_ratio": overlap,
        })
    write_csv(output_root / "stage7_legacy_30frame_ratios.csv", rows)
    sequence_summary = {}
    summary_path = STAGE7 / "results/full_sequence_world/sequence_summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as stream:
            sequence_summary = next(csv.DictReader(stream))
    diagnosis = {
        "old_weight": str(old_weight),
        "checkpoint_names": predictor.names,
        "old_config_classes": old_config["classes"],
        "fruit_model_class_id": 0,
        "tree_model_class_id": 1,
        "trunk_class_present": False,
        "mask_resize": "bilinear resize then threshold > 0.5",
        "cross_class_overlap_handling": "old Stage7 kept class masks independently; overlapping pixels could enter both PCDs",
        "background_assignment": "background was not projected; only positive class masks were back-projected",
        "pcd_color_mapping": "BGR config was packed to PCD RGB correctly",
        "sequence_summary": sequence_summary,
        "diagnostic_30frame_mean": {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key.endswith("ratio")},
    }
    (output_root / "stage7_legacy_diagnosis.json").write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return diagnosis


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "figures/formal_semantic_sanity")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    classes, background, world = load_formal_config()
    frames = read_frame_index(Path(world["paths"]["frames_index"]))
    poses = read_poses(Path(world["paths"]["vins_poses"]))
    scores = score_frames(frames, poses)
    selected_rows, map_indices, window_score = select_frames(frames, scores)
    selected_indices = [row[0] for row in selected_rows]
    selection_rows = []
    section_names = ["early"] * 10 + ["middle"] * 10 + ["late"] * 10
    for section, row in zip(section_names, selected_rows):
        index, score, orange, green, brown, valid_depth, pose_valid = row
        selection_rows.append({
            "section": section, "frame_index": index, "frame": frames[index]["frame"],
            "scene_score": score, "orange_ratio": orange, "green_ratio": green,
            "brown_ratio": brown, "valid_depth_ratio": valid_depth, "pose_valid": int(pose_valid),
        })
    write_csv(args.output / "selected_30_frames.csv", selection_rows)
    write_csv(args.output / "selected_200_frame_window.csv", [{
        "start_index": map_indices[0], "end_index": map_indices[-1], "frame_count": len(map_indices),
        "start_frame": frames[map_indices[0]]["frame"], "end_frame": frames[map_indices[-1]]["frame"],
        "window_score": window_score,
    }])

    model_specs = [
        ("BiSeNetV1", "bisenet", ROOT / "training/bisenetv1/best.pth"),
        ("YOLOv8n", "yolo", ROOT / "training/yolo_runs/yolov8n_seg/weights/best.pt"),
        ("YOLO11n", "yolo", ROOT / "training/yolo_runs/yolo11n_seg/weights/best.pt"),
        ("YOLO26n", "yolo", ROOT / "training/yolo_runs/yolo26n_seg/weights/best.pt"),
    ]
    model_names = [item[0] for item in model_specs]
    stats, map_rows, model_triptychs, model_metadata = [], [], {}, {}
    scores_by_index = {row[0]: row for row in scores}
    for model_name, kind, weight in model_specs:
        if not weight.is_file():
            raise FileNotFoundError(weight)
        print(f"Loading {model_name}: {weight}", flush=True)
        predictor = BiSeNetPredictor(weight, args.device) if kind == "bisenet" else YoloPredictor(weight, args.device)
        model_metadata[model_name] = {"weight": str(weight), "kind": kind, "checkpoint_names": getattr(predictor, "names", None)}
        save_30frame_outputs(model_name, predictor, selected_indices, frames, scores_by_index, classes, world, args.output, stats)
        metrics, triptychs = save_200frame_map(model_name, predictor, map_indices, frames, poses, classes, world, args.output)
        map_rows.append(metrics)
        model_triptychs[model_name] = triptychs
        del predictor
        torch.cuda.empty_cache()
    write_csv(args.output / "formal_30frame_pixel_ratios.csv", stats)
    write_csv(args.output / "formal_200frame_world_metrics.csv", map_rows)
    create_30_contact(selected_indices, frames, model_names, args.output)
    create_200_contact(model_triptychs, model_names, args.output)
    legacy = legacy_stage7_diagnosis(selected_indices, frames, args.output)
    metadata = {
        "formal_classes": classes, "background_bgr": background, "models": model_metadata,
        "selected_30_indices": selected_indices, "selected_200_start": map_indices[0],
        "selected_200_end": map_indices[-1], "legacy_stage7": legacy,
    }
    (args.output / "sanity_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
