#!/usr/bin/env python3
"""Evaluate Orchard1 models on one frozen test set with common pixel metrics."""

from __future__ import annotations

import os
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import yaml


CLASS_NAMES = ["background", "citrus", "tree", "trunk"]
YOLO_TO_SEMANTIC = {0: 1, 1: 2, 2: 3}
DRAW_ORDER = [1, 2, 0]  # tree, trunk, citrus; foreground details overwrite canopy.


def parse_model(value: str):
    parts = value.split("=", 2)
    if len(parts) != 3 or parts[1] not in {"yolo", "bisenet"}:
        raise argparse.ArgumentTypeError("Model must be NAME=yolo|bisenet=WEIGHT_PATH")
    return tuple(parts)


def load_pairs(data_yaml: Path, semantic_root: Path, split: str, image_list: Path | None):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if image_list:
        image_paths = [Path(line.strip()) for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        split_source = Path(data[split])
        split_source = split_source if split_source.is_absolute() else root / split_source
        if split_source.is_file():
            image_paths = [Path(line.strip()) for line in split_source.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            image_paths = sorted(split_source.iterdir())
    pairs = []
    for image_path in image_paths:
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        mask_path = semantic_root / "labels" / split / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        pairs.append((image_path, mask_path))
    return pairs


def update_confusion(confusion: np.ndarray, truth: np.ndarray, prediction: np.ndarray):
    valid = (truth >= 0) & (truth < len(CLASS_NAMES))
    encoded = len(CLASS_NAMES) * truth[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), -1)


def summarize_confusion(confusion: np.ndarray):
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    summary = {
        "precision": float(precision[1:].mean()),
        "recall": float(recall[1:].mean()),
        "f1": float(f1[1:].mean()),
        "dice": float(f1[1:].mean()),
        "f1_dice": float(f1[1:].mean()),
        "foreground_miou": float(iou[1:].mean()),
        "overall_miou": float(iou.mean()),
        "miou": float(iou[1:].mean()),
        "miou_all_including_background": float(iou.mean()),
        **{f"{name}_iou": float(iou[index]) for index, name in enumerate(CLASS_NAMES)},
    }
    per_class = [
        {
            "class_id": index, "class_name": name, "precision": float(precision[index]),
            "recall": float(recall[index]), "f1": float(f1[index]),
            "dice": float(f1[index]), "iou": float(iou[index]),
            "support_pixels": int(confusion[index].sum()),
        }
        for index, name in enumerate(CLASS_NAMES)
    ]
    return summary, per_class


class YoloPredictor:
    def __init__(self, weight: Path, device: str, confidence: float):
        from ultralytics import YOLO

        self.model = YOLO(str(weight))
        self.device = device
        self.confidence = confidence

    def __call__(self, image_bgr: np.ndarray):
        height, width = image_bgr.shape[:2]
        result = self.model.predict(
            image_bgr,
            imgsz=640,
            conf=self.confidence,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )[0]
        class_masks = {class_id: np.zeros((height, width), dtype=np.uint8) for class_id in YOLO_TO_SEMANTIC}
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for mask, class_id in zip(masks, classes):
                if class_id not in class_masks:
                    continue
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
                class_masks[class_id][mask > 0.5] = 1
        semantic = np.zeros((height, width), dtype=np.uint8)
        for class_id in DRAW_ORDER:
            semantic[class_masks[class_id] > 0] = YOLO_TO_SEMANTIC[class_id]
        return semantic

    def instance_metrics(self, data_yaml: Path, output_root: Path, name: str, split: str):
        result = self.model.val(
            data=str(data_yaml),
            split=split,
            imgsz=640,
            batch=1,
            device=self.device,
            plots=False,
            project=str(output_root),
            name=name,
            exist_ok=True,
            verbose=False,
        )
        return {
            "mask_map50": float(result.seg.map50),
            "mask_map50_95": float(result.seg.map),
            "box_map50": float(result.box.map50),
            "box_map50_95": float(result.box.map),
        }


class BiSeNetPredictor:
    def __init__(self, weight: Path, device: str, repository: Path):
        sys.path.insert(0, str(repository))
        from lib.models import model_factory

        self.device = torch.device(device)
        self.model = model_factory["bisenetv1"](4)
        state = torch.load(weight, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.aux_mode = "eval"
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.4, 0.4, 0.4], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2, 0.2, 0.2], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        logits = self.model(tensor)[0]
        if logits.shape[-2:] != image_bgr.shape[:2]:
            logits = functional.interpolate(logits, image_bgr.shape[:2], mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].byte().cpu().numpy()


def write_rows(path: Path, rows):
    fields = [
        "model",
        "weight",
        "evaluation_split",
        "image_count",
        "precision",
        "recall",
        "f1",
        "dice",
        "f1_dice",
        "foreground_miou",
        "overall_miou",
        "miou",
        "miou_all_including_background",
        "background_iou",
        "citrus_iou",
        "tree_iou",
        "trunk_iou",
        "mask_map50",
        "mask_map50_95",
        "box_map50",
        "box_map50_95",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--bisenet-repository", type=Path, default=Path(os.environ.get("BISENET_REPOSITORY", "third_party/BiSeNet-model-conversion")))
    parser.add_argument("--instance-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class-output", type=Path)
    args = parser.parse_args()

    pairs = load_pairs(args.data_yaml, args.semantic_root, args.split, args.image_list)
    rows = []
    per_class_rows = []
    for name, kind, weight_text in args.model:
        weight = Path(weight_text)
        base = {
            "model": name,
            "weight": str(weight),
            "evaluation_split": args.split,
            "image_count": len(pairs),
            "mask_map50": "NA",
            "mask_map50_95": "NA",
            "box_map50": "NA",
            "box_map50_95": "NA",
            "status": "failed",
            "notes": "",
        }
        try:
            if not weight.is_file():
                raise FileNotFoundError(weight)
            predictor = YoloPredictor(weight, args.device, args.confidence) if kind == "yolo" else BiSeNetPredictor(weight, args.device, args.bisenet_repository)
            confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
            for index, (image_path, mask_path) in enumerate(pairs, 1):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                prediction = predictor(image)
                update_confusion(confusion, truth, prediction)
                if index % 20 == 0:
                    print(f"{name}: {index}/{len(pairs)}", flush=True)
            summary, class_rows = summarize_confusion(confusion)
            base.update(summary)
            for class_row in class_rows:
                per_class_rows.append({
                    "model": name,
                    "evaluation_split": args.split,
                    "image_count": len(pairs),
                    **class_row,
                })
            if kind == "yolo":
                base.update(predictor.instance_metrics(args.data_yaml, args.instance_output, name, args.split))
            base["status"] = "success"
        except Exception as exc:
            base["notes"] = f"{type(exc).__name__}: {exc}"
            print(f"{name} failed: {base['notes']}", file=sys.stderr, flush=True)
        rows.append(base)
        write_rows(args.output, rows)
        if args.per_class_output:
            args.per_class_output.parent.mkdir(parents=True, exist_ok=True)
            fields = ["model", "evaluation_split", "image_count", "class_id", "class_name", "precision", "recall", "f1", "dice", "iou", "support_pixels"]
            with args.per_class_output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(per_class_rows)


if __name__ == "__main__":
    main()
