#!/usr/bin/env python3
"""Apply the targeted V3.1 publication-figure readability fixes.

Only Figs. 4, 7, and 8 are re-rendered. The script performs visualization and
scientific-source checks; it does not run inference, SLAM, mapping, or fitting.
"""

from __future__ import annotations

import os
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


PAPER = Path(os.environ.get("CITRUS_PAPER_ROOT", Path(__file__).resolve().parents[2]))
V3 = PAPER / "manuscript_submission_candidate_p3_v3"
OUT = PAPER / "manuscript_submission_candidate_p3_v3_1"
FIGURES = OUT / "figures"
AUDIT = OUT / "audit"
V2 = PAPER / "manuscript_final_revision_v2"
FINAL = PAPER / "final_paper_results"
CORRECTED = PAPER / "pose_corrected_rebuild_v1"
RESTORATION = Path(os.environ.get("CITRUS_RESTORATION_ROOT", Path(__file__).resolve().parents[2] / "external/original_pipeline_restoration_v1"))
PHASE30 = RESTORATION / "30_p2_p3_reassessment"

RAW_PCD = RESTORATION / "15_async_segmentation/full_tests/yolo8_async_full_v1/RAW.pcd"
SOR_PCD = RESTORATION / "15_async_segmentation/full_tests/yolo8_async_full_v1/SOR.pcd"
P3_CSV = PHASE30 / "data/deduplicated_candidates.csv"
P3_MEMBERSHIP = PHASE30 / "data/sphere_support_membership.npz"

CLASS_NAMES = ("citrus", "tree", "trunk")
CLASS_COLORS = {"citrus": "#f28e2b", "tree": "#78a982", "trunk": "#785548"}
MODEL_KEYS = ("bisenet", "yolov8n", "yolo11n", "yolo26n")
MODEL_NAMES = ("BiSeNetV1", "YOLOv8", "YOLO11", "YOLO26")
MODEL_SOURCE_NAMES = ("BiSeNetV1", "YOLOv8n-seg", "YOLO11n-seg", "YOLO26n-seg")
BACKENDS = (("vins", "VINS-RGBD"), ("orbslam3", "ORB-SLAM3"), ("rtabmap", "RTAB-Map"))
BACKEND_COLORS = {"VINS-RGBD": "#3b6ea8", "ORB-SLAM3": "#b44d6a", "RTAB-Map": "#238b78"}

MASK_RGB_TO_ID = {
    (0, 0, 0): 0,
    (255, 140, 0): 1,
    (70, 165, 40): 2,
    (170, 75, 95): 3,
}
PALETTE = {
    0: np.asarray([0, 0, 0], dtype=np.uint8),
    1: np.asarray([242, 142, 43], dtype=np.uint8),
    2: np.asarray([120, 169, 130], dtype=np.uint8),
    3: np.asarray([120, 85, 72], dtype=np.uint8),
}
PACKED = {
    "citrus": np.uint32(0xFF008CFF),
    "tree": np.uint32(0xFF28A546),
    "trunk": np.uint32(0xFF5F4BAA),
    "background": np.uint32(0xFF000000),
}
ROI_1 = (24.70, 27.10, -13.25, -10.85, 0.25, 1.15)
ROI_2 = (21.00, 23.40, -12.55, -10.15, 0.25, 1.15)
GLOBAL_ONLINE = (18.0, 28.75, -13.75, -9.0, -0.23, 1.94)
P3_RADIUS_RANGE_M = (0.020, 0.045)
P3_BOUNDARY_MARGIN_M = 0.001


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_figure(figure: plt.Figure, stem: str, dpi: int = 450, svg: bool = True) -> None:
    path = FIGURES / stem
    figure.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white", bbox_inches="tight")
    if svg:
        figure.savefig(path.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def read_pcd_xyz(path: Path) -> np.ndarray:
    header_lines = 0
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if line.strip().lower() == "data ascii":
                header_lines = index + 1
                break
    if not header_lines:
        raise ValueError(f"ASCII PCD header not found: {path}")
    points = np.loadtxt(path, skiprows=header_lines, usecols=(0, 1, 2), dtype=np.float64)
    points = np.atleast_2d(points)
    return points[np.isfinite(points).all(axis=1)]


def read_semantic_pcd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header_lines = 0
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if line.strip().lower() == "data ascii":
                header_lines = index + 1
                break
    data = np.loadtxt(path, skiprows=header_lines, usecols=(0, 1, 2, 3), dtype=np.float64)
    xyz = np.asarray(data[:, :3], dtype=np.float64)
    packed = np.asarray(data[:, 3], dtype=np.uint32)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Non-finite semantic PCD: {path}")
    return xyz, packed


def deterministic_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=int)]


def style_2d(axis: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#e1e5e8", linewidth=0.35, zorder=0)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)


def semantic_legend(figure: plt.Figure, y: float = 0.02) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5, color=CLASS_COLORS[name], label=name)
        for name in CLASS_NAMES
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, frameon=False)


def generate_fig1() -> None:
    source = V2 / "assets/FIG1_SYSTEM_FRAMEWORK_V2.png"
    image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(source)
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", 19)
    targets = [
        ((658, 229, 801, 269), "YOLOv8", (216, 241, 213)),
        ((658, 278, 801, 318), "YOLO11", (249, 235, 215)),
        ((658, 327, 801, 367), "YOLO26", (222, 212, 244)),
    ]
    for (x0, y0, x1, y1), label, fill in targets:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=fill)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(((x0 + x1 - (box[2] - box[0])) / 2, (y0 + y1 - (box[3] - box[1])) / 2 - 2), label, font=font, fill="#111111")

    draw.rectangle((1001, 563, 1193, 620), fill=(252, 252, 249))
    bold = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf", 15)
    draw.text((1022, 574), "6. Accumulated 3D", font=bold, fill="#111111")
    draw.text((1043, 595), "semantic map", font=bold, fill="#111111")
    pil.save(FIGURES / "FIG1_FINAL.png", dpi=(450, 450))


def generate_fig2() -> None:
    source_doc = V2 / "MANUSCRIPT_REVISED_EN_V2.docx"
    with zipfile.ZipFile(source_doc) as archive:
        source = Image.open(io.BytesIO(archive.read("word/media/image2.png"))).convert("RGB")
    # The embedded source also contains legacy page material below the schematic.
    panel = source.crop((0, 0, source.width, 456))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf", 12)
    draw.text((1225, 347), "(Eq. 11)", font=font, fill="#15264b")
    panel.save(FIGURES / "FIG2_FINAL.png", dpi=(450, 450))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_label(path: Path) -> np.ndarray:
    label = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if not set(np.unique(label)).issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected GT labels: {path}")
    return label


def decode_prediction(path: Path) -> np.ndarray:
    rgb = load_rgb(path)
    label = np.zeros(rgb.shape[:2], dtype=np.uint8)
    recognized = np.zeros(rgb.shape[:2], dtype=bool)
    for color, class_id in MASK_RGB_TO_ID.items():
        selected = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)
        label[selected] = class_id
        recognized |= selected
    if not recognized.all():
        raise ValueError(f"Unknown frozen-mask colors: {path}")
    return label


def overlay(image: np.ndarray, label: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    result = image.astype(np.float32).copy()
    colors = np.zeros_like(image)
    for class_id in (1, 2, 3):
        colors[label == class_id] = PALETTE[class_id]
    selected = label > 0
    result[selected] = (1.0 - alpha) * result[selected] + alpha * colors[selected]
    return np.clip(result, 0, 255).astype(np.uint8)


def largest_component_crop(label: np.ndarray, target: int | None, width: int = 220, height: int = 165) -> tuple[int, int, int, int]:
    mask = (label > 0) if target is None else (label == target)
    count, components, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        center_x, center_y = label.shape[1] // 2, label.shape[0] // 2
    else:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        center_x, center_y = map(int, np.round(centroids[component]))
    x0 = int(np.clip(center_x - width // 2, 0, label.shape[1] - width))
    y0 = int(np.clip(center_y - height // 2, 0, label.shape[0] - height))
    return x0, y0, x0 + width, y0 + height


def add_error_contours(array: np.ndarray, gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = ((gt != prediction) & ((gt > 0) | (prediction > 0))).astype(np.uint8)
    contours, _ = cv2.findContours(error, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    cv2.drawContours(output, contours, -1, (48, 48, 220), 1, lineType=cv2.LINE_AA)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def panel_with_inset(array: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    result = array.copy()
    x0, y0, x1, y1 = crop
    cv2.rectangle(result, (x0, y0), (x1, y1), (245, 245, 245), 2)
    inset = cv2.resize(array[y0:y1, x0:x1], (230, 173), interpolation=cv2.INTER_NEAREST)
    result[480 - 181 : 480 - 8, 640 - 238 : 640 - 8] = inset
    cv2.rectangle(result, (640 - 239, 480 - 182), (640 - 7, 480 - 7), (20, 20, 20), 2)
    return result


def generate_fig3() -> None:
    rgb_root = Path(os.environ.get("CITRUS_TEST_IMAGE_ROOT", Path(__file__).resolve().parents[2] / "data/images/test"))
    gt_root = PAPER / "training/bisenet_dataset/labels/test"
    pred_root = PAPER / "figures/segmentation_comparison/semantic_masks"
    scenes = (
        ("clear_citrus", "791", "Citrus boundary", 1),
        ("trunk_visible", "858", "Trunk structure", 3),
        ("occlusion_complexity", "514", "Occlusion", None),
        ("background_interference", "890", "Background interference", 1),
    )
    pred_keys = ("bisenetv1", "yolov8n", "yolo11n", "yolo26n")
    columns = ("RGB", "Ground truth") + MODEL_NAMES
    figure, axes = plt.subplots(len(scenes), len(columns), figsize=(12.4, 7.6))
    for row, (_, stem, label, target_class) in enumerate(scenes):
        rgb = load_rgb(rgb_root / f"{stem}.png")
        gt = load_label(gt_root / f"{stem}.png")
        crop = largest_component_crop(gt, target_class)
        panels = [panel_with_inset(rgb, crop), panel_with_inset(overlay(rgb, gt), crop)]
        for key in pred_keys:
            prediction = decode_prediction(pred_root / key / f"{stem}.png")
            rendered = add_error_contours(overlay(rgb, prediction), gt, prediction)
            panels.append(panel_with_inset(rendered, crop))
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].set_axis_off()
            if row == 0:
                axes[row, column].set_title(columns[column], fontsize=9, pad=4)
            if column == 0:
                axes[row, column].text(
                    -0.04, 0.5, label, rotation=90, va="center", ha="right",
                    transform=axes[row, column].transAxes, fontsize=8.2,
                )
    figure.text(0.995, 0.012, "Red contours: semantic mismatch against manual GT", ha="right", fontsize=7.3, color="#8f2736")
    figure.subplots_adjust(left=0.075, right=0.995, top=0.96, bottom=0.045, wspace=0.025, hspace=0.07)
    save_figure(figure, "FIG3_FINAL")


def load_corrected_voxel_clouds() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for model in MODEL_KEYS:
        source = CORRECTED / f"mapping/{model}/global_vote_history.npz"
        with np.load(source) as data:
            voxels = np.asarray(data["voxel_xyz"], dtype=np.int64)
            labels = np.asarray(data["final_class_id"], dtype=np.int64)
        result[model] = ((voxels.astype(np.float64) + 0.5) * 0.05, labels)
    return result


def crop_bounds(points: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)


def draw_voxel_map(
    axis: plt.Axes,
    points: np.ndarray,
    labels: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    emphasis: int | None = None,
    size_scale: float = 1.0,
    alpha_scale: float = 1.0,
) -> None:
    for class_id, class_name in ((1, "tree"), (2, "trunk"), (0, "citrus")):
        selected = labels == class_id
        alpha = 0.75 if emphasis is None else (1.0 if class_id == emphasis else 0.12)
        size = 0.22 if emphasis is None else (0.95 if class_id == emphasis else 0.18)
        axis.scatter(
            points[selected, 0], points[selected, 1],
            s=size * size_scale,
            c=CLASS_COLORS[class_name],
            alpha=min(1.0, alpha * alpha_scale),
            linewidths=0,
            rasterized=True,
        )
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    axis.tick_params(labelsize=6, length=2)


def corrected_rois() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    document = yaml.safe_load((CORRECTED / "configs/CORRECTED_VISUAL_ROIS.yaml").read_text(encoding="utf-8"))
    output = {}
    for row in document["rois"]:
        b = row["bounds"]
        output[row["id"]] = (
            np.asarray([b["x_min"], b["y_min"], b["z_min"]]),
            np.asarray([b["x_max"], b["y_max"], b["z_max"]]),
        )
    return output


def generate_fig4() -> None:
    clouds = load_corrected_voxel_clouds()
    pooled = np.concatenate([value[0] for value in clouds.values()], axis=0)
    low = np.percentile(pooled[:, :2], 0.25, axis=0)
    high = np.percentile(pooled[:, :2], 99.75, axis=0)
    margin = (high - low) * 0.015
    global_limits = ((low[0] - margin[0], high[0] + margin[0]), (low[1] - margin[1], high[1] + margin[1]))
    rois = corrected_rois()
    figure = plt.figure(figsize=(11.8, 7.8))
    grid = figure.add_gridspec(4, 4, height_ratios=(1.0, 1.0, 1.0, 0.62), hspace=0.26, wspace=0.10)
    for column, (model, name) in enumerate(zip(MODEL_KEYS, MODEL_NAMES)):
        points, labels = clouds[model]
        axis = figure.add_subplot(grid[0, column])
        draw_voxel_map(axis, points, labels, global_limits)
        axis.set_title(name, fontsize=9.3, pad=3)
        if column == 0:
            axis.set_ylabel("Global map\ny (m)", fontsize=8)
        axis.set_xticklabels([])

        for row, (roi_id, emphasis, row_label) in enumerate((("ROI-A", 0, "Citrus-rich ROI"), ("ROI-B", 2, "Trunk-visible ROI")), start=1):
            roi = rois[roi_id]
            selected = crop_bounds(points, roi)
            local = points[selected]
            local_labels = labels[selected]
            axis = figure.add_subplot(grid[row, column])
            limits = ((roi[0][0], roi[1][0]), (roi[0][1], roi[1][1]))
            draw_voxel_map(
                axis, local, local_labels, limits,
                emphasis=emphasis, size_scale=1.8, alpha_scale=1.15,
            )
            if column == 0:
                axis.set_ylabel(f"{row_label}\ny (m)", fontsize=8)

    table = read_csv(CORRECTED / "tables/Table4_perception_controlled_mapping.csv")
    counts = np.asarray([[float(row[f"{name}_voxel_count"]) for name in CLASS_NAMES] for row in table])
    relative = 100.0 * (counts - counts.mean(axis=0, keepdims=True)) / counts.mean(axis=0, keepdims=True)
    axis = figure.add_subplot(grid[3, :])
    limit = max(5.0, float(np.max(np.abs(relative))))
    image = axis.imshow(relative.T, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    axis.set_xticks(np.arange(4), MODEL_NAMES)
    axis.set_yticks(np.arange(3), ["citrus", "tree", "trunk"])
    axis.set_title("Final class allocation relative to the four-model mean (%)", fontsize=8.5, pad=4)
    for y in range(3):
        for x in range(4):
            axis.text(x, y, f"{relative[x, y]:+.1f}%", ha="center", va="center", fontsize=7, color="#111111")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.018, pad=0.012)
    colorbar.ax.tick_params(labelsize=6)
    semantic_legend(figure, y=0.003)
    figure.subplots_adjust(left=0.07, right=0.96, top=0.965, bottom=0.075)
    save_figure(figure, "FIG4_FINAL")


def selected_pose_data(key: str) -> tuple[list[int], np.ndarray]:
    frame_set = PAPER / "slam_comparison/frame_sets/frequency_controlled_common_frames.txt"
    selected = [int(value) for value in frame_set.read_text(encoding="utf-8").split()]
    source_key = {"vins": "vins", "orbslam3": "orb", "rtabmap": "rtab"}[key]
    rows = read_csv(PAPER / f"slam_comparison/normalized_frame_poses/{source_key}.csv")
    xyz = np.asarray([[float(rows[index][name]) for name in ("tx", "ty", "tz")] for index in selected])
    if len(xyz) != 1804:
        raise AssertionError(f"Expected 1804 common poses for {key}, got {len(xyz)}")
    return selected, xyz


def generate_fig5() -> None:
    trajectories = {key: selected_pose_data(key) for key, _ in BACKENDS}
    detailed = {row["method"]: row for row in read_csv(FINAL / "tables/TableS3_localization_continuity_detailed.csv")}
    pooled = np.concatenate([xyz for _, xyz in trajectories.values()], axis=0)
    low, high = pooled[:, :2].min(axis=0), pooled[:, :2].max(axis=0)
    margin = np.maximum((high - low) * 0.04, 0.5)

    figure = plt.figure(figsize=(11.7, 6.2))
    grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 1.12), hspace=0.38, wspace=0.28)
    for column, (key, label) in enumerate(BACKENDS):
        frames, xyz = trajectories[key]
        axis = figure.add_subplot(grid[0, column])
        axis.plot(xyz[:, 0], xyz[:, 1], color=BACKEND_COLORS[label], linewidth=0.8)
        row = detailed[label]
        before = int(row["translation_max_from_frame"])
        after = int(row["translation_max_to_frame"])
        index = frames.index(before)
        if key == "orbslam3":
            axis.plot(xyz[index:index + 2, 0], xyz[index:index + 2, 1], color="#d1495b", linewidth=2.2, zorder=3)
            axis.scatter(xyz[index:index + 2, 0], xyz[index:index + 2, 1], c=["#ed9b40", "#d1495b"], s=28, zorder=4)
        axis.set_xlim(low[0] - margin[0], high[0] + margin[0])
        axis.set_ylim(low[1] - margin[1], high[1] + margin[1])
        axis.set_aspect("equal", adjustable="box")
        style_2d(axis, "x (m)", "y (m)" if column == 0 else None)
        axis.set_title(f"{label}\nmaximum step = {float(row['translation_max_m']):.3f} m", fontsize=8.5)

    axis = figure.add_subplot(grid[0, 3])
    x = np.arange(3)
    labels = [label for _, label in BACKENDS]
    p95 = [float(detailed[label]["translation_p95_m"]) for label in labels]
    maximum = [float(detailed[label]["translation_max_m"]) for label in labels]
    axis.scatter(x - 0.08, p95, color="#4c78a8", label="P95", s=28)
    axis.scatter(x + 0.08, maximum, marker="D", color="#b44d5b", label="Maximum", s=28)
    for i in range(3):
        axis.plot([x[i] - 0.08, x[i] + 0.08], [p95[i], maximum[i]], color="#aab1b6", linewidth=0.7)
    axis.set_yscale("log")
    axis.set_xticks(x, ["VINS", "ORB", "RTAB"])
    axis.set_ylabel("Translation step (m, log scale)")
    axis.set_title("Tail versus maximum")
    style_2d(axis)
    axis.legend(frameon=False, fontsize=7)

    axis = figure.add_subplot(grid[1, :])
    orb_max_index = None
    for key, label in BACKENDS:
        frames, xyz = trajectories[key]
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        axis.plot(np.arange(1, len(xyz)), steps, color=BACKEND_COLORS[label], linewidth=0.65, alpha=0.9, label=label)
        if key == "orbslam3":
            orb_max_index = int(np.argmax(steps)) + 1
            axis.scatter([orb_max_index], [steps[orb_max_index - 1]], s=40, color="#d1495b", zorder=5)
            axis.annotate(
                "45.9721 m discontinuity",
                xy=(orb_max_index, steps[orb_max_index - 1]),
                xytext=(orb_max_index + 90, 3.0),
                fontsize=8,
                color="#992f3f",
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#992f3f"},
            )
    axis.set_yscale("log")
    axis.set_xlabel("Common-observation transition index")
    axis.set_ylabel("Adjacent translation step (m, log scale)")
    axis.set_title("All 1,803 adjacent steps on the 1,804 common observation times", fontsize=9)
    style_2d(axis)
    axis.legend(frameon=False, ncol=3, loc="upper right")
    figure.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.10)
    save_figure(figure, "FIG5_FINAL")


def load_localization_clouds() -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for key, _ in BACKENDS:
        directory = PAPER / f"slam_comparison/frequency_controlled/{key}"
        output[key] = {name: read_pcd_xyz(directory / f"global_world_{name}_cloud.pcd") for name in CLASS_NAMES}
    return output


def draw_cloud(axis: plt.Axes, clouds: dict[str, np.ndarray], dims: tuple[int, int], limits, local_origin: np.ndarray | None = None) -> None:
    for name in ("tree", "trunk", "citrus"):
        points = clouds[name]
        if local_origin is not None:
            points = points - local_origin
        points = deterministic_points(points, 180_000 if name == "tree" else 90_000)
        axis.scatter(points[:, dims[0]], points[:, dims[1]], s={"tree": 0.2, "trunk": 0.45, "citrus": 0.8}[name], c=CLASS_COLORS[name], alpha={"tree": 0.40, "trunk": 0.78, "citrus": 0.92}[name], linewidths=0, rasterized=True)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    axis.tick_params(labelsize=6, length=2)


def generate_fig6() -> dict[str, float]:
    clouds = load_localization_clouds()
    trajectories = {key: selected_pose_data(key) for key, _ in BACKENDS}
    all_points = np.concatenate([points for backend in clouds.values() for points in backend.values()], axis=0)
    map_low, map_high = all_points.min(axis=0), all_points.max(axis=0)
    map_margin = np.maximum((map_high - map_low) * 0.02, 0.1)
    map_low -= map_margin
    map_high += map_margin
    all_trajectory = np.concatenate([xyz for _, xyz in trajectories.values()], axis=0)
    traj_low, traj_high = all_trajectory[:, :2].min(axis=0), all_trajectory[:, :2].max(axis=0)
    traj_margin = np.maximum((traj_high - traj_low) * 0.03, 0.3)
    anchor_doc = yaml.safe_load((PAPER / "slam_comparison/SLAM_VISUAL_ANCHORS.yaml").read_text(encoding="utf-8"))
    anchor = anchor_doc["anchors"][0]

    figure, axes = plt.subplots(4, 3, figsize=(10.7, 9.0))
    z_ranges = {}
    for column, (key, label) in enumerate(BACKENDS):
        _, trajectory = trajectories[key]
        axis = axes[0, column]
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=BACKEND_COLORS[label], linewidth=0.8)
        axis.set_xlim(traj_low[0] - traj_margin[0], traj_high[0] + traj_margin[0])
        axis.set_ylim(traj_low[1] - traj_margin[1], traj_high[1] + traj_margin[1])
        axis.set_aspect("equal", adjustable="box")
        style_2d(axis)
        axis.set_title(label, fontsize=9.2)

        draw_cloud(axes[1, column], clouds[key], (0, 1), ((map_low[0], map_high[0]), (map_low[1], map_high[1])))
        draw_cloud(axes[2, column], clouds[key], (0, 2), ((map_low[0], map_high[0]), (map_low[2], map_high[2])))
        z_ranges[label] = float(trajectory[:, 2].max() - trajectory[:, 2].min())

        center_key = {"vins": "vins", "orbslam3": "orb", "rtabmap": "rtab"}[key]
        center = np.asarray(anchor["centers"][center_key], dtype=float)
        half = 1.5
        local = {}
        for name, points in clouds[key].items():
            selected = np.all((points >= center - half) & (points <= center + half), axis=1)
            local[name] = points[selected]
        draw_cloud(axes[3, column], local, (0, 1), ((-half, half), (-half, half)), local_origin=center)

    row_labels = ("Common trajectory", "All-class BEV", "Side view (x-z)", "Camera-centered ROI")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(f"{label}\n" + ("y (m)" if row != 2 else "z (m)"), fontsize=8)
    for column in range(3):
        axes[3, column].set_xlabel("local x (m)")
        axes[1, column].set_xlabel("x (m)")
        axes[2, column].set_xlabel("x (m)")
    semantic_legend(figure, y=0.003)
    figure.subplots_adjust(left=0.095, right=0.985, top=0.965, bottom=0.075, wspace=0.12, hspace=0.24)
    save_figure(figure, "FIG6_FINAL")
    return z_ranges


def load_class_cloud_set(directory: Path) -> dict[str, np.ndarray]:
    return {name: read_pcd_xyz(directory / f"global_world_{name}_cloud.pcd") for name in CLASS_NAMES}


def generate_fig7() -> None:
    failure = read_csv(FINAL / "tables/orb_failure_case_metrics.csv")
    before_row, after_row = failure
    orb_rows = [row for row in read_csv(PAPER / "slam_comparison/normalized_frame_poses/orb.csv") if int(row["pose_valid"]) == 1]
    frames = [int(row["frame_index"]) for row in orb_rows]
    xyz = np.asarray([[float(row[name]) for name in ("tx", "ty", "tz")] for row in orb_rows])
    first, second = int(before_row["native_jump_from_frame"]), int(before_row["native_jump_to_frame"])
    by_frame = {frame: index for index, frame in enumerate(frames)}
    a, b = xyz[by_frame[first]], xyz[by_frame[second]]

    before = load_class_cloud_set(PAPER / "slam_comparison/orb_jump_before")
    after = load_class_cloud_set(PAPER / "slam_comparison/orb_jump_after")
    windows = [before, after]
    centers, spans = [], []
    for cloud in windows:
        pooled = np.concatenate(list(cloud.values()), axis=0)
        centers.append((pooled.min(axis=0) + pooled.max(axis=0)) / 2)
        spans.append(pooled.max(axis=0) - pooled.min(axis=0))
    common_span = np.maximum(np.max(np.vstack(spans), axis=0) * 1.05, 0.1)

    corrected = load_corrected_voxel_clouds()
    roia = corrected_rois()["ROI-A"]
    figure = plt.figure(figsize=(11.4, 6.9))
    grid = figure.add_gridspec(2, 3, hspace=0.32, wspace=0.23)

    axis = figure.add_subplot(grid[0, 0])
    axis.plot(xyz[:, 0], xyz[:, 1], color=BACKEND_COLORS["ORB-SLAM3"], linewidth=0.7)
    axis.plot([a[0], b[0]], [a[1], b[1]], color="#d1495b", linewidth=2.2)
    axis.scatter([a[0], b[0]], [a[1], b[1]], c=["#ed9b40", "#d1495b"], s=28)
    axis.annotate("45.9721 m", xy=((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), xytext=(6, 6), textcoords="offset points", color="#992f3f", fontsize=8)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title("(a) ORB-SLAM3 trajectory discontinuity")
    style_2d(axis, "x (m)", "y (m)")

    for column, (cloud, center, title, count) in enumerate(
        ((before, centers[0], "Before window", before_row["frozen_frame_count"]), (after, centers[1], "After window", after_row["frozen_frame_count"])), start=1
    ):
        axis = figure.add_subplot(grid[0, column])
        local_limits = ((-common_span[0] / 2, common_span[0] / 2), (-common_span[1] / 2, common_span[1] / 2))
        draw_cloud(axis, cloud, (0, 1), local_limits, local_origin=center)
        axis.set_title(f"{title} ({count} valid observations)")
        axis.set_xlabel("local x (m)")
        axis.set_ylabel("local y (m)")

    axis = figure.add_subplot(grid[1, 0])
    metric_names = ("semantic_conflict_rate", "normalized_semantic_entropy", "isolated_semantic_voxel_ratio")
    labels = ("SCR", "Entropy", "Isolated ratio")
    before_values = [float(before_row[name]) for name in metric_names]
    after_values = [float(after_row[name]) for name in metric_names]
    x = np.arange(3)
    axis.bar(x - 0.18, before_values, width=0.36, color="#6b9ac4", label="Before")
    axis.bar(x + 0.18, after_values, width=0.36, color="#d66b5d", label="After")
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_title("Window statistics (descriptive)")
    axis.set_ylabel("Value (log scale)")
    axis.legend(frameon=False, fontsize=7)
    style_2d(axis)

    for column, (model, name) in enumerate((("yolov8n", "YOLOv8"), ("yolo11n", "YOLO11")), start=1):
        points, labels_array = corrected[model]
        selected = crop_bounds(points, roia)
        axis = figure.add_subplot(grid[1, column])
        draw_voxel_map(
            axis,
            points[selected],
            labels_array[selected],
            ((roia[0][0], roia[1][0]), (roia[0][1], roia[1][1])),
            size_scale=3.0,
            alpha_scale=1.25,
        )
        axis.set_title(f"{name}: fixed ROI-A")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    semantic_legend(figure, y=0.002)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.085)
    save_figure(figure, "FIG7_FINAL")


def crop_mask(points: np.ndarray, bounds: tuple[float, ...]) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bounds
    return (
        (points[:, 0] >= x0) & (points[:, 0] <= x1)
        & (points[:, 1] >= y0) & (points[:, 1] <= y1)
        & (points[:, 2] >= z0) & (points[:, 2] <= z1)
    )


def project_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth, elevation = np.radians(82.0), np.radians(5.0)
    line = np.asarray([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
    up = np.asarray([0.0, 0.0, 1.0])
    horizontal = np.cross(up, line)
    horizontal /= np.linalg.norm(horizontal)
    vertical = up - np.dot(up, line) * line
    vertical /= np.linalg.norm(vertical)
    return horizontal, vertical, line


PROJ_U, PROJ_V, PROJ_D = project_basis()


def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return points @ PROJ_U, points @ PROJ_V, points @ PROJ_D


def bounds_corners(bounds: tuple[float, ...]) -> np.ndarray:
    return np.asarray([[x, y, z] for x in bounds[:2] for y in bounds[2:4] for z in bounds[4:]], dtype=float)


def projected_style(axis: plt.Axes, bounds: tuple[float, ...]) -> None:
    u, v, _ = project(bounds_corners(bounds))
    axis.set_xlim(u.min(), u.max())
    axis.set_ylim(v.min(), v.max())
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def projected_scatter(axis: plt.Axes, points: np.ndarray, color: str, size: float, alpha: float, maximum: int | None = None) -> None:
    if maximum:
        points = deterministic_points(points, maximum)
    if not len(points):
        return
    u, v, depth = project(points)
    order = np.argsort(depth)
    axis.scatter(u[order], v[order], s=size, c=color, alpha=alpha, linewidths=0, rasterized=True)


def draw_online_semantics(axis: plt.Axes, xyz: np.ndarray, packed: np.ndarray, bounds: tuple[float, ...], maximum: int | None = None) -> None:
    selected = crop_mask(xyz, bounds)
    points, labels = xyz[selected], packed[selected]
    for name, alpha, size in (("tree", 0.24, 0.18), ("trunk", 0.58, 0.28), ("citrus", 0.96, 0.40)):
        projected_scatter(axis, points[labels == PACKED[name]], CLASS_COLORS[name], size, alpha, maximum)
    projected_style(axis, bounds)


def p3_candidates() -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = read_csv(P3_CSV)
    membership = np.load(P3_MEMBERSHIP, allow_pickle=False)
    member_ids = membership["candidate_id"].astype(str)
    starts = membership["point_start"].astype(np.int64)
    counts = membership["point_count"].astype(np.int64)
    source_indices = membership["source_point_index"].astype(np.int64)
    slices = {candidate_id: source_indices[start : start + count] for candidate_id, start, count in zip(member_ids, starts, counts)}
    output = []
    for row in rows:
        if row["profile"] != "P3" or row["scope"] not in ("ROI_1_FROZEN", "ROI_2_INDEPENDENT"):
            continue
        if row["post_dedup_status"] != "FINAL_REVALIDATED":
            continue
        candidate_id = row["candidate_id"]
        if candidate_id not in slices:
            raise KeyError(f"Missing support membership: {candidate_id}")
        output.append(
            {
                "candidate_id": candidate_id,
                "scope": row["scope"],
                "center": np.asarray([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]),
                "radius": float(row["radius_m"]),
                "support": int(row["final_support"]),
                "independent": int(row["final_independent_voxels_10mm"]),
                "condition": float(row["final_jacobian_condition_number"]),
                "span": float(row["final_surface_angular_span_deg"]),
                "boundary": row["boundary_status"],
                "duplicate": row["duplicate_status"],
                "evidence": row["evidence_label"] or "UNVERIFIED",
                "support_indices": slices[candidate_id],
            }
        )
    if len(output) != 84:
        raise AssertionError(f"Expected 84 P3 FINAL_REVALIDATED ROI hypotheses, got {len(output)}")
    return output, slices


def sphere_wire_2d(axis: plt.Axes, candidate: dict) -> None:
    angle = np.linspace(0, 2 * np.pi, 120)
    center, radius = candidate["center"], candidate["radius"]
    for first, second in ((PROJ_U, PROJ_V), (PROJ_U, PROJ_D), (PROJ_V, PROJ_D)):
        points = center + radius * (np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second)
        u, v, _ = project(points)
        axis.plot(u, v, color="#b3242c", linewidth=0.72, alpha=0.82, zorder=10)


def select_p3_representatives(candidates: list[dict]) -> list[dict]:
    def boundary_clearance(candidate: dict) -> float:
        return min(
            candidate["radius"] - P3_RADIUS_RANGE_M[0],
            P3_RADIUS_RANGE_M[1] - candidate["radius"],
        )

    eligible = [
        candidate for candidate in candidates
        if candidate["boundary"] != "BOUNDARY_LIMITED"
        and boundary_clearance(candidate) > P3_BOUNDARY_MARGIN_M
        and candidate["duplicate"] != "POSSIBLE_DUPLICATE"
    ]
    eligible.sort(
        key=lambda c: (
            c["evidence"] != "FRUIT_LIKE_VISUAL_EVIDENCE",
            -c["independent"],
            c["condition"],
            -c["span"],
            -c["support"],
            c["candidate_id"],
        )
    )
    selected = []
    for candidate in eligible:
        if selected and min(np.linalg.norm(candidate["center"] - existing["center"]) for existing in selected) < 0.20:
            continue
        selected.append(candidate)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError("Could not select three spatially separated P3 representatives")
    return selected


def draw_local_sphere(axis, candidate: dict, support_points: np.ndarray) -> None:
    centered = support_points - candidate["center"]
    radius = candidate["radius"]
    axis.scatter(centered[:, 0], centered[:, 1], centered[:, 2], s=15, c=CLASS_COLORS["citrus"], alpha=0.94, linewidths=0, depthshade=False)
    u = np.linspace(0, 2 * np.pi, 56)
    v = np.linspace(0, np.pi, 28)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_wireframe(x, y, z, rstride=3, cstride=3, color="#b3242c", linewidth=0.62, alpha=0.90)
    axis.scatter(0, 0, 0, s=24, c="#b3242c", edgecolors="white", linewidth=0.5)
    extent = max(0.050, 1.30 * radius)
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(-extent, extent)
    axis.set_box_aspect((1, 1, 1))
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])


def generate_fig8() -> dict[str, object]:
    raw_xyz, raw_packed = read_semantic_pcd(RAW_PCD)
    sor_xyz, sor_packed = read_semantic_pcd(SOR_PCD)
    sor_citrus = sor_xyz[sor_packed == PACKED["citrus"]]
    candidates, _ = p3_candidates()
    for candidate in candidates:
        indices = candidate["support_indices"]
        if len(indices) != candidate["support"] or (len(indices) and int(indices.max()) >= len(sor_citrus)):
            raise AssertionError(f"Support membership mismatch: {candidate['candidate_id']}")

    figure = plt.figure(figsize=(7.35, 6.0), facecolor="white")
    axis_a = figure.add_axes([0.035, 0.57, 0.39, 0.355])
    axis_b1 = figure.add_axes([0.455, 0.57, 0.245, 0.355])
    axis_b2 = figure.add_axes([0.72, 0.57, 0.245, 0.355])
    draw_online_semantics(axis_a, raw_xyz, raw_packed, GLOBAL_ONLINE, maximum=150000)
    draw_online_semantics(axis_b1, raw_xyz, raw_packed, ROI_1)
    draw_online_semantics(axis_b2, raw_xyz, raw_packed, ROI_2)
    for axis, scope in ((axis_b1, "ROI_1_FROZEN"), (axis_b2, "ROI_2_INDEPENDENT")):
        for candidate in candidates:
            if candidate["scope"] == scope:
                sphere_wire_2d(axis, candidate)
    roi_u, roi_v, _ = project(bounds_corners(ROI_1))
    axis_a.add_patch(Rectangle((roi_u.min(), roi_v.min()), np.ptp(roi_u), np.ptp(roi_v), fill=False, edgecolor="#333333", linewidth=0.9))
    axis_a.set_title("(a) Online 3D Semantic Map", loc="left", fontsize=10, fontweight="bold", pad=3)
    axis_b1.set_title("(b) Local Citrus Geometric Reconstruction", loc="left", fontsize=10, fontweight="bold", pad=3)
    axis_b1.text(0.02, 0.02, "Fixed ROI-1", transform=axis_b1.transAxes, fontsize=7, color="#333333")
    axis_b2.text(0.02, 0.02, "Fixed ROI-2", transform=axis_b2.transAxes, fontsize=7, color="#333333")
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS["citrus"], label="Citrus-class points", markersize=4),
        Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS["tree"], label="Tree", markersize=4),
        Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS["trunk"], label="Trunk", markersize=4),
        Line2D([], [], color="#b3242c", label="Fitted sphere hypotheses", linewidth=1.1),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.71, 0.565),
        ncol=2,
        frameon=False,
        fontsize=6.2,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    figure.text(0.04, 0.505, "(c) Representative Point-to-Sphere Geometry", fontsize=10, fontweight="bold")
    selected = select_p3_representatives(candidates)
    for index, (candidate, left) in enumerate(zip(selected, (0.05, 0.365, 0.68))):
        axis = figure.add_axes([left, 0.085, 0.27, 0.36], projection="3d")
        support = sor_citrus[candidate["support_indices"]]
        draw_local_sphere(axis, candidate, support)
        axis.view_init(elev=18, azim=-58)
        axis.set_title(
            f"Measured support n={candidate['support']}\nr={candidate['radius'] * 1000:.1f} mm",
            fontsize=7.5,
            pad=1,
        )
        if index == 0:
            axis.text2D(0.0, 0.01, "Relative XYZ (m)", transform=axis.transAxes, fontsize=7, color="#444444")
    figure.text(0.5, 0.025, "Point-supported geometric hypotheses; no fruit-identity or size ground truth", ha="center", fontsize=7.2, color="#444444")
    figure.savefig(FIGURES / "FIG8_P3_FINAL.png", dpi=500, facecolor="white")
    figure.savefig(FIGURES / "FIG8_P3_FINAL.svg", facecolor="white")
    plt.close(figure)
    return {
        "p3_final_count": len(candidates),
        "roi_1_count": sum(c["scope"] == "ROI_1_FROZEN" for c in candidates),
        "roi_2_count": sum(c["scope"] == "ROI_2_INDEPENDENT" for c in candidates),
        "boundary_limited": sum(c["boundary"] == "BOUNDARY_LIMITED" for c in candidates),
        "possible_duplicate": sum(c["duplicate"] == "POSSIBLE_DUPLICATE" for c in candidates),
        "representatives": [
            {
                "candidate_id": candidate["candidate_id"],
                "radius_m": candidate["radius"],
                "boundary_clearance_m": min(
                    candidate["radius"] - P3_RADIUS_RANGE_M[0],
                    P3_RADIUS_RANGE_M[1] - candidate["radius"],
                ),
                "passes_frozen_non_boundary_rule": min(
                    candidate["radius"] - P3_RADIUS_RANGE_M[0],
                    P3_RADIUS_RANGE_M[1] - candidate["radius"],
                ) > P3_BOUNDARY_MARGIN_M,
            }
            for candidate in selected
        ],
    }


def write_provenance(fig6_z_ranges: dict[str, float], fig8_stats: dict[str, object]) -> None:
    inputs = [
        V2 / "MANUSCRIPT_REVISED_CN_V2.docx",
        V2 / "MANUSCRIPT_REVISED_EN_V2.docx",
        CORRECTED / "tables/Table4_perception_controlled_mapping.csv",
        FINAL / "tables/Table3_pixel_semantic_performance.csv",
        FINAL / "tables/Table5_localization_continuity.csv",
        FINAL / "tables/Table6_localization_controlled_mapping.csv",
        P3_CSV,
        P3_MEMBERSHIP,
        RAW_PCD,
        SOR_PCD,
    ]
    payload = {
        "inputs": {str(path): sha256(path) for path in inputs},
        "figure_4_source": "pose_corrected_rebuild_v1 voxel histories and corrected ROI definitions",
        "figure_5_6_source": "1804 identical frequency-controlled observation times; no interpolation",
        "figure_6_camera_trajectory_z_ranges_m": fig6_z_ranges,
        "figure_8_profile": {
            "reader_facing_name": "relaxed offline sphere fitting on frozen online semantic point clouds",
            "minimum_support": 12,
            "radius_m": [0.020, 0.045],
            "inlier_distance_m": 0.010,
            "minimum_inlier_ratio": 0.30,
            "maximum_median_residual_m": 0.012,
        },
        "figure_8_statistics": fig8_stats,
        "scientific_boundary": "P3 candidates are point-supported geometric hypotheses, not verified fruit identities, counts, or diameters.",
        "outputs": {str(path): sha256(path) for path in sorted(FIGURES.glob("FIG*FINAL.*"))},
    }
    (AUDIT / "FIGURE_INPUT_PROVENANCE.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    configure_style()
    generate_fig4()
    generate_fig7()
    fig8_stats = generate_fig8()
    audit = {
        "scope": "Targeted V3.1 render-only revision of Figs. 4, 7, and 8",
        "figure_4_change": "Fixed-ROI marker size x1.8 and opacity x1.15; data, ROI, coordinates, and labels unchanged",
        "figure_7_change": "YOLOv8/YOLO11 fixed-ROI marker size x3.0 and opacity x1.25; semantic labels unchanged",
        "figure_8_change": "Legend moved outside ROI-2 data axes; no sphere center, radius, support, or point data changed",
        "figure_8_boundary_rule": {
            "radius_range_m": list(P3_RADIUS_RANGE_M),
            "boundary_margin_m": P3_BOUNDARY_MARGIN_M,
        },
        "figure_8_statistics": fig8_stats,
        "frozen_fig6_sha256": sha256(FIGURES / "FIG6_FINAL.png"),
        "source_v3_fig6_sha256": sha256(V3 / "figures/FIG6_FINAL.png"),
        "outputs": {
            str(path): sha256(path)
            for path in (FIGURES / "FIG4_FINAL.png", FIGURES / "FIG7_FINAL.png", FIGURES / "FIG8_P3_FINAL.png")
        },
    }
    if audit["frozen_fig6_sha256"] != audit["source_v3_fig6_sha256"]:
        raise AssertionError("Fig. 6 changed during the V3.1 targeted revision")
    (AUDIT / "V3_1_FIGURE_REVISION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=True))


if __name__ == "__main__":
    main()
