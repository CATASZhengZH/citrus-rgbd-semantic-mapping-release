#!/usr/bin/env python3
"""Rebuild Fig. 1 as editable, publication-resolution artwork.

The diagrams preserve the frozen scientific relations and numerical example.
They are drawn from vector primitives; no low-resolution source image is
interpolated and no generative image asset is used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(os.environ.get("CITRUS_V3_3_ROOT", Path(__file__).resolve().parents[2] / "artwork/v3_3"))
WORD = ROOT / "figures/word"
SUBMISSION = ROOT / "figures/submission"
EDITABLE = ROOT / "figures/editable_sources"

INK = "#13233a"
MUTED = "#566270"
LINE = "#5d6872"
BLUE = "#2d72b8"
BLUE_LIGHT = "#eaf4fb"
GREEN = "#4f9b5f"
GREEN_LIGHT = "#edf7ed"
ORANGE = "#f28e2b"
ORANGE_LIGHT = "#fff1de"
TRUNK = "#9b5b4f"
RED = "#a72035"
RED_LIGHT = "#fff1f3"
PURPLE = "#7a60a8"
PURPLE_LIGHT = "#f1edf9"
GRAY_LIGHT = "#f4f6f7"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stixsans",
        }
    )


def box(ax, xy, width, height, *, face="white", edge=LINE, radius=0.012, lw=0.8, ls="-"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle=ls,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, size=8, weight="normal", color=INK, ha="center", va="center", **kwargs):
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kwargs)


def arrow(ax, start, end, *, color=LINE, lw=1.2, scale=10, style="-|>"):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale, linewidth=lw, color=color)
    ax.add_patch(patch)
    return patch


def draw_orchard(ax, x, y, w, h, *, semantic=False, border=True):
    if border:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#89939c", linewidth=0.55))
    if not semantic:
        ax.add_patch(Rectangle((x, y), w, h * 0.28, facecolor="#d5c4a4", edgecolor="none"))
        ax.add_patch(Rectangle((x, y + h * 0.28), w, h * 0.72, facecolor="#d9e8ef", edgecolor="none"))
    canopy = GREEN if semantic else "#567f45"
    trunk = TRUNK if semantic else "#765846"
    citrus = ORANGE if semantic else "#e7a126"
    for center, scale in ((0.27, 0.95), (0.70, 1.07)):
        cx = x + center * w
        ax.add_patch(Rectangle((cx - 0.024 * w, y + 0.11 * h), 0.048 * w, 0.35 * h, facecolor=trunk, edgecolor="none"))
        circles = ((-0.12, 0.56, 0.19), (0.02, 0.68, 0.22), (0.14, 0.54, 0.18))
        for dx, cy, radius in circles:
            ax.add_patch(Circle((cx + dx * w * scale, y + cy * h), radius * h * scale, facecolor=canopy, edgecolor="none"))
        for dx, dy in ((-0.11, 0.62), (0.08, 0.72), (0.16, 0.52), (-0.01, 0.48)):
            ax.add_patch(Circle((cx + dx * w * scale, y + dy * h), 0.035 * h, facecolor=citrus, edgecolor="#c66e12", linewidth=0.25))


def draw_depth(ax, x, y, w, h):
    colors = ("#254ca3", "#1d99c1", "#42c794", "#f3d34a", "#ee7d32")
    for i, color in enumerate(colors):
        ax.add_patch(Rectangle((x + i * w / len(colors), y), w / len(colors), h, facecolor=color, edgecolor="none"))
    for cx, cy, r in ((0.20, 0.57, 0.19), (0.62, 0.62, 0.24), (0.80, 0.49, 0.17)):
        ax.add_patch(Circle((x + cx * w, y + cy * h), r * h, facecolor="#8ee18c", edgecolor="none"))
    ax.add_patch(Rectangle((x + 0.60 * w, y + 0.08 * h), 0.04 * w, 0.44 * h, facecolor="#f27e32", edgecolor="none"))
    ax.add_patch(Rectangle((x + 0.18 * w, y + 0.08 * h), 0.035 * w, 0.40 * h, facecolor="#48d7ba", edgecolor="none"))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#89939c", linewidth=0.55))


def draw_imu(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#89939c", linewidth=0.55))
    xs = np.linspace(x + 0.06 * w, x + 0.96 * w, 30)
    for offset, color, phase in ((0.73, BLUE, 0.0), (0.50, ORANGE, 0.8), (0.27, GREEN, 1.7)):
        ys = y + h * (offset + 0.065 * np.sin(np.linspace(0, 4 * np.pi, len(xs)) + phase))
        ax.plot(xs, ys, color=color, linewidth=0.65)
    for yy, txt in ((0.74, r"$a_x$"), (0.51, r"$a_y$"), (0.28, r"$\omega_z$")):
        label(ax, x + 0.025 * w, y + yy * h, txt, size=5.5, ha="left")


def draw_camera_schematic(ax, x, y, w, h):
    body = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.015", facecolor="#d9dde0", edgecolor="#4e555b", linewidth=0.9)
    ax.add_patch(body)
    ax.add_patch(Rectangle((x + 0.05 * w, y + 0.17 * h), 0.90 * w, 0.63 * h, facecolor="#2b3136", edgecolor="none"))
    for cx in (0.20, 0.50, 0.79):
        ax.add_patch(Circle((x + cx * w, y + 0.49 * h), 0.12 * h, facecolor="#101820", edgecolor="#6e7d89", linewidth=0.8))
        ax.add_patch(Circle((x + cx * w, y + 0.49 * h), 0.055 * h, facecolor="#274c5b", edgecolor="#8da6af", linewidth=0.45))
    label(ax, x + 0.50 * w, y + 0.94 * h, "RGB-D + IMU", size=6.4, weight="bold")


def draw_trajectory(ax, x, y, w, h):
    t = np.linspace(0, 2 * np.pi, 160)
    xx = x + w * (0.50 + 0.39 * np.cos(t) + 0.09 * np.cos(3 * t))
    yy = y + h * (0.48 + 0.32 * np.sin(t) - 0.08 * np.sin(2 * t))
    ax.plot(xx, yy, color=BLUE, linewidth=1.4)
    ax.scatter([xx[0]], [yy[0]], s=9, c=GREEN, zorder=4)
    ax.scatter([xx[-2]], [yy[-2]], s=9, c=RED, zorder=4)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#89939c", linewidth=0.55))
    label(ax, x + 0.02 * w, y + 0.96 * h, r"Camera trajectory $T_{wc}(t)$", size=6.2, ha="left", va="top")


def draw_point_cloud(ax, x, y, w, h, *, dense=False):
    rng = np.random.default_rng(23 if dense else 17)
    n = 300 if dense else 150
    centers = np.array([[0.30, 0.56], [0.66, 0.61], [0.51, 0.40]])
    pts = []
    for center in centers:
        cloud = center + rng.normal(scale=(0.13, 0.11), size=(n // len(centers), 2))
        pts.append(cloud)
    pts = np.vstack(pts)
    pts = pts[(pts[:, 0] > 0.05) & (pts[:, 0] < 0.95) & (pts[:, 1] > 0.12) & (pts[:, 1] < 0.93)]
    ax.scatter(x + pts[:, 0] * w, y + pts[:, 1] * h, s=2.8 if dense else 3.3, c=GREEN, linewidths=0)
    citrus = np.array([[0.23, 0.60], [0.34, 0.69], [0.59, 0.70], [0.72, 0.58], [0.53, 0.48]])
    ax.scatter(x + citrus[:, 0] * w, y + citrus[:, 1] * h, s=13, c=ORANGE, edgecolors="#bd650c", linewidths=0.25)
    ax.plot([x + 0.30 * w, x + 0.30 * w], [y + 0.12 * h, y + 0.45 * h], color=TRUNK, linewidth=3)
    ax.plot([x + 0.67 * w, x + 0.67 * w], [y + 0.12 * h, y + 0.47 * h], color=TRUNK, linewidth=3)


def export(fig, stem: str) -> None:
    WORD.mkdir(parents=True, exist_ok=True)
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    EDITABLE.mkdir(parents=True, exist_ok=True)
    fig.savefig(WORD / f"{stem}.png", dpi=600, facecolor="white", bbox_inches=None)
    fig.savefig(EDITABLE / f"{stem}.svg", facecolor="white", bbox_inches=None)
    fig.savefig(SUBMISSION / f"{stem}.eps", facecolor="white", bbox_inches=None)
    tif = SUBMISSION / f"{stem}.tif"
    fig.savefig(
        tif,
        dpi=600,
        facecolor="white",
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(tif) as image:
        image.convert("RGB").save(tif, compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)


def fig1() -> None:
    fig = plt.figure(figsize=(7.10, 5.325), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    panels = [
        (0.015, 0.585, 0.307, 0.395, "RGB-D Acquisition System", RED_LIGHT, RED),
        (0.337, 0.585, 0.322, 0.395, "Semantic Perception System", BLUE_LIGHT, RED),
        (0.674, 0.585, 0.311, 0.395, "Localization System", GREEN_LIGHT, RED),
    ]
    for x, y, w, h, title, face, color in panels:
        box(ax, (x, y), w, h, face=face, edge=LINE, radius=0.012, lw=0.9)
        label(ax, x + w / 2, y + h - 0.029, title, size=9.1, weight="bold", color=color)

    # Acquisition panel: author-created camera schematic and vector sensor cues.
    draw_camera_schematic(ax, 0.035, 0.702, 0.142, 0.105)
    label(ax, 0.106, 0.672, "Intel RealSense D435i", size=6.5)
    arrow(ax, (0.182, 0.758), (0.202, 0.758), scale=7)
    label(ax, 0.212, 0.915, "RGB", size=6.6, weight="bold", ha="left")
    draw_orchard(ax, 0.212, 0.829, 0.094, 0.073)
    label(ax, 0.212, 0.812, "Depth", size=6.6, weight="bold", ha="left")
    draw_depth(ax, 0.212, 0.727, 0.094, 0.073)
    label(ax, 0.212, 0.710, "IMU", size=6.6, weight="bold", ha="left")
    draw_imu(ax, 0.212, 0.615, 0.094, 0.083)

    # Semantic panel.
    label(ax, 0.349, 0.802, "RGB frames", size=6.4, ha="left")
    draw_orchard(ax, 0.349, 0.700, 0.080, 0.090)
    arrow(ax, (0.431, 0.744), (0.447, 0.744), scale=7)
    box(ax, (0.448, 0.625), 0.115, 0.290, face="white", edge="#7f8992", radius=0.009, ls="--")
    label(ax, 0.505, 0.889, "Candidate segmentation\nbackbones", size=5.2, weight="bold")
    models = (("BiSeNetV1", BLUE_LIGHT, BLUE), ("YOLOv8", GREEN_LIGHT, GREEN), ("YOLO11", ORANGE_LIGHT, ORANGE), ("YOLO26", PURPLE_LIGHT, PURPLE))
    for i, (name, face, edge) in enumerate(models):
        yy = 0.842 - i * 0.057
        box(ax, (0.463, yy), 0.085, 0.037, face=face, edge=edge, radius=0.010, lw=0.55)
        label(ax, 0.505, yy + 0.0185, name, size=6.3)
    arrow(ax, (0.566, 0.744), (0.581, 0.744), scale=7)
    label(ax, 0.586, 0.802, "Semantic masks", size=5.2, ha="left")
    draw_orchard(ax, 0.586, 0.700, 0.060, 0.090, semantic=True)
    for i, (name, color) in enumerate((("citrus", ORANGE), ("tree", GREEN), ("trunk", TRUNK))):
        ax.add_patch(Rectangle((0.590, 0.672 - i * 0.028), 0.010, 0.015, facecolor=color, edgecolor="none"))
        label(ax, 0.604, 0.680 - i * 0.028, name, size=5.6, ha="left")

    # Localization panel.
    box(ax, (0.690, 0.635), 0.119, 0.276, face="white", edge="#7f8992", radius=0.009, ls="--")
    label(ax, 0.749, 0.887, "Localization backends", size=5.3, weight="bold")
    for i, (name, face, edge) in enumerate(
        (("VINS-RGBD", BLUE_LIGHT, BLUE), ("ORB-SLAM3", GREEN_LIGHT, GREEN), ("RTAB-Map", ORANGE_LIGHT, ORANGE))
    ):
        yy = 0.818 - i * 0.067
        box(ax, (0.704, yy), 0.091, 0.043, face=face, edge=edge, radius=0.009, lw=0.55)
        label(ax, 0.7495, yy + 0.0215, name, size=6.2)
    arrow(ax, (0.811, 0.744), (0.827, 0.744), scale=7)
    draw_trajectory(ax, 0.831, 0.662, 0.132, 0.211)
    # Coordinate triad.
    base = (0.950, 0.610)
    arrow(ax, base, (0.973, 0.610), color=RED, scale=6, lw=0.8)
    arrow(ax, base, (0.963, 0.631), color=GREEN, scale=6, lw=0.8)
    arrow(ax, base, (0.950, 0.645), color=BLUE, scale=6, lw=0.8)
    label(ax, 0.978, 0.610, "X", size=5.4, color=RED, ha="left")
    label(ax, 0.965, 0.636, "Y", size=5.4, color=GREEN, ha="left")
    label(ax, 0.950, 0.650, "Z", size=5.4, color=BLUE)

    # Link three inputs into the controlled mapping framework.
    ax.plot([0.168, 0.168, 0.499], [0.583, 0.558, 0.558], color=LINE, linewidth=1.2)
    ax.plot([0.499, 0.499], [0.583, 0.532], color=LINE, linewidth=1.2)
    ax.plot([0.830, 0.830, 0.499], [0.583, 0.558, 0.558], color=LINE, linewidth=1.2)
    arrow(ax, (0.499, 0.558), (0.499, 0.535), scale=8)

    box(ax, (0.015, 0.025), 0.970, 0.490, face="#fffdf8", edge=LINE, radius=0.012, lw=0.9)
    label(ax, 0.50, 0.481, "Semantic Mapping and Controlled Evaluation Framework", size=9.3, weight="bold", color=RED)

    step_x = [0.030, 0.157, 0.291, 0.412, 0.540, 0.671]
    step_w = [0.116, 0.123, 0.110, 0.116, 0.120, 0.154]
    for idx, (xx, ww) in enumerate(zip(step_x, step_w), 1):
        box(ax, (xx, 0.058), ww, 0.381, face="white", edge="#7d8790", radius=0.007, lw=0.65)
        if idx < 6:
            arrow(ax, (xx + ww + 0.003, 0.250), (step_x[idx] - 0.004, 0.250), scale=8)

    label(ax, 0.088, 0.410, "1. Input\nobservations", size=5.2, weight="bold")
    label(ax, 0.038, 0.369, "RGB frames", size=5.4, ha="left")
    draw_orchard(ax, 0.040, 0.290, 0.096, 0.066)
    label(ax, 0.038, 0.274, "Depth map", size=5.4, ha="left")
    draw_depth(ax, 0.040, 0.190, 0.096, 0.070)
    label(ax, 0.038, 0.174, "Semantic mask", size=5.4, ha="left")
    draw_orchard(ax, 0.040, 0.081, 0.096, 0.080, semantic=True)

    label(ax, 0.2185, 0.407, "2. Depth-semantic\nprojection", size=5.2, weight="bold")
    label(ax, 0.2185, 0.354, "Back-project image\npixels to 3D points", size=5.2)
    # Projection schematic.
    ax.add_patch(Polygon(((0.172, 0.202), (0.194, 0.315), (0.194, 0.150)), closed=True, facecolor="#d8e1e8", edgecolor=LINE, linewidth=0.6))
    for yy, color in ((0.286, ORANGE), (0.250, GREEN), (0.216, BLUE), (0.176, ORANGE)):
        ax.add_patch(Circle((0.190, yy), 0.004, facecolor=color, edgecolor="none"))
        ax.plot([0.194, 0.250], [yy, yy + (yy - 0.23) * 0.55], color="#9aa3aa", linewidth=0.45, linestyle="--")
        ax.add_patch(Circle((0.250, yy + (yy - 0.23) * 0.55), 0.004, facecolor=color, edgecolor="none"))
    label(ax, 0.2185, 0.105, r"$p_c=D(u,v)K^{-1}[u,v,1]^T$", size=5.4)

    label(ax, 0.346, 0.407, "3. 3D semantic\nobservations", size=5.2, weight="bold")
    label(ax, 0.346, 0.354, r"World points $p_W$", size=5.2)
    draw_point_cloud(ax, 0.305, 0.125, 0.082, 0.210)

    label(ax, 0.470, 0.407, "4. World-coordinate\ntransformation", size=5.2, weight="bold")
    label(ax, 0.470, 0.350, "Transform using\ncamera pose $T_{wc}$", size=5.2)
    label(ax, 0.470, 0.278, r"$\tilde{p}_W=T_{wc}\,\tilde{p}_c$", size=6.0, weight="bold")
    label(ax, 0.440, 0.170, "{C}", size=6.0)
    label(ax, 0.503, 0.104, "{W}", size=6.0)
    arrow(ax, (0.447, 0.155), (0.487, 0.125), color=BLUE, scale=6, lw=0.8)
    arrow(ax, (0.447, 0.155), (0.430, 0.185), color=GREEN, scale=6, lw=0.8)
    arrow(ax, (0.447, 0.155), (0.465, 0.180), color=RED, scale=6, lw=0.8)
    arrow(ax, (0.496, 0.105), (0.522, 0.105), color=RED, scale=6, lw=0.8)
    arrow(ax, (0.496, 0.105), (0.508, 0.130), color=GREEN, scale=6, lw=0.8)
    arrow(ax, (0.496, 0.105), (0.496, 0.145), color=BLUE, scale=6, lw=0.8)
    arrow(ax, (0.456, 0.168), (0.497, 0.135), style="-|>", scale=7)

    label(ax, 0.600, 0.407, "5. Voxel semantic\nfusion (0.05 m)", size=5.2, weight="bold")
    label(ax, 0.600, 0.350, "Colored voxel grid", size=5.2)
    for ix in range(4):
        for iy in range(5):
            color = GREEN if (ix + iy) % 3 else ORANGE
            if iy < 2 and ix == 1:
                color = TRUNK
            ax.add_patch(Rectangle((0.565 + ix * 0.017, 0.160 + iy * 0.026), 0.015, 0.022, facecolor=color, edgecolor="#42484d", linewidth=0.28))
    for i, (name, color) in enumerate((("citrus", ORANGE), ("tree", GREEN), ("trunk", TRUNK))):
        ax.add_patch(Rectangle((0.560, 0.121 - i * 0.025), 0.011, 0.014, facecolor=color, edgecolor="none"))
        label(ax, 0.575, 0.128 - i * 0.025, name, size=5.0, ha="left")

    label(ax, 0.748, 0.407, "6. Accumulated 3D\nsemantic map", size=5.2, weight="bold")
    label(ax, 0.748, 0.350, "Dense 3D semantic map", size=5.2)
    draw_point_cloud(ax, 0.690, 0.119, 0.116, 0.205, dense=True)
    for i, (name, color) in enumerate((("citrus", ORANGE), ("tree", GREEN), ("trunk", TRUNK))):
        ax.add_patch(Rectangle((0.687 + i * 0.047, 0.080), 0.010, 0.014, facecolor=color, edgecolor="none"))
        label(ax, 0.699 + i * 0.047, 0.087, name, size=4.8, ha="left")

    box(ax, (0.839, 0.058), 0.131, 0.381, face="white", edge="#7f8992", radius=0.007, lw=0.65, ls="--")
    label(ax, 0.9045, 0.414, "Controlled\nexperiments", size=5.3, weight="bold", color=RED)
    box(ax, (0.852, 0.260), 0.105, 0.126, face=BLUE_LIGHT, edge=BLUE, radius=0.009, lw=0.55)
    label(ax, 0.9045, 0.363, "Experiment A", size=6.4, weight="bold")
    label(ax, 0.9045, 0.315, "Fixed VINS-RGBD\nCompare four\nsegmentation\nbackbones", size=4.8)
    box(ax, (0.852, 0.091), 0.105, 0.137, face=GREEN_LIGHT, edge=GREEN, radius=0.009, lw=0.55)
    label(ax, 0.9045, 0.205, "Experiment B", size=6.4, weight="bold")
    label(ax, 0.9045, 0.149, "Fixed YOLOv8\nCompare three\nlocalization\nbackends", size=4.8)

    export(fig, "FIG1_FINAL")


def draw_vote_scene(ax, x, y, w, h, variant: int):
    # Vector schematic of the unchanged three-frame example; the numerical
    # observations remain explicit beside each scene.
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#dbe9ed", edgecolor="#7f8b93", linewidth=0.45))
    ax.add_patch(Rectangle((x, y), w, h * 0.25, facecolor="#cdbb96", edgecolor="none"))
    ax.add_patch(Rectangle((x + 0.44 * w, y + 0.08 * h), 0.08 * w, 0.45 * h, facecolor=TRUNK, edgecolor="none"))
    ax.add_patch(Circle((x + 0.48 * w, y + 0.62 * h), 0.31 * h, facecolor="#4b7c42", edgecolor="none"))
    positions = (
        ((0.28, 0.62), (0.45, 0.73), (0.63, 0.57), (0.53, 0.44)),
        ((0.38, 0.67), (0.61, 0.54)),
        ((0.26, 0.57), (0.45, 0.71), (0.62, 0.58), (0.57, 0.40)),
    )[variant]
    for px, py in positions:
        ax.add_patch(Circle((x + px * w, y + py * h), 0.050 * h, facecolor=ORANGE, edgecolor="#c56b12", linewidth=0.25))


def draw_semantic_scene(ax, x, y, w, h, variant: int):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#111820", edgecolor="#7f8b93", linewidth=0.45))
    ax.add_patch(Rectangle((x + 0.44 * w, y + 0.08 * h), 0.08 * w, 0.45 * h, facecolor=TRUNK, edgecolor="none"))
    ax.add_patch(Circle((x + 0.48 * w, y + 0.62 * h), 0.31 * h, facecolor=GREEN, edgecolor="none"))
    positions = (
        ((0.28, 0.62), (0.45, 0.73), (0.63, 0.57), (0.53, 0.44)),
        ((0.38, 0.67), (0.61, 0.54)),
        ((0.26, 0.57), (0.45, 0.71), (0.62, 0.58), (0.57, 0.40)),
    )[variant]
    for px, py in positions:
        ax.add_patch(Circle((x + px * w, y + py * h), 0.052 * h, facecolor=ORANGE, edgecolor="#b45d0b", linewidth=0.22))


def fig2() -> None:
    fig = plt.figure(figsize=(7.10, 2.1078), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, (0.006, 0.025), 0.416, 0.950, face=BLUE_LIGHT, edge=BLUE, radius=0.009, lw=0.65)
    box(ax, (0.431, 0.025), 0.318, 0.950, face=GREEN_LIGHT, edge=GREEN, radius=0.009, lw=0.65)
    box(ax, (0.758, 0.025), 0.236, 0.950, face=RED_LIGHT, edge=RED, radius=0.009, lw=0.65)

    label(ax, 0.214, 0.930, "(a)  Within-frame aggregation", size=7.1, weight="bold", color="#183866")
    label(ax, 0.214, 0.872, "Pixel observations → one voxel, one vote (per frame)", size=5.8, color="#183866")
    label(ax, 0.591, 0.930, "(b)  Cross-view equal voting", size=7.1, weight="bold", color="#17481d")
    label(ax, 0.591, 0.872, "Frame-level votes → multi-view accumulation", size=5.8, color="#17481d")
    label(ax, 0.876, 0.930, "(c)  Map-level evaluation", size=7.1, weight="bold", color="#6c1724")
    label(ax, 0.876, 0.872, "Vote distribution → map metrics", size=5.8, color="#6c1724")

    counts = ((12, 3, 1, "citrus"), (2, 9, 0, "tree"), (4, 1, 0, "citrus"))
    row_y = (0.690, 0.480, 0.270)
    for i, (cy, ty, ry, vote) in enumerate(counts):
        y = row_y[i]
        label(ax, 0.050, y + 0.020, rf"Frame $t_{i+1}$", size=5.0, ha="right")
        draw_vote_scene(ax, 0.055, y - 0.060, 0.058, 0.135, i)
        draw_semantic_scene(ax, 0.118, y - 0.060, 0.058, 0.135, i)
        label(ax, 0.181, y + 0.091, "Pixel counts in voxel $q$" if i == 0 else "", size=4.6, ha="left")
        for j, (name, value, color) in enumerate((("citrus", cy, ORANGE), ("tree", ty, GREEN), ("trunk", ry, "#17385f"))):
            yy = y + 0.040 - j * 0.045
            ax.add_patch(Circle((0.186, yy), 0.0044, facecolor=color, edgecolor="none"))
            label(ax, 0.194, yy, f"{name}: {value}", size=5.0, ha="left")
        arrow(ax, (0.248, y + 0.005), (0.260, y + 0.005), scale=6, lw=0.8)
        box(ax, (0.264, y - 0.047), 0.067, 0.105, face="white", edge="#9aa3aa", radius=0.006, lw=0.5)
        label(ax, 0.2975, y + 0.020, "Frame-level\ndecision\n(Eq. 5)", size=4.8)
        arrow(ax, (0.333, y + 0.005), (0.346, y + 0.005), scale=6, lw=0.8)
        vote_face = ORANGE_LIGHT if vote == "citrus" else GREEN_LIGHT
        vote_edge = ORANGE if vote == "citrus" else GREEN
        box(ax, (0.350, y - 0.034), 0.058, 0.078, face=vote_face, edge=vote_edge, radius=0.006, lw=0.55)
        label(ax, 0.379, y + 0.006, f"Vote:\n{vote}", size=5.0, weight="bold")
        arrow(ax, (0.410, y + 0.005), (0.448, y + 0.005), scale=6, lw=0.8)

    box(ax, (0.023, 0.060), 0.386, 0.112, face="white", edge=BLUE, radius=0.006, lw=0.55, ls="--")
    label(ax, 0.216, 0.116, "Multiple pixels in the same voxel (possibly with different classes)\nproduce one class vote per frame (count, confidence, and tie-break).", size=5.0, color="#183866")

    # Cross-view accumulation.
    for i, (vote, color, face) in enumerate((("citrus", ORANGE, ORANGE_LIGHT), ("tree", GREEN, GREEN_LIGHT), ("citrus", ORANGE, ORANGE_LIGHT))):
        yy = row_y[i]
        box(ax, (0.444, yy - 0.028), 0.055, 0.066, face=face, edge=color, radius=0.006, lw=0.55)
        label(ax, 0.4715, yy + 0.005, vote, size=5.0, weight="bold")
        arrow(ax, (0.500, yy + 0.005), (0.538, 0.565), scale=8, lw=1.4)

    # Vector voxel cube.
    cube_x, cube_y = 0.560, 0.535
    for i in range(4):
        ax.plot([cube_x + i * 0.025, cube_x + i * 0.025], [cube_y - 0.125, cube_y + 0.145], color="#77818a", linewidth=0.4)
        ax.plot([cube_x, cube_x + 0.075], [cube_y - 0.125 + i * 0.068, cube_y - 0.125 + i * 0.068], color="#77818a", linewidth=0.4)
    ax.add_patch(Rectangle((cube_x + 0.025, cube_y - 0.057), 0.025, 0.068, facecolor="#f5b065", edgecolor="#a55c11", linewidth=0.5))
    label(ax, cube_x + 0.037, cube_y - 0.157, "Voxel $q$\n($s=0.05$ m)", size=6.0)

    box(ax, (0.651, 0.622), 0.083, 0.210, face="white", edge=GREEN, radius=0.006, lw=0.55, ls="--")
    label(ax, 0.6925, 0.797, "Accumulated votes", size=5.0, weight="bold")
    label(ax, 0.6925, 0.736, r"$N_{q,\mathrm{citrus}}=2$", size=6.0)
    label(ax, 0.6925, 0.690, r"$N_{q,\mathrm{tree}}=1$", size=6.0)
    label(ax, 0.6925, 0.644, r"$N_{q,\mathrm{trunk}}=0$", size=6.0)
    arrow(ax, (0.6925, 0.618), (0.6925, 0.574), scale=7)
    box(ax, (0.651, 0.482), 0.083, 0.077, face="white", edge="#9aa3aa", radius=0.006, lw=0.5)
    label(ax, 0.6925, 0.520, "Final label\n(Eq. 7)", size=5.7)
    arrow(ax, (0.6925, 0.478), (0.6925, 0.434), scale=7)
    box(ax, (0.650, 0.357), 0.085, 0.061, face=ORANGE_LIGHT, edge=ORANGE, radius=0.006, lw=0.55)
    label(ax, 0.6925, 0.387, r"$c_q^*=$ citrus", size=6.3, weight="bold")
    box(ax, (0.444, 0.060), 0.291, 0.112, face="white", edge=GREEN, radius=0.006, lw=0.55, ls="--")
    label(ax, 0.5895, 0.116, "Each frame contributes one vote with equal weight,\nregardless of the number of pixels.", size=5.1, color="#17481d")

    # Map-level metrics.
    box(ax, (0.781, 0.555), 0.190, 0.275, face="white", edge=RED, radius=0.006, lw=0.55, ls="--")
    label(ax, 0.876, 0.789, "Class probability of voxel $q$", size=5.5, weight="bold")
    label(ax, 0.876, 0.721, r"$p(\mathrm{citrus}\mid q)=\frac{2}{3}$", size=7.1)
    label(ax, 0.876, 0.657, r"$p(\mathrm{tree}\mid q)=\frac{1}{3}$", size=7.1)
    label(ax, 0.876, 0.593, r"$p(\mathrm{trunk}\mid q)=0$", size=7.1)
    box(ax, (0.774, 0.292), 0.100, 0.210, face="white", edge="#d36375", radius=0.006, lw=0.5)
    label(ax, 0.824, 0.467, "Semantic\nconflict", size=5.0, weight="bold", color="#183866")
    label(ax, 0.824, 0.405, r"$|\{c:N_{q,c}>0\}|=2$", size=6.2)
    label(ax, 0.824, 0.353, "→ Conflict:  Yes", size=6.2, weight="bold", color=RED)
    label(ax, 0.824, 0.315, "(Eq. 11)", size=5.2)
    box(ax, (0.885, 0.292), 0.100, 0.210, face="white", edge="#d36375", radius=0.006, lw=0.5)
    label(ax, 0.935, 0.467, "Normalized\nentropy", size=5.0, weight="bold", color="#183866")
    label(ax, 0.935, 0.405, r"$H(q)=0.58$", size=7.2)
    label(ax, 0.935, 0.343, "(Eq. 12)", size=5.6)
    box(ax, (0.774, 0.060), 0.211, 0.173, face="white", edge=RED, radius=0.006, lw=0.55, ls="--")
    label(ax, 0.8795, 0.146, "Voxel-level vote statistics are used to\ncompute SCR, normalized semantic entropy,\nand other map-level metrics (Sec. 2.2.4).", size=4.8, color="#5c2330")

    export(fig, "FIG2_FINAL")


def main() -> None:
    configure()
    fig1()
    print("Generated editable SVG, submission EPS/TIFF, and 600 dpi Word PNG for Fig. 1.")


if __name__ == "__main__":
    main()
