# Controlled evaluation of semantic segmentation and localization methods for RGB-D semantic mapping in citrus orchards

Research code and reproducibility material for a manuscript currently in submission preparation. This repository supports controlled comparisons of semantic segmentation front ends and localization back ends for accumulated RGB-D semantic mapping in a citrus orchard.

## Overview

The evaluation separates two controlled experiments. The perception comparison fixes the VINS-RGBD trajectory while changing the semantic front end (BiSeNetV1, YOLOv8, YOLO11, and YOLO26). The localization comparison fixes frozen YOLOv8 semantic observations while changing the localization source (VINS-RGBD, ORB-SLAM3, and RTAB-Map).

## Method

RGB-D observations are projected into 3D and transformed with the documented camera pose chain. Semantic evidence is fused by frame-balanced voxel voting. Map consistency is summarized with semantic conflict rate (SCR) and normalized semantic entropy. Trajectory availability on native timestamps is reported separately from continuity on the shared timestamp set.

## Reproducibility scope

- Table 3 uses a fixed 91-image manually annotated test split and frozen model checkpoints.
- Table 4 uses 5,159 RGB-D observations with valid matched VINS poses, the corrected camera pose chain, and 0.05 m voxels.
- Tables 5 and 6 use 1,804 common observation times without trajectory interpolation.
- Figure 8 shows 84 point-supported local sphere hypotheses (52 in ROI-1 and 32 in ROI-2) from a relaxed offline fit to frozen online semantic point clouds. Of these, 33 carry a radius-boundary risk flag and 28 carry a possible-duplicate flag. They are not fruit-count ground truth.

## Dependencies

The online runtime targets Ubuntu 20.04 and ROS Noetic. Offline evaluation uses Python with NumPy, SciPy, OpenCV, PyTorch, Ultralytics, PyYAML, Matplotlib, and Pillow. See `requirements/` and `docs/environment.md`.

## Installation

Create separate ROS and offline Python environments. Install the offline requirements with:

```bash
python3 -m pip install -r requirements/offline.txt
```

No untested Docker image is claimed.

## Data preparation

The raw rosbag, the complete 911-image dataset, ground-truth annotations, model weights, and full point clouds are not distributed here because data and redistribution permissions require author confirmation. Configure local paths through environment variables or the example YAML files under `configs/`. Do not commit private data or credentials.

## Running examples

The evaluation scripts expose command-line interfaces. Start with:

```bash
python3 evaluation/evaluate_semantic_map.py --help
python3 evaluation/evaluate_segmentation_models.py --help
python3 mapping/formal_semantic_mapping.py --help
```

The formal mapping script additionally reads `CITRUS_PAPER_ROOT`, `CITRUS_STAGE7_ROOT`, and `BISENET_REPOSITORY` when applicable.

## Controlled evaluation

The perception-controlled protocol fixes RGB-D inputs, camera trajectory, projection rules, voxel size, and fusion. The localization-controlled protocol fixes YOLOv8 observations and the 1,804 common observation times. These protocols have different observation scopes and their absolute map statistics should not be compared without that distinction.

## Evaluation metrics

`evaluation/evaluate_semantic_map.py` implements frame-balanced semantic voting summaries, SCR, and normalized semantic entropy. Trajectory scripts report native coverage and common-set step continuity without interpolation, scale correction, or ICP.

## Visualization

Permission-cleared previews are under `figures/`. Figure 1 is withheld pending confirmation for the embedded camera product material; Figure 3 is withheld pending confirmation for orchard-image redistribution. Figure-generation scripts are retained with configurable roots but require the private frozen inputs.

## Citation

The manuscript is not yet published or accepted. See `CITATION.md`; no DOI is available.

## License and permissions

No project-wide open-source license is granted in this release pending author confirmation. Third-party components remain under their original licenses. See `LICENSES_AND_PERMISSIONS.md`.
