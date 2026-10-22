# Environment separation

Use three separate environments:

1. ROS online runtime: Ubuntu 20.04 and ROS Noetic for the VINS-RGBD pipeline.
2. Offline mapping/evaluation: Python 3.8-compatible scientific and deep-learning packages.
3. Artwork: Matplotlib, Pillow, python-docx, and LibreOffice for document rendering checks.

The archived experiment was not validated as a Docker image. GPU and TensorRT details are machine-specific and should be revalidated before use.
