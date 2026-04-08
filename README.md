# strayscanner_converter

Tools for processing and visualizing [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662) LiDAR data captured on iPhone.

## What it does

Stray Scanner exports an mp4 video alongside per-frame depth and confidence maps, but does not provide synchronized RGB images. This project:

- **extract_rgb_frames.py** — Extracts individual PNG frames from the mp4 video so they are synchronized 1:1 with depth/confidence data.
- **visualize_depth.py** — Generates a 2x2 visualization grid (RGB, raw depth, confidence-filtered depth, confidence map) for any given frame.

## Stray Scanner data format

A typical Stray Scanner export contains:

```
scan_data/
├── rgb.mp4              # RGB video
├── depth/               # Per-frame 16-bit depth PNGs (mm)
│   ├── 000000.png
│   └── ...
├── confidence/          # Per-frame confidence PNGs (0=low, 1=medium, 2=high)
│   ├── 000000.png
│   └── ...
├── camera_matrix.csv    # 3x3 intrinsic matrix
├── odometry.csv         # Per-frame pose (timestamp, position, quaternion)
└── imu.csv              # IMU readings
```

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/MasahiroOgawa/strayscanner_converter.git
cd strayscanner_converter
uv sync
```

## Usage

### 1. Extract RGB frames

```bash
uv run python src/extract_rgb_frames.py <path_to_scan>/rgb.mp4
```

Frames are saved to `<path_to_scan>/images/` by default.

```bash
# Custom output directory
uv run python src/extract_rgb_frames.py <path_to_scan>/rgb.mp4 -o /custom/output/dir
```

### 2. Visualize depth

```bash
uv run python src/visualize_depth.py <path_to_scan> --rgb-dir <path_to_scan>/images --frame 10
```

Output PNG is saved to `<path_to_scan>/images/depth_vis_000010.png` by default.

```bash
# Custom output path
uv run python src/visualize_depth.py <path_to_scan> --rgb-dir <path_to_scan>/images --frame 10 -o custom_output.png
```
