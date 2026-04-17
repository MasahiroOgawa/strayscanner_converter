# strayscanner_converter

Tools for processing and visualizing [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662) LiDAR data captured on iPhone.

## What it does

Stray Scanner exports an mp4 video alongside per-frame depth and confidence maps, but does not provide synchronized RGB images. This project:

- **extract_rgb_frames.py** — Extracts individual PNG frames from the mp4 video so they are synchronized 1:1 with depth/confidence data.
- **visualize_depth.py** — Generates a 2x2 visualization grid (RGB, raw depth, confidence-filtered depth, confidence map) for a given frame, so you can inspect depth quality and choose a confidence threshold before downstream use.
- **convert_to_colmap.py** — Converts the extracted frames, intrinsics, and odometry poses into a COLMAP sparse reconstruction (`cameras.bin`, `images.bin`, `points3D.bin`), with an initial point cloud unprojected from the depth maps. The output is ready to use as input for 3D Gaussian Splatting.

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

Inspect depth quality and the effect of confidence filtering on a single frame. Useful for sanity-checking a scan and picking a confidence threshold before running downstream pipelines.

```bash
uv run python src/visualize_depth.py <path_to_scan> --rgb-dir <path_to_scan>/images --frame 10
```

Output PNG is saved to `<path_to_scan>/images/depth_vis_000010.png` by default.

```bash
# Custom output path
uv run python src/visualize_depth.py <path_to_scan> --rgb-dir <path_to_scan>/images --frame 10 -o custom_output.png
```

### 3. Convert to COLMAP format (for 3D Gaussian Splatting)

Produces a COLMAP sparse model from the extracted frames plus the Stray Scanner intrinsics/odometry, and seeds `points3D.bin` by unprojecting the depth maps into the world frame. If `<path_to_scan>/images/` is empty or missing, frames are automatically extracted from `rgb.mp4` first, so step 1 is optional.

```bash
uv run python src/convert_to_colmap.py <path_to_scan>
```

This writes to `<path_to_scan>/` by default:

```
<path_to_scan>/
├── images/                 # symlinked if output dir differs
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

```bash
# Custom output directory and subsampling
uv run python src/convert_to_colmap.py <path_to_scan> /custom/output/dir \
    --subsample-frames 10 --subsample-pixels 3
```

`--subsample-frames N` uses every Nth frame when generating the point cloud; `--subsample-pixels N` uses every Nth pixel within each depth map. Increase them for a lighter point cloud, decrease (down to 1) for a denser one.
