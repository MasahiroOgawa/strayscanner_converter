"""Convert Stray Scanner data to COLMAP format for 3D Gaussian Splatting."""

import argparse
import csv
import struct
from pathlib import Path

import cv2
import numpy as np

# COLMAP camera model IDs
PINHOLE_MODEL_ID = 1


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


def read_odometry(csv_path):
    """Read odometry CSV and return list of frame dicts."""
    frames = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        for row in reader:
            d = {header[i]: row[i].strip() for i in range(len(row))}
            frames.append({
                "frame": d["frame"],
                "x": float(d["x"]), "y": float(d["y"]), "z": float(d["z"]),
                "qx": float(d["qx"]), "qy": float(d["qy"]),
                "qz": float(d["qz"]), "qw": float(d["qw"]),
                "fx": float(d["fx"]), "fy": float(d["fy"]),
                "cx": float(d["cx"]), "cy": float(d["cy"]),
            })
    return frames


def write_cameras_bin(path, frames, width, height):
    """Write cameras.bin with one PINHOLE camera per frame."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for i, fr in enumerate(frames):
            camera_id = i + 1
            f.write(struct.pack("<iI", camera_id, PINHOLE_MODEL_ID))
            f.write(struct.pack("<QQ", width, height))
            # PINHOLE params: fx, fy, cx, cy
            f.write(struct.pack("<4d", fr["fx"], fr["fy"], fr["cx"], fr["cy"]))


def write_images_bin(path, frames):
    """Write images.bin with extrinsics per frame.

    Odometry gives camera-to-world (t_wc, R_wc).
    COLMAP expects world-to-camera (t_cw, R_cw).
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(frames)))
        for i, fr in enumerate(frames):
            image_id = i + 1
            camera_id = i + 1

            # Camera-to-world rotation
            R_wc = quaternion_to_rotation_matrix(fr["qw"], fr["qx"], fr["qy"], fr["qz"])
            t_wc = np.array([fr["x"], fr["y"], fr["z"]])

            # Invert to world-to-camera
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc

            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_cw)
            tx, ty, tz = t_cw

            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", qw, qx, qy, qz))
            f.write(struct.pack("<3d", tx, ty, tz))
            f.write(struct.pack("<I", camera_id))

            # Image name (null-terminated)
            name = f"{fr['frame']}.png"
            f.write(name.encode("utf-8") + b"\x00")

            # No 2D point observations
            f.write(struct.pack("<Q", 0))


def generate_points3d(data_dir, frames, subsample_frames=10, subsample_pixels=8):
    """Generate 3D points by unprojecting depth maps.

    Args:
        data_dir: Path to stray scanner data directory.
        frames: List of frame dicts from odometry.
        subsample_frames: Use every Nth frame.
        subsample_pixels: Use every Nth pixel in depth map.
    """
    depth_dir = data_dir / "depth"
    image_dir = data_dir / "image"
    points = []
    colors = []

    # Depth images are 256x192, RGB images are 1920x1440
    # Scale factor between them
    sample_depth = cv2.imread(str(depth_dir / f"{frames[0]['frame']}.png"), cv2.IMREAD_UNCHANGED)
    if sample_depth is None:
        print("Warning: no depth images found, writing empty points3D.bin")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    depth_h, depth_w = sample_depth.shape[:2]

    for idx in range(0, len(frames), subsample_frames):
        fr = frames[idx]
        depth_path = depth_dir / f"{fr['frame']}.png"
        image_path = image_dir / f"{fr['frame']}.png"

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(image_path))
        if depth is None or rgb is None:
            continue

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # Resize RGB to depth resolution for color lookup
        rgb_small = cv2.resize(rgb, (depth_w, depth_h))

        # Depth intrinsics scaled from image intrinsics
        img_h, img_w = 1440, 1920
        sx, sy = depth_w / img_w, depth_h / img_h
        fx_d, fy_d = fr["fx"] * sx, fr["fy"] * sy
        cx_d, cy_d = fr["cx"] * sx, fr["cy"] * sy

        # Camera-to-world transform
        R_wc = quaternion_to_rotation_matrix(fr["qw"], fr["qx"], fr["qy"], fr["qz"])
        t_wc = np.array([fr["x"], fr["y"], fr["z"]])

        # Subsample pixels
        vs, us = np.mgrid[0:depth_h:subsample_pixels, 0:depth_w:subsample_pixels]
        vs, us = vs.ravel(), us.ravel()
        z_vals = depth[vs, us].astype(np.float32) / 1000.0  # mm to meters

        # Filter invalid depth
        valid = z_vals > 0.01
        vs, us, z_vals = vs[valid], us[valid], z_vals[valid]

        # Unproject to camera coordinates
        x_cam = (us - cx_d) / fx_d * z_vals
        y_cam = (vs - cy_d) / fy_d * z_vals
        pts_cam = np.stack([x_cam, y_cam, z_vals], axis=1)

        # Transform to world
        pts_world = (R_wc @ pts_cam.T).T + t_wc
        point_colors = rgb_small[vs, us]

        points.append(pts_world)
        colors.append(point_colors)

    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(points), np.concatenate(colors).astype(np.uint8)


def write_points3d_bin(path, points, colors):
    """Write points3D.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i in range(len(points)):
            point3d_id = i + 1
            x, y, z = points[i]
            r, g, b = colors[i]
            error = 0.0
            track_length = 0
            f.write(struct.pack("<Q", point3d_id))
            f.write(struct.pack("<3d", x, y, z))
            f.write(struct.pack("<3B", int(r), int(g), int(b)))
            f.write(struct.pack("<d", error))
            f.write(struct.pack("<Q", track_length))


def main():
    parser = argparse.ArgumentParser(description="Convert Stray Scanner data to COLMAP format")
    parser.add_argument("input_dir", type=Path, help="Path to Stray Scanner data directory")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None,
                        help="Path to output COLMAP dataset directory (default: <input_dir>/images/)")
    parser.add_argument("--subsample-frames", type=int, default=10,
                        help="Use every Nth frame for point cloud (default: 10)")
    parser.add_argument("--subsample-pixels", type=int, default=8,
                        help="Use every Nth pixel in depth for point cloud (default: 8)")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir / "images"

    print(f"Reading odometry from {args.input_dir}")
    frames = read_odometry(args.input_dir / "odometry.csv")

    # Filter to frames that have corresponding images
    image_dir = args.input_dir / "image"
    frames = [f for f in frames if (image_dir / f"{f['frame']}.png").exists()]
    print(f"Found {len(frames)} frames with images")

    # Create output directories
    images_out = args.output_dir / "images"
    sparse_out = args.output_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    # Copy/symlink images
    print("Symlinking images...")
    for fr in frames:
        src = (image_dir / f"{fr['frame']}.png").resolve()
        dst = images_out / f"{fr['frame']}.png"
        if not dst.exists():
            dst.symlink_to(src)

    # Get image dimensions
    sample_img = cv2.imread(str(image_dir / f"{frames[0]['frame']}.png"))
    img_h, img_w = sample_img.shape[:2]

    # Write cameras.bin
    print("Writing cameras.bin...")
    write_cameras_bin(sparse_out / "cameras.bin", frames, img_w, img_h)

    # Write images.bin
    print("Writing images.bin...")
    write_images_bin(sparse_out / "images.bin", frames)

    # Generate and write points3D.bin
    print("Generating 3D points from depth maps...")
    points, colors = generate_points3d(
        args.input_dir, frames,
        subsample_frames=args.subsample_frames,
        subsample_pixels=args.subsample_pixels,
    )
    print(f"Generated {len(points)} 3D points")
    write_points3d_bin(sparse_out / "points3D.bin", points, colors)

    print(f"\nCOLMAP dataset written to {args.output_dir}")
    print(f"  {images_out}/ ({len(frames)} images)")
    print(f"  {sparse_out}/cameras.bin")
    print(f"  {sparse_out}/images.bin")
    print(f"  {sparse_out}/points3D.bin")


if __name__ == "__main__":
    main()
