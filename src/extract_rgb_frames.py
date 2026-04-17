import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path) -> int:
    """Extract PNG frames from an mp4 video. Returns the number of frames written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{frame_idx:06d}.png"), frame)
        frame_idx += 1
    cap.release()
    return frame_idx


def main():
    parser = argparse.ArgumentParser(
        description="Extract RGB frames from a Stray Scanner mp4 video into individual PNG images."
    )
    parser.add_argument("input", type=Path, help="Path to input mp4 video (e.g. data/rgb.mp4)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory for PNG frames (default: <input_data_dir>/images/)")
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_dir = args.output.resolve() if args.output else input_path.parent / "images"

    count = extract_frames(input_path, out_dir)
    print(f"Extracted {count} frames to {out_dir}/")


if __name__ == "__main__":
    main()
