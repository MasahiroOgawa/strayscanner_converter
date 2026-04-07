import argparse
import cv2
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Extract RGB frames from a Stray Scanner mp4 video into individual PNG images."
)
parser.add_argument("input", type=Path, help="Path to input mp4 video (e.g. data/rgb.mp4)")
parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory for PNG frames (default: output/<input_data_name>/)")
args = parser.parse_args()

input_path = args.input.resolve()
if args.output:
    out_dir = args.output.resolve()
else:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "output" / input_path.parent.name
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(input_path))
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(str(out_dir / f"{frame_idx:06d}.png"), frame)
    frame_idx += 1

cap.release()
print(f"Extracted {frame_idx} frames to {out_dir}/")
