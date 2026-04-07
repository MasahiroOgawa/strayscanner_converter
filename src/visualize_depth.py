import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Visualize Stray Scanner depth, confidence, and RGB data in a 2x2 grid."
)
parser.add_argument("input", type=Path, help="Path to Stray Scanner data directory (containing depth/, confidence/)")
parser.add_argument("--rgb-dir", type=Path, default=None, help="Path to extracted RGB frames directory (default: input/rgb_frames/)")
parser.add_argument("--frame", type=int, default=0, help="Frame number to visualize (default: 0)")
parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path (default: output/<input_data_name>/depth_vis_NNNNNN.png)")
args = parser.parse_args()

data = args.input.resolve()
rgb_dir = args.rgb_dir.resolve() if args.rgb_dir else data / "rgb_frames"
frame_id = f"{args.frame:06d}"

rgb_path = rgb_dir / f"{frame_id}.png"
depth_path = data / f"depth/{frame_id}.png"
conf_path = data / f"confidence/{frame_id}.png"

if not rgb_dir.exists():
    print(f"Error: {rgb_dir} not found. Run extract_rgb_frames.py first.")
    raise SystemExit(1)
for p in [rgb_path, depth_path, conf_path]:
    if not p.exists():
        print(f"Error: {p} not found.")
        raise SystemExit(1)

rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
confidence = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)

h, w = depth.shape
rgb_resized = cv2.resize(rgb, (w, h))

# Mask low-confidence depth
depth_masked = np.where(confidence > 0, depth, np.nan)

# Confidence colormap: 0=red, 1=yellow, 2=green
conf_color = np.zeros((h, w, 3), dtype=np.uint8)
conf_color[confidence == 0] = [200, 0, 0]
conf_color[confidence == 1] = [200, 200, 0]
conf_color[confidence == 2] = [0, 200, 0]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(f"Frame {frame_id}", fontsize=14)

axes[0, 0].imshow(rgb_resized)
axes[0, 0].set_title("RGB")

im = axes[0, 1].imshow(depth, cmap="turbo")
axes[0, 1].set_title("Depth (raw)")
fig.colorbar(im, ax=axes[0, 1], label="mm")

im2 = axes[1, 0].imshow(depth_masked, cmap="turbo")
axes[1, 0].set_title("Depth (confidence > 0)")
fig.colorbar(im2, ax=axes[1, 0], label="mm")

axes[1, 1].imshow(conf_color)
axes[1, 1].set_title("Confidence (R=0, Y=1, G=2)")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
if args.output:
    out = args.output.resolve()
else:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "output" / data.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"depth_vis_{frame_id}.png"
fig.savefig(str(out), dpi=150)
print(f"Saved to {out}")
