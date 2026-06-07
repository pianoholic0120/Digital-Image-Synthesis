#!/usr/bin/env python3
"""Side-by-side comparison with PSNR overlay."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio

from PIL import Image, ImageDraw


def load_rgb(path):
    img = iio.imread(path).astype("float32")
    if img.max() > 1.5:
        img /= 255.0
    return (img[..., :3].clip(0, 1) * 255).astype("uint8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rasterizer", required=True)
    p.add_argument("--pbrt", required=True, help="PBRT render (EXR/PNG)")
    p.add_argument("--out", default="side_by_side.png")
    args = p.parse_args()

    a = load_rgb(args.rasterizer)
    b = load_rgb(args.pbrt)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]

    eval_script = Path(__file__).with_name("eval_psnr.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        mj = f.name
    subprocess.run(
        [sys.executable, str(eval_script), "--pred", args.pbrt, "--ref", args.rasterizer,
         "--out", mj],
        check=True,
    )
    psnr = json.loads(Path(mj).read_text())["psnr"]

    gap = 4
    canvas = Image.new("RGB", (w * 2 + gap, h + 28), (20, 20, 20))
    canvas.paste(Image.fromarray(a), (0, 28))
    canvas.paste(Image.fromarray(b), (w + gap, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "Rasterizer", fill=(200, 200, 200))
    draw.text((w + gap + 4, 4), f"PBRT+ (PSNR {psnr:.2f} dB)", fill=(200, 200, 200))
    canvas.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
