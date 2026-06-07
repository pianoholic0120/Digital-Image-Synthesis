#!/usr/bin/env python3
"""Cornell GI comparison figure with annotated regions."""

import argparse
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


def annotate(img, boxes, labels):
    out = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(out)
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        draw.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=2)
        draw.text((x0 + 2, y0 + 2), label, fill=(255, 220, 0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pbrt", required=True)
    p.add_argument("--rasterizer", required=True)
    p.add_argument("--out", default="gi_comparison.png")
    args = p.parse_args()

    a = annotate(load_rgb(args.rasterizer), [(20, 20, 180, 180), (220, 80, 380, 240)],
                 ["ref: color bleed", "ref: shadow"])
    b = annotate(load_rgb(args.pbrt), [(20, 20, 180, 180), (220, 80, 380, 240)],
                 ["ours: GI", "ours: soft shadow"])

    h = min(a.height, b.height)
    gap = 6
    canvas = Image.new("RGB", (a.width + b.width + gap, h + 24), (16, 16, 16))
    canvas.paste(a.crop((0, 0, a.width, h)), (0, 24))
    canvas.paste(b.crop((0, 0, b.width, h)), (a.width + gap, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "Rasterizer reference", fill=(220, 220, 220))
    draw.text((a.width + gap + 4, 4), "PBRT+ path trace", fill=(220, 220, 220))
    canvas.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
