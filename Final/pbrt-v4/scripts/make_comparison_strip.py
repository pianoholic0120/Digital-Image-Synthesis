#!/usr/bin/env python3
"""Horizontal strip of renders at increasing SPP with labels."""

import argparse
from pathlib import Path

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio

from PIL import Image, ImageDraw, ImageFont


def load_rgb(path):
    img = iio.imread(path).astype("float32")
    if img.max() > 1.5:
        img /= 255.0
    return (img[..., :3].clip(0, 1) * 255).astype("uint8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("renders", nargs="+", help="EXR/PNG paths sorted by increasing SPP")
    p.add_argument("--labels", nargs="*", help="Optional labels (default: filenames)")
    p.add_argument("--out", default="comparison_strip.png")
    args = p.parse_args()

    imgs = [load_rgb(r) for r in args.renders]
    labels = args.labels or [Path(r).stem for r in args.renders]
    h = min(im.shape[0] for im in imgs)
    imgs = [im[:h] for im in imgs]

    gap = 8
    label_h = 24
    w = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h + label_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for im, label in zip(imgs, labels):
        canvas.paste(Image.fromarray(im), (x, label_h))
        draw.text((x + 4, 4), label, fill=(220, 220, 220))
        x += im.shape[1] + gap

    canvas.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
