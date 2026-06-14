#!/usr/bin/env python3
"""Tonemap a pbrt RGB EXR to sRGB PNG for preview (IDE / Windows viewers)."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import OpenEXR
    import Imath
except ImportError as exc:
    raise SystemExit("pip install OpenEXR Imath numpy pillow") from exc


def read_exr(path: Path) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.HALF)
    channels = []
    for ch in "RGB":
        buf = exr.channel(ch, pt)
        channels.append(np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(h, w))
    return np.stack(channels, axis=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exr", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument(
        "--exposure", type=float, default=1.0, help="Linear scale before tonemap"
    )
    args = p.parse_args()
    out = args.output or args.exr.with_suffix(".png")

    rgb = read_exr(args.exr) * args.exposure
    linear = np.clip(rgb, 0.0, 1.0)
    Image.fromarray((linear * 255.0 + 0.5).astype(np.uint8)).save(out)
    print(f"Wrote {out} (peak linear {float(rgb.max()):.4f}, 3DGS-style save)")


if __name__ == "__main__":
    main()
