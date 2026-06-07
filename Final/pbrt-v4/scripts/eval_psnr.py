#!/usr/bin/env python3
"""Compute PSNR / SSIM between rendered EXR and reference PNG."""

import argparse
import json
import numpy as np

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio


def load_rgb(path):
    img = iio.imread(path).astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3]


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def ssim_simple(a, b):
    # Lightweight SSIM approximation (no external deps)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", default="metrics.json")
    args = p.parse_args()

    pred = load_rgb(args.pred)
    ref = load_rgb(args.ref)
    h, w = min(pred.shape[0], ref.shape[0]), min(pred.shape[1], ref.shape[1])
    pred, ref = pred[:h, :w], ref[:h, :w]

    metrics = {
        "psnr": psnr(pred, ref),
        "ssim": ssim_simple(pred, ref),
    }
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
