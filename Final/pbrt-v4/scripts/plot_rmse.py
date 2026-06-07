#!/usr/bin/env python3
"""Plot RMSE vs SPP from eval_psnr JSON files."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def psnr_to_rmse(psnr):
    if math.isinf(psnr):
        return 0.0
    return 10 ** (-psnr / 20.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+")
    p.add_argument("--out", default="rmse_vs_spp.pdf")
    p.add_argument("--label", default="CPU")
    args = p.parse_args()

    spps, rmses = [], []
    for path in sorted(args.metrics):
        data = json.loads(Path(path).read_text())
        spp = int(Path(path).stem.split("_")[-1]) if "_" in path else len(spps) + 1
        spps.append(spp)
        rmses.append(psnr_to_rmse(data["psnr"]))

    plt.figure(figsize=(6, 4))
    plt.loglog(spps, rmses, "o-", label=args.label)
    plt.xlabel("SPP")
    plt.ylabel("RMSE")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
