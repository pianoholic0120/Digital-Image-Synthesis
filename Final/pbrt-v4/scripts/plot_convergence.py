#!/usr/bin/env python3
"""Plot PSNR vs SPP from JSON metric files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+", help="JSON files from eval_psnr.py")
    p.add_argument("--out", default="psnr_vs_spp.pdf")
    p.add_argument("--labels", nargs="*", help="Optional series labels (one per metrics group)")
    args = p.parse_args()

    plt.figure(figsize=(6, 4))
    if args.labels:
        # Group metrics by label prefix in filename or explicit groups via ';' in path
        groups = {}
        for path in args.metrics:
            label = Path(path).parent.name if Path(path).parent.name not in (".", "") else "series"
            groups.setdefault(label, []).append(path)
        for label, paths in groups.items():
            spps, psnrs = [], []
            for path in sorted(paths):
                data = json.loads(Path(path).read_text())
                spp = int(Path(path).stem.split("_")[-1]) if "_" in path else len(spps) + 1
                spps.append(spp)
                psnrs.append(data["psnr"])
            plt.semilogx(spps, psnrs, "o-", label=label)
        plt.legend()
    else:
        spps, psnrs = [], []
        for path in sorted(args.metrics):
            data = json.loads(Path(path).read_text())
            spp = int(Path(path).stem.split("_")[-1]) if "_" in path else len(spps) + 1
            spps.append(spp)
            psnrs.append(data["psnr"])
        plt.semilogx(spps, psnrs, "o-")
    plt.xlabel("SPP")
    plt.ylabel("PSNR (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
