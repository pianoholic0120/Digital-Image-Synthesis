#!/usr/bin/env python3
"""Bar chart of PSNR per ablation variant."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="ablation_results.csv")
    p.add_argument("--out", default="ablations_psnr.pdf")
    args = p.parse_args()

    rows = list(csv.DictReader(Path(args.csv).open()))
    names = [r["variant"] for r in rows if "psnr" in r]
    psnrs = [float(r["psnr"]) for r in rows if "psnr" in r]

    plt.figure(figsize=(8, 4))
    plt.bar(names, psnrs)
    plt.ylabel("PSNR (dB)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
