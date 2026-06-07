#!/usr/bin/env python3
"""Render time vs Gaussian count from perf_benchmark JSON."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", help="JSON from perf_benchmark.py or CSV with gaussians column")
    p.add_argument("--out", default="time_vs_gaussians.pdf")
    args = p.parse_args()

    path = Path(args.benchmark)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        rows = list(csv.DictReader(path.open()))

    gaussians = [float(r.get("gaussians", r.get("num_gaussians", i + 1))) for i, r in enumerate(rows)]
    times = [float(r["seconds"]) for r in rows]

    plt.figure(figsize=(6, 4))
    plt.loglog(gaussians, times, "o-")
    plt.xlabel("# Gaussians")
    plt.ylabel("Render time (s)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
