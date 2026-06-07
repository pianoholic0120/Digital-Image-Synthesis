#!/usr/bin/env python3
"""Benchmark render time for 3DGS scenes at fixed SPP."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pbrt", default="build/Release/pbrt.exe")
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--spp", type=int, default=64)
    p.add_argument("--out", default="benchmark.json")
    args = p.parse_args()

    results = []
    for scene in args.scenes:
        scene_path = Path(scene)
        t0 = time.perf_counter()
        subprocess.run([args.pbrt, str(scene_path), f"--pixelsamples={args.spp}"], check=True)
        elapsed = time.perf_counter() - t0
        results.append({"scene": str(scene_path), "spp": args.spp, "seconds": elapsed})
        print(f"{scene_path.name}: {elapsed:.2f}s")

    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
