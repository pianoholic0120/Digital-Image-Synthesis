#!/usr/bin/env python3
"""Batch PSNR/SSIM over paired pred/ref directories."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--ref_dir", required=True)
    p.add_argument("--out", default="batch_metrics.csv")
    p.add_argument("--eval_script", default=str(Path(__file__).with_name("eval_psnr.py")))
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)
    rows = []

    for pred in sorted(pred_dir.glob("*")):
        if not pred.is_file():
            continue
        ref = ref_dir / pred.name
        if not ref.exists():
            ref = ref_dir / (pred.stem + ".png")
        if not ref.exists():
            print(f"skip (no ref): {pred.name}", file=sys.stderr)
            continue

        out_json = pred.with_suffix(".metrics.json")
        subprocess.run(
            [sys.executable, args.eval_script, "--pred", str(pred), "--ref", str(ref),
             "--out", str(out_json)],
            check=True,
        )
        metrics = json.loads(out_json.read_text())
        rows.append({"image": pred.name, **metrics})

    if not rows:
        print("No pairs evaluated.", file=sys.stderr)
        return 1

    for key in ("psnr", "ssim"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            rows.append({"image": f"__{key}_mean__", key: mean})
            rows.append({"image": f"__{key}_std__", key: var ** 0.5})

    fieldnames = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
