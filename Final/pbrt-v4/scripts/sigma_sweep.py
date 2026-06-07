#!/usr/bin/env python3
"""Sigma cutoff ablation on lego scene."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scenes/lego_3dgs.pbrt")
    p.add_argument("--sigmas", nargs="+", type=float, required=True)
    p.add_argument("--ref", default="gt/lego.png")
    p.add_argument("--pbrt", default="build/Release/pbrt.exe")
    p.add_argument("--out", default="sigma_sweep.csv")
    args = p.parse_args()

    template = Path(args.scene).read_text()
    eval_script = Path(__file__).with_name("eval_psnr.py")
    rows = []

    for sigma in args.sigmas:
        text = template.replace('"float sigma_cutoff" [2.828]',
                                f'"float sigma_cutoff" [{sigma}]')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scene = tmp / f"sigma_{sigma}.pbrt"
            out_exr = tmp / f"sigma_{sigma}.exr"
            scene.write_text(text)
            subprocess.run([args.pbrt, str(scene), f"--outfile={out_exr}"], check=True)

            row = {"sigma": sigma}
            if Path(args.ref).exists():
                mj = tmp / "m.json"
                subprocess.run(
                    [sys.executable, str(eval_script), "--pred", str(out_exr),
                     "--ref", args.ref, "--out", str(mj)],
                    check=True,
                )
                row.update(json.loads(mj.read_text()))
            rows.append(row)
            print(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
