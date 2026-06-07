#!/usr/bin/env python3
"""Multi-sample N sweep: render at different internal sample counts, report PSNR."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Base .pbrt scene path")
    p.add_argument("--N", nargs="+", type=int, required=True, help="SPP values to sweep")
    p.add_argument("--ref", help="Reference image for PSNR")
    p.add_argument("--pbrt", default="build/Release/pbrt.exe")
    p.add_argument("--out", default="multisample_sweep.csv")
    args = p.parse_args()

    scene_path = Path(args.scene)
    template = scene_path.read_text()
    eval_script = Path(__file__).with_name("eval_psnr.py")
    rows = []

    for n in args.N:
        text = template
        import re
        text = re.sub(r'"integer pixelsamples"\s*\[\d+\]', f'"integer pixelsamples" [{n}]', text)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scene = tmp / f"scene_spp{n}.pbrt"
            out_exr = tmp / f"render_spp{n}.exr"
            scene.write_text(text)

            t0 = __import__("time").perf_counter()
            subprocess.run([args.pbrt, str(scene), f"--outfile={out_exr}"], check=True)
            elapsed = __import__("time").perf_counter() - t0

            row = {"N": n, "seconds": elapsed, "render": str(out_exr)}
            if args.ref and Path(args.ref).exists():
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
