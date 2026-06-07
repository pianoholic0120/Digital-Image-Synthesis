#!/usr/bin/env python3
"""Run ablation variants defined in YAML, render with PBRT, collect metrics."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def render_scene(pbrt, scene_path, out_exr, extra_args):
    cmd = [str(pbrt), str(scene_path), f'--outfile={out_exr}'] + list(extra_args)
    subprocess.run(cmd, check=True)


def patch_scene(template: str, overrides: dict) -> str:
    text = template
    for key, value in overrides.items():
        if key == "sigma_cutoff":
            text = text.replace('"float sigma_cutoff" [2.828]',
                                f'"float sigma_cutoff" [{value}]')
        elif key == "use_center_depth":
            text = text.replace('"bool use_center_depth" [true]',
                                f'"bool use_center_depth" [{"true" if value else "false"}]')
            text = text.replace('"bool use_center_depth" [false]',
                                f'"bool use_center_depth" [{"true" if value else "false"}]')
        elif key == "internal_accel":
            text = text.replace('"string internal_accel" ["bvh"]',
                                f'"string internal_accel" ["{value}"]')
            text = text.replace('"string internal_accel" ["kdtree"]',
                                f'"string internal_accel" ["{value}"]')
        elif key == "pixelsamples":
            text = text.replace('"integer pixelsamples" [64]',
                                f'"integer pixelsamples" [{int(value)}]')
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--pbrt", default="build/Release/pbrt.exe")
    p.add_argument("--out", default="ablation_results.csv")
    args = p.parse_args()

    if yaml is None:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(Path(args.config).read_text())
    base_scene = Path(cfg["scene"])
    template = base_scene.read_text()
    eval_script = Path(__file__).with_name("eval_psnr.py")
    ref = cfg.get("reference")
    rows = []

    for variant in cfg.get("variants", []):
        name = variant["name"]
        overrides = variant.get("overrides", {})
        pbrt_args = variant.get("pbrt_args", [])

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scene = tmp / f"{name}.pbrt"
            out_exr = tmp / f"{name}.exr"
            scene.write_text(patch_scene(template, overrides))

            render_scene(args.pbrt, scene, out_exr, pbrt_args)

            row = {"variant": name, "render": str(out_exr)}
            if ref and Path(ref).exists():
                metrics_json = tmp / f"{name}.json"
                subprocess.run(
                    [sys.executable, str(eval_script), "--pred", str(out_exr),
                     "--ref", ref, "--out", str(metrics_json)],
                    check=True,
                )
                row.update(json.loads(metrics_json.read_text()))
            rows.append(row)
            print(name, row)

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
