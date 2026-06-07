#!/usr/bin/env python3
"""Export Camera blocks from 3DGS training cameras.json to PBRT snippets."""

import argparse
import json
import math
from pathlib import Path


def fov_from_focal(fx, width):
    return math.degrees(2 * math.atan(width / (2 * fx)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cameras_json")
    p.add_argument("--out_dir", default="cameras")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=800)
    args = p.parse_args()

    data = json.loads(Path(args.cameras_json).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, cam in enumerate(data):
        if cam.get("type") != "perspective":
            continue
        w2c = cam["world_to_camera"]
        # COLMAP / 3DGS: extract position and forward
        R = [w2c[0][:3], w2c[1][:3], w2c[2][:3]]
        t = [w2c[0][3], w2c[1][3], w2c[2][3]]
        eye = [-sum(R[j][k] * t[k] for k in range(3)) for j in range(3)]
        forward = [-R[2][0], -R[2][1], -R[2][2]]
        up = [R[1][0], R[1][1], R[1][2]]
        look = [eye[j] + forward[j] for j in range(3)]
        fx = cam.get("fx", cam.get("fl_x", 1000))
        fov = fov_from_focal(fx, args.width)

        snippet = f'''LookAt {eye[0]:.6f} {eye[1]:.6f} {eye[2]:.6f} \\
     {look[0]:.6f} {look[1]:.6f} {look[2]:.6f} \\
     {up[0]:.6f} {up[1]:.6f} {up[2]:.6f}
Camera "perspective"
    "float fov" [{fov:.4f}]
'''
        (out_dir / f"camera_{i:03d}.pbrt").write_text(snippet)
        print(f"Wrote camera_{i:03d}.pbrt")

    print(f"Exported to {out_dir}/")


if __name__ == "__main__":
    main()
