#!/usr/bin/env python3
"""Convert 3DGS training cameras.json entry to pbrt LookAt + perspective fov."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def camera_to_transform(cam: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the exact PBRT Transform matrix for a 3DGS camera.

    cameras.json stores rotation as the camera-to-world (C2W) rotation matrix R.
    The 3DGS/OpenCV camera axes in world space are the COLUMNS of R:
      - Camera X (right)   = R[:, 0]
      - Camera Y (down)    = R[:, 1]  (OpenCV Y is down)
      - Camera Z (forward) = R[:, 2]

    PBRT's perspective camera generates rays in camera space where:
      - camera_x = (px - cx) / fx  (positive = right in image)
      - camera_y = -(py - cy) / fy  (positive = UP, PBRT flips raster y)

    So the desired worldFromCamera matrix maps:
      camera X  ->  R[:, 0]   (right)
      camera Y  -> -R[:, 1]   (PBRT cam-Y is up; 3DGS Y is down)
      camera Z  ->  R[:, 2]   (forward)
      origin    ->  cam pos

      worldFromCamera (wfc) =
        [[R[0,0], -R[0,1], R[0,2], pos[0]],
         [R[1,0], -R[1,1], R[1,2], pos[1]],
         [R[2,0], -R[2,1], R[2,2], pos[2]],
         [0,       0,       0,       1   ]]

    PBRT's scene code (scene.cpp BasicSceneBuilder::Camera) treats the current
    transformation matrix (CTM) as cameraFromWorld:
        cameraFromWorld = ctm
        worldFromCamera = Inverse(ctm)

    BasicSceneBuilder::Transform stores the CTM as Transpose(M_input) where
    M_input is the 16 row-major floats we write in the scene file.
    Therefore:
        CTM = Transpose(M_input)

    We need:  Inverse(CTM) = wfc
    =>        CTM = cfw  (where cfw = Inverse(wfc))
    =>        Transpose(M_input) = cfw
    =>        M_input = Transpose(cfw) = cfw^T

    So we write cfw^T = Transpose(Inverse(wfc)) in the Transform directive.
    """
    R = np.array(cam["rotation"], dtype=np.float64)
    pos = np.array(cam["position"], dtype=np.float64)
    height = cam.get("height", 800)
    fy = cam.get("fy", cam.get("fl_y", cam.get("fx", cam.get("fl_x", 1000.0))))
    fov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

    # Construct worldFromCamera (wfc)
    wfc = np.array([
        [R[0, 0], -R[0, 1], R[0, 2], pos[0]],
        [R[1, 0], -R[1, 1], R[1, 2], pos[1]],
        [R[2, 0], -R[2, 1], R[2, 2], pos[2]],
        [0.0,      0.0,      0.0,     1.0   ],
    ], dtype=np.float64)

    # mat_T = cfw^T = Transpose(Inverse(wfc))
    # PBRT will internally compute CTM = Transpose(mat_T) = cfw,
    # and then worldFromCamera = Inverse(CTM) = Inverse(cfw) = wfc.
    cfw = np.linalg.inv(wfc)
    mat_T = cfw.T
    return mat_T, pos, fov


def format_transform_block(mat_T: np.ndarray, fov: float) -> str:
    """Format a PBRT Transform + Camera block from the (pre-transposed) matrix."""
    flat = " ".join(f"{v:.9f}" for v in mat_T.flatten())
    return (
        f"Transform [{flat}]\n"
        f'Camera "perspective"\n    "float fov" [{fov}]'
    )


# ---------------------------------------------------------------------------
# Backwards-compatible shims for scripts that call camera_to_lookat /
# format_lookat_block.  They now route through the correct Transform path.
# ---------------------------------------------------------------------------

def camera_to_lookat(cam: dict):
    """Return (eye, target, up, fov) using the Transform-based camera.

    NOTE: eye/target/up are kept for API compatibility but the real camera
    orientation is encoded in format_lookat_block via format_transform_block.
    Scripts that call camera_to_lookat + format_lookat_block get the correct
    Transform automatically; scripts that manually build LookAt strings from
    eye/target/up will still produce a mirrored image.
    """
    mat_T, pos, fov = camera_to_transform(cam)
    # Reconstruct eye/target/up from the (now correct) matrix for compatibility.
    R = np.array(cam["rotation"], dtype=np.float64)
    eye = pos
    target = pos + R[:, 2]
    up = -R[:, 1]
    return eye, target, up, fov


def format_lookat_block(eye, target, up, fov, cam: dict | None = None) -> str:
    """Return a PBRT Transform+Camera block (ignores eye/target/up, uses cam).

    For full correctness, pass the original `cam` dict.  If not provided,
    falls back to the old LookAt string (will be horizontally mirrored for
    cameras whose R[:,0] has negative x).
    """
    if cam is not None:
        mat_T, _, fov2 = camera_to_transform(cam)
        return format_transform_block(mat_T, fov2)
    # Legacy fallback – known to be mirrored; kept only for callers that don't
    # pass the cam dict and where the result is not written to a PBRT file.
    return (
        f"LookAt {eye[0]:.9f} {eye[1]:.9f} {eye[2]:.9f}  "
        f"{target[0]:.9f} {target[1]:.9f} {target[2]:.9f}  "
        f"{up[0]:.9f} {up[1]:.9f} {up[2]:.9f}\n"
        f'Camera "perspective"\n    "float fov" [{fov}]'
    )


def format_gaussiancloud_camera_params(cam: dict) -> str:
    """Return the PBRT shape parameter lines for 2D Gaussian projection.

    These are appended to the Shape "gaussiancloud" block so that
    IntersectComposite uses the rasterizer-equivalent 2D Mahalanobis distance.

    cameras.json rotation is stored row-major: R[i] = i-th row of the C2W matrix.
    Camera axes in world space = COLUMNS of R:
        rx = R[:,0]  (camera right,    3DGS X)
        ry = R[:,1]  (camera down,     3DGS Y)
        rz = R[:,2]  (camera forward,  3DGS Z)
    """
    R = np.array(cam["rotation"], dtype=np.float64)
    pos = np.array(cam["position"], dtype=np.float64)
    fx = float(cam.get("fx", cam.get("fl_x", 1000.0)))
    fy = float(cam.get("fy", cam.get("fl_y", fx)))
    width  = float(cam.get("width",  800))
    height = float(cam.get("height", 800))
    cx = float(cam.get("cx", width  / 2.0))
    cy = float(cam.get("cy", height / 2.0))
    rx = R[:, 0]
    ry = R[:, 1]
    rz = R[:, 2]
    return (
        f'        "float cam_fx" [{fx:.6f}]\n'
        f'        "float cam_fy" [{fy:.6f}]\n'
        f'        "float cam_cx" [{cx:.6f}]\n'
        f'        "float cam_cy" [{cy:.6f}]\n'
        f'        "float cam_rx" [{rx[0]:.9f} {rx[1]:.9f} {rx[2]:.9f}]\n'
        f'        "float cam_ry" [{ry[0]:.9f} {ry[1]:.9f} {ry[2]:.9f}]\n'
        f'        "float cam_rz" [{rz[0]:.9f} {rz[1]:.9f} {rz[2]:.9f}]\n'
        f'        "float cam_pos" [{pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}]\n'
        f'        "integer cam_width" [{int(width)}]\n'
        f'        "integer cam_height" [{int(height)}]'
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cameras_json", type=Path)
    p.add_argument("--frame", type=int, default=0)
    args = p.parse_args()

    data = json.loads(args.cameras_json.read_text())
    if args.frame < 0 or args.frame >= len(data):
        raise SystemExit(f"frame {args.frame} out of range (0..{len(data) - 1})")

    cam = data[args.frame]
    mat_T, pos, fov = camera_to_transform(cam)
    print(f"# frame {args.frame} img_name={cam.get('img_name', '?')}")
    print(format_transform_block(mat_T, fov))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
