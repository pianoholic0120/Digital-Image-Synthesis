#!/usr/bin/env python3
"""Alpha-composite Gaussian Splatting renders for clean backgrounds (NeRF Synthetic)."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Allow running from repo root or scripts/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from argparse import Namespace
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def _load_cfg(model_path: str) -> Namespace:
    cfg_path = os.path.join(model_path, "cfg_args")
    with open(cfg_path) as f:
        return eval(f.read())


def _make_namespace(cfg: Namespace) -> Namespace:
    keys = [
        "sh_degree", "source_path", "model_path", "images", "depths",
        "resolution", "white_background", "train_test_exp", "data_device", "eval",
    ]
    return Namespace(**{k: getattr(cfg, k) for k in keys})


def composite_view(render_img: torch.Tensor, gt_img: torch.Tensor,
                   alpha: torch.Tensor, bg: torch.Tensor) -> tuple:
    bg_view = bg.view(3, 1, 1)
    render_c = render_img * alpha + bg_view * (1.0 - alpha)
    gt_c = gt_img * alpha + bg_view * (1.0 - alpha)
    return render_c, gt_c


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    arr = torch.clamp(tensor, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def process_split(views, gaussians, pipe, bg, out_dir: str, split_name: str):
    render_dir = os.path.join(out_dir, "renders")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    psnrs, ssims, l1s = [], [], []

    for idx, view in enumerate(tqdm(views, desc=f"Compositing {split_name}")):
        with torch.no_grad():
            img = render(view, gaussians, pipe, bg)["render"]
        gt = view.original_image[:3].cuda()
        alpha = view.alpha_mask.cuda() if view.alpha_mask is not None else torch.ones_like(gt[:1])

        render_c, gt_c = composite_view(img, gt, alpha, bg)

        fname = f"{idx:05d}.png"
        save_tensor_image(render_c, os.path.join(render_dir, fname))
        save_tensor_image(gt_c, os.path.join(gt_dir, fname))

        psnrs.append(psnr(render_c, gt_c).mean().item())
        l1s.append(l1_loss(render_c, gt_c).item())
        ssims.append(ssim(render_c.unsqueeze(0), gt_c.unsqueeze(0)).item())

    metrics = {
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "l1_mean": float(np.mean(l1s)),
        "num_views": len(views),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Alpha-composite renders for clean backgrounds.")
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--antialiasing", action="store_true", default=True)
    parser.add_argument("--no-antialiasing", dest="antialiasing", action="store_false")
    args = parser.parse_args()

    cfg = _load_cfg(args.model_path)
    lp = _make_namespace(cfg)
    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=args.antialiasing,
    )

    gaussians = GaussianModel(cfg.sh_degree, "default")
    scene = Scene(lp, gaussians, load_iteration=args.iteration, shuffle=False)

    if cfg.white_background:
        bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    else:
        bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    all_metrics = {"iteration": args.iteration, "white_background": cfg.white_background}

    if not args.skip_train:
        train_views = scene.getTrainCameras()
        if len(train_views) > 0:
            out = os.path.join(
                args.model_path, "train", f"ours_{args.iteration}", "composited"
            )
            all_metrics["train"] = process_split(
                train_views, gaussians, pipe, bg, out, "train"
            )

    if not args.skip_test:
        test_views = scene.getTestCameras()
        if len(test_views) > 0:
            out = os.path.join(
                args.model_path, "test", f"ours_{args.iteration}", "composited"
            )
            all_metrics["test"] = process_split(
                test_views, gaussians, pipe, bg, out, "test"
            )

    metrics_path = os.path.join(args.model_path, f"metrics_composited_{args.iteration}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nComposited metrics saved to {metrics_path}")
    for split in ("test", "train"):
        if split in all_metrics:
            m = all_metrics[split]
            print(
                f"  [{split}] PSNR={m['psnr_mean']:.2f}  "
                f"SSIM={m['ssim_mean']:.4f}  L1={m['l1_mean']:.4f}  "
                f"({m['num_views']} views)"
            )


if __name__ == "__main__":
    main()
