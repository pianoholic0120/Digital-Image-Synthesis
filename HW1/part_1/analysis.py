#!/usr/bin/env python3
"""
Generate qualitative and quantitative analysis artifacts for Metropolis results.

Outputs are written to:
    part_1/analysis/
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SAMPLES = [1, 4, 8, 64, 256, 1024]
METHODS = {
    "method_1_uniform_init": "copied_images_1",
    "method_2_rejection_init": "copied_images_2",
}
INPUTS = ["jaguar", "lion", "tiger", "zebra"]


def load_gray_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / m))


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    # Lightweight SSIM approximation using global statistics.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    cov_ab = float(np.mean((a - mu_a) * (b - mu_b)))
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(numerator / denominator)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_qualitative_grid(
    output_path: Path,
    reference: np.ndarray,
    method_images: Dict[str, Dict[int, np.ndarray]],
    title: str,
) -> None:
    rows = 1 + len(method_images)
    cols = len(SAMPLES) + 1  # reference + each spp
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    axes[0, 0].imshow(reference, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Reference")
    axes[0, 0].axis("off")
    for c, spp in enumerate(SAMPLES, start=1):
        axes[0, c].axis("off")
        axes[0, c].set_title(f"{spp} spp")

    for r, (method_name, images_by_spp) in enumerate(method_images.items(), start=1):
        axes[r, 0].text(
            0.5,
            0.5,
            method_name.replace("_", "\n"),
            ha="center",
            va="center",
            fontsize=9,
        )
        axes[r, 0].axis("off")
        for c, spp in enumerate(SAMPLES, start=1):
            axes[r, c].imshow(images_by_spp[spp], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_error_map_grid(
    output_path: Path,
    reference: np.ndarray,
    method_images: Dict[str, Dict[int, np.ndarray]],
    title: str,
) -> None:
    rows = len(method_images)
    cols = len(SAMPLES)
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 2.8 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    vmax = 0.35
    for r, (method_name, images_by_spp) in enumerate(method_images.items()):
        for c, spp in enumerate(SAMPLES):
            err = np.abs(images_by_spp[spp] - reference)
            ax = axes[r, c]
            im = ax.imshow(err, cmap="magma", vmin=0, vmax=vmax)
            ax.axis("off")
            ax.set_title(f"{method_name}\n{spp} spp", fontsize=8)

    fig.suptitle(title, fontsize=14)
    # Reserve a dedicated colorbar axis to avoid overlapping subplots.
    fig.subplots_adjust(left=0.04, right=0.90, top=0.84, bottom=0.08, wspace=0.04, hspace=0.25)
    cbar_ax = fig.add_axes([0.915, 0.16, 0.015, 0.62])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Absolute error")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_line_plots(records: List[dict], out_dir: Path) -> None:
    method_styles = {
        "method_1_uniform_init": {
            "label": "method 1 uniform init",
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "s",
            "x_scale": 0.96,
        },
        "method_2_rejection_init": {
            "label": "method 2 rejection init",
            "color": "#ff7f0e",
            "linestyle": "-",
            "marker": "o",
            "x_scale": 1.04,
        },
    }

    for metric in ["mse", "mae", "psnr", "ssim"]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.ravel()
        for i, image_name in enumerate(INPUTS):
            ax = axes[i]
            for method in METHODS.keys():
                y = []
                for spp in SAMPLES:
                    for rec in records:
                        if (
                            rec["image"] == image_name
                            and rec["method"] == method
                            and rec["spp"] == spp
                        ):
                            y.append(rec[metric])
                            break
                style = method_styles[method]
                xvals = [s * style["x_scale"] for s in SAMPLES]
                ax.plot(
                    xvals,
                    y,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=5,
                    linewidth=1.8,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    alpha=0.95,
                )
            ax.set_xscale("log", base=2)
            ax.set_title(image_name)
            ax.set_xlabel("Samples per pixel")
            ax.set_ylabel(metric.upper())
            ax.set_xticks(SAMPLES)
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{metric.upper()} vs samples per pixel")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_vs_spp.png", dpi=180)
        plt.close(fig)


def make_method_delta_plot(records: List[dict], out_dir: Path) -> None:
    # Positive delta means method_2 better for PSNR/SSIM, lower for MSE/MAE.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metric_specs = [
        ("mse", "MSE (m1 - m2)"),
        ("mae", "MAE (m1 - m2)"),
        ("psnr", "PSNR (m2 - m1)"),
        ("ssim", "SSIM (m2 - m1)"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metric_specs):
        for image_name in INPUTS:
            delta = []
            for spp in SAMPLES:
                m1 = next(
                    rec[metric]
                    for rec in records
                    if rec["image"] == image_name
                    and rec["method"] == "method_1_uniform_init"
                    and rec["spp"] == spp
                )
                m2 = next(
                    rec[metric]
                    for rec in records
                    if rec["image"] == image_name
                    and rec["method"] == "method_2_rejection_init"
                    and rec["spp"] == spp
                )
                if metric in ("mse", "mae"):
                    delta.append(m1 - m2)
                else:
                    delta.append(m2 - m1)
            ax.plot(SAMPLES, delta, marker="o", label=image_name)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_title(title)
        ax.set_xlabel("Samples per pixel")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Method advantage (positive favors method_2)")
    fig.tight_layout()
    fig.savefig(out_dir / "method2_advantage.png", dpi=180)
    plt.close(fig)


def write_markdown_summary(
    output_path: Path, records: List[dict], summary_rows: List[dict]
) -> None:
    lines = []
    lines.append("# Metropolis Analysis Summary")
    lines.append("")
    lines.append("## Dataset")
    lines.append("- Inputs: jaguar, lion, tiger, zebra")
    lines.append("- Methods: method_1_uniform_init, method_2_rejection_init")
    lines.append("- Samples per pixel: 1, 4, 8, 64, 256, 1024")
    lines.append("")
    lines.append("## Aggregate Metrics by Method and SPP")
    lines.append("")
    lines.append("| method | spp | mse | mae | psnr | ssim |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['spp']} | {row['mse']:.6f} | {row['mae']:.6f} | {row['psnr']:.4f} | {row['ssim']:.5f} |"
        )

    lines.append("")
    lines.append("## Best Config per Image (max PSNR)")
    lines.append("")
    lines.append("| image | method | spp | psnr | ssim |")
    lines.append("|---|---|---:|---:|---:|")
    for image_name in INPUTS:
        best = max(
            [r for r in records if r["image"] == image_name],
            key=lambda x: x["psnr"],
        )
        lines.append(
            f"| {image_name} | {best['method']} | {best['spp']} | {best['psnr']:.4f} | {best['ssim']:.5f} |"
        )

    lines.append("")
    lines.append("## Suggested Insights to Highlight")
    lines.append("- How fast each metric saturates as spp increases (diminishing returns).")
    lines.append("- Which images are harder to reconstruct (texture-heavy regions).")
    lines.append("- Whether rejection-based initialization helps most at low spp.")
    lines.append("- Quality vs computation trade-off between 64/256/1024 spp.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "analysis"
    ensure_dir(out_dir)

    refs = {
        name: load_gray_image(root / "source_images_gray" / f"{name}.png")
        for name in INPUTS
    }

    records: List[dict] = []
    loaded: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    for method_name, folder in METHODS.items():
        loaded[method_name] = {}
        for image_name in INPUTS:
            loaded[method_name][image_name] = {}
            ref = refs[image_name]
            for spp in SAMPLES:
                img_path = root / folder / f"sample_{spp}" / f"{image_name}_metro_{spp}spp.png"
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing output image: {img_path}")
                cur = load_gray_image(img_path)
                loaded[method_name][image_name][spp] = cur
                records.append(
                    {
                        "method": method_name,
                        "image": image_name,
                        "spp": spp,
                        "mse": mse(ref, cur),
                        "mae": mae(ref, cur),
                        "psnr": psnr(ref, cur),
                        "ssim": ssim_global(ref, cur),
                    }
                )

    # Per-image qualitative sheets.
    for image_name in INPUTS:
        method_images = {
            method_name: loaded[method_name][image_name] for method_name in METHODS.keys()
        }
        render_qualitative_grid(
            out_dir / f"qualitative_{image_name}.png",
            refs[image_name],
            method_images,
            f"{image_name}: qualitative comparison",
        )
        render_error_map_grid(
            out_dir / f"error_map_{image_name}.png",
            refs[image_name],
            method_images,
            f"{image_name}: absolute error map",
        )

    # Numeric tables.
    records_sorted = sorted(records, key=lambda x: (x["image"], x["spp"], x["method"]))
    write_csv(
        out_dir / "metrics_per_case.csv",
        records_sorted,
        ["method", "image", "spp", "mse", "mae", "psnr", "ssim"],
    )

    summary_rows = []
    for method_name in METHODS.keys():
        for spp in SAMPLES:
            subset = [r for r in records if r["method"] == method_name and r["spp"] == spp]
            summary_rows.append(
                {
                    "method": method_name,
                    "spp": spp,
                    "mse": float(np.mean([x["mse"] for x in subset])),
                    "mae": float(np.mean([x["mae"] for x in subset])),
                    "psnr": float(np.mean([x["psnr"] for x in subset])),
                    "ssim": float(np.mean([x["ssim"] for x in subset])),
                }
            )
    write_csv(
        out_dir / "metrics_summary_by_method_spp.csv",
        summary_rows,
        ["method", "spp", "mse", "mae", "psnr", "ssim"],
    )

    make_line_plots(records, out_dir)
    make_method_delta_plot(records, out_dir)
    write_markdown_summary(out_dir / "summary.md", records, summary_rows)

    print(f"Analysis complete. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
