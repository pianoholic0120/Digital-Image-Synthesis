#!/usr/bin/env python3
"""
Benchmark CPU acceleration structures for pbrt-v4: BVH, KD-Tree, Uniform Grid,
and Two-Level Uniform Grid.

Uses the **CPU** tool `build_uniform_grid` only (`RenderCPU` respects the scene
`Accelerator` directive). Do **not** use `build_uniform_grid_gpu` here — the GPU
renderer always traces with OptiX and ignores `Accelerator`.

"Construction time" / "Rendering time" from build_uniform_grid are top-level aggregate
build (`CreateAggregate`) and `integrator->Render()` only. "Total work time" is wall
clock from scene parse through end of `RenderCPU` (includes parse, texture/light/material
setup, those two phases, and integrator construction). Progress bar "(Xs)" may differ.

Usage:
  python benchmark_accelerators.py scene.pbrt
  python benchmark_accelerators.py scene.pbrt --exe path/to/build_uniform_grid.exe

If --exe is omitted, common CMake Release output folders are searched for
build_uniform_grid (.exe on Windows).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ACCELERATORS = ["bvh", "kdtree", "uniformgrid", "twoleveluniformgrid"]

SCRIPT_DIR = Path(__file__).resolve().parent


def safe_console_text(text: str, stream) -> str:
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
        return text
    except UnicodeEncodeError:
        return text.encode(enc, errors="replace").decode(enc, errors="replace")


def cpu_exe_name() -> str:
    return "build_uniform_grid.exe" if sys.platform == "win32" else "build_uniform_grid"


def is_gpu_benchmark_exe(exe_path: Path) -> bool:
    stem = exe_path.stem.lower()
    return "build_uniform_grid_gpu" in stem or stem.endswith("_gpu")


def _optional_zlib_dirs_windows() -> list[Path]:
    """Well-known locations of zlib1.dll (no guarantee)."""
    out: list[Path] = []
    for d in (
        Path(r"C:\Program Files\Git\mingw64\bin"),
        Path(r"C:\Program Files\Git\usr\bin"),
    ):
        if (d / "zlib1.dll").is_file():
            out.append(d)
    return out


def _cuda_bin_dirs_windows() -> list[Path]:
    """Typical CUDA Toolkit locations (prepend to PATH so cudart64_*.dll loads)."""
    out: list[Path] = []
    roots = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir(), reverse=True):
            b = child / "bin"
            if b.is_dir():
                out.append(b)
    return out


def subprocess_env_for_pbrt_exe(
    exe_path: Path, extra_path_dirs: list[Path] | None = None
) -> dict[str, str]:
    """Extend PATH so CUDA / zlib runtime DLLs resolve when spawning the renderer."""
    env = dict(os.environ)
    prepend: list[str] = []
    if extra_path_dirs:
        for d in extra_path_dirs:
            if d.is_dir():
                prepend.append(str(d.resolve()))
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        p = Path(cuda_path) / "bin"
        if p.is_dir():
            prepend.append(str(p))
    if sys.platform == "win32":
        for d in _optional_zlib_dirs_windows():
            prepend.append(str(d))
        prepend.extend(str(p) for p in _cuda_bin_dirs_windows())
        # Directory of the .exe (CMake may copy zlib1.dll here).
        prepend.insert(0, str(exe_path.resolve().parent))
    cur = env.get("PATH", "")
    sep = os.pathsep
    for d in prepend:
        if d and d not in cur:
            cur = d + sep + cur
    env["PATH"] = cur
    return env


def explain_windows_return_code(return_code: int | None) -> str:
    if return_code is None or sys.platform != "win32":
        return ""
    rc = return_code
    if rc < 0:
        rc_u = rc + (1 << 32)
    else:
        rc_u = rc & 0xFFFFFFFF
    if rc_u == 0xC0000135:
        return (
            "Windows STATUS_DLL_NOT_FOUND (0xC0000135). pbrt links zlib1.dll and CUDA "
            "runtimes: (1) Place zlib1.dll next to build_uniform_grid.exe, or add its "
            "folder to PATH (see --extra-path). (2) Ensure CUDA Toolkit "
            r"'...\NVIDIA GPU Computing Toolkit\CUDA\v*\bin' is on PATH."
        )
    return ""


def find_default_cpu_executable() -> Path | None:
    """Locate build_uniform_grid next to typical CMake build folders."""
    name = cpu_exe_name()
    rel_dirs = [
        Path("build-cpu-only") / "Release",
        Path("build-cpu-only") / "RelWithDebInfo",
        Path("build-cuda124-customtoolset") / "Release",
        Path("build") / "Release",
        Path("Release"),
        Path("build") / "RelWithDebInfo",
        Path("out") / "build" / "Release",
    ]
    for root in (Path.cwd(), SCRIPT_DIR):
        for sub in rel_dirs:
            p = (root / sub / name).resolve()
            if p.is_file():
                return p
    return None


def set_scene_accelerator(scene_text: str, accelerator: str) -> str:
    accel_line = f'Accelerator "{accelerator}"'
    pattern = re.compile(r'^\s*Accelerator\s+"[^"]+"\s*$', re.MULTILINE)
    if pattern.search(scene_text):
        return pattern.sub(accel_line, scene_text, count=1)
    return accel_line + "\n" + scene_text


def parse_timings(output_text: str):
    timings = {}
    # build_uniform_grid prints integer ms (may be 64-bit values on large runs)
    int_ms = r"(\d+)"
    build_match = re.search(rf"Construction time:\s*{int_ms}\s*ms", output_text)
    render_match = re.search(rf"Rendering time:\s*{int_ms}\s*ms", output_text)
    total_match = re.search(rf"Total work time:\s*{int_ms}\s*ms", output_text)
    if build_match:
        timings["construction_ms"] = int(build_match.group(1))
    if render_match:
        timings["rendering_ms"] = int(render_match.group(1))
    if total_match:
        timings["total_work_ms"] = int(total_match.group(1))
    prog = re.findall(r"\(\s*(\d+\.?\d*)\s*s\s*\)", output_text)
    if prog:
        timings["progress_bar_last_s"] = float(prog[-1])
    return timings


def set_scene_output_filename(scene_text: str, output_file: Path) -> str:
    output_text = output_file.as_posix()
    updated = re.sub(
        r'("string\s+filename"\s*\[\s*")([^"]*)("\s*\])',
        rf'\1{output_text}\3',
        scene_text,
        count=1,
    )
    if updated != scene_text:
        return updated

    updated = re.sub(
        r'("string\s+filename"\s*")([^"]*)(")',
        rf'\1{output_text}\3',
        scene_text,
        count=1,
    )
    if updated != scene_text:
        return updated

    lines = scene_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("Film ") or stripped.startswith("# Film "):
            film_line = stripped[2:] if stripped.startswith("# ") else stripped
            if '"string filename"' not in film_line:
                film_line += f' "string filename" ["{output_text}"]'
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = indent + film_line
            return "\n".join(lines) + ("\n" if scene_text.endswith("\n") else "")

    injected = f'Film "rgb" "string filename" ["{output_text}"]\n'
    return injected + scene_text


def run_case(
    exe_path: Path,
    scene_path: Path,
    timeout_s: int,
    accel: str,
    extra_path_dirs: list[Path] | None = None,
):
    cmd = [str(exe_path), str(scene_path)]
    start_mono = time.perf_counter()
    output_chunks = []
    tail_buffer = deque(maxlen=8000)
    child_env = subprocess_env_for_pbrt_exe(exe_path, extra_path_dirs)
    exe_dir = str(exe_path.parent.resolve())

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=-1,
        env=child_env,
        cwd=exe_dir,
    )

    def reader_thread():
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            output_chunks.append(chunk)
            tail_buffer.extend(chunk)
            sys.stdout.write(chunk)
            sys.stdout.flush()

    thread = threading.Thread(target=reader_thread, daemon=True)
    thread.start()

    last_status = 0.0
    timed_out = False
    while thread.is_alive():
        now = time.perf_counter()
        if now - start_mono > timeout_s:
            timed_out = True
            proc.kill()
            break
        if now - last_status >= 1.0:
            elapsed = now - start_mono
            sys.stdout.write(
                f"\r[{accel}] elapsed {elapsed:.1f}s / timeout {timeout_s}s"
            )
            sys.stdout.flush()
            last_status = now
        time.sleep(0.1)

    thread.join(timeout=30.0)
    try:
        return_code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        return_code = proc.wait()
    sys.stdout.write("\n")
    elapsed_ms = int((time.perf_counter() - start_mono) * 1000)
    merged = "".join(output_chunks)
    timings = parse_timings(merged)
    c = timings.get("construction_ms")
    r = timings.get("rendering_ms")
    tw = timings.get("total_work_ms")
    if tw is None and c is not None and r is not None:
        tw = c + r
    wall_ms = tw if tw is not None else elapsed_ms

    def finalize(d):
        d.setdefault("construction_ms", c)
        d.setdefault("rendering_ms", r)
        d.setdefault("total_work_ms", tw)
        if timings.get("progress_bar_last_s") is not None:
            d["progress_bar_last_s"] = timings["progress_bar_last_s"]
        wk = d.get("wall_time_ms")
        cc = d.get("construction_ms")
        rr = d.get("rendering_ms")
        if wk is not None and cc is not None and rr is not None:
            d["overhead_ms"] = max(0, wk - cc - rr)
        sw = d.get("subprocess_wall_ms")
        if sw is not None and wk is not None:
            d["outside_work_ms"] = max(0, sw - wk)
        return d

    def attach_failure_fields(d: dict) -> dict:
        rc = d.get("return_code")
        if rc not in (0, None):
            hint = explain_windows_return_code(rc)
            if hint:
                d["exit_hint"] = hint
            if merged.strip():
                d["process_output_snippet"] = merged[-12000:]
            elif hint:
                d["process_output_snippet"] = ""
        return d

    if timed_out:
        return attach_failure_fields(
            finalize(
                {
                    "return_code": -1,
                    "error": f"timeout ({timeout_s}s)",
                    "wall_time_ms": wall_ms,
                    "subprocess_wall_ms": elapsed_ms,
                    "construction_ms": None,
                    "rendering_ms": None,
                    "output_tail": "".join(tail_buffer),
                }
            )
        )
    return attach_failure_fields(
        finalize(
            {
                "return_code": return_code,
                "wall_time_ms": wall_ms,
                "subprocess_wall_ms": elapsed_ms,
                "construction_ms": c,
                "rendering_ms": r,
                "total_work_ms": tw,
                "output_tail": "".join(tail_buffer),
            }
        )
    )


def benchmark(
    exe_path: Path,
    scene_file: Path,
    timeout_s: int,
    images_dir: Path,
    extra_path_dirs: list[Path] | None = None,
):
    base_scene = scene_file.read_text(encoding="utf-8")
    results = {}
    patched_files = []
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        for accel in ACCELERATORS:
            print(f"\n=== {accel.upper()} ===")
            patched = set_scene_accelerator(base_scene, accel)
            image_path = images_dir / f"{scene_file.stem}.{accel}.exr"
            patched = set_scene_output_filename(patched, image_path)
            patched_scene = scene_file.with_name(f"{scene_file.stem}.{accel}.bench.pbrt")
            patched_scene.write_text(patched, encoding="utf-8")
            patched_files.append(patched_scene)
            result = run_case(exe_path, patched_scene, timeout_s, accel, extra_path_dirs)
            result["image_file"] = str(image_path.resolve())
            result["accelerator"] = accel
            results[accel] = result
            status = "OK" if result.get("return_code") == 0 else "FAILED"
            print(
                f"status={status}, work_wall={result.get('wall_time_ms', 0)} ms, "
                f"subproc={result.get('subprocess_wall_ms', 0)} ms"
            )
            if result.get("construction_ms") is not None and result.get("rendering_ms") is not None:
                print(
                    f"construction={result['construction_ms']} ms, "
                    f"rendering={result['rendering_ms']} ms"
                )
            if result.get("progress_bar_last_s") is not None:
                print(f"progress_bar_last={result['progress_bar_last_s']} s (pbrt UI)")
            rc = result.get("return_code")
            if rc is not None and rc != 0:
                if result.get("exit_hint"):
                    print(f"return_code={rc}", file=sys.stderr)
                    print(result["exit_hint"], file=sys.stderr)
                out = result.get("process_output_snippet") or result.get("output_tail", "")
                if out and str(out).strip():
                    print("--- subprocess output (tail) ---")
                    print(str(out)[-6000:])
    finally:
        for p in patched_files:
            p.unlink(missing_ok=True)
    return results


def print_summary(results):
    lines = []
    lines.append("=" * 72)
    lines.append("CPU accelerator benchmark (scene Accelerator line applied per run)")
    lines.append("=" * 72)
    lines.append(
        f"{'Accelerator':<14}{'Status':<10}{'Construct':<10}{'Render':<10}"
        f"{'Wall(work)':<12}{'Subproc':<10}"
    )
    lines.append("-" * 72)
    for accel in ACCELERATORS:
        r = results.get(accel, {})
        status = "OK" if r.get("return_code") == 0 else "FAILED"
        c = "-" if r.get("construction_ms") is None else str(r["construction_ms"])
        ren = "-" if r.get("rendering_ms") is None else str(r["rendering_ms"])
        wall = r.get("wall_time_ms", "-")
        subp = r.get("subprocess_wall_ms", "-")
        lines.append(f"{accel:<14}{status:<10}{c:<10}{ren:<10}{wall:<12}{subp:<10}")

    ranked = []
    for accel in ACCELERATORS:
        r = results.get(accel, {})
        if r.get("return_code") != 0:
            continue
        w = r.get("wall_time_ms")
        if w is not None:
            ranked.append((accel, int(w)))
    ranked.sort(key=lambda x: x[1])

    lines.append("")
    if ranked:
        lines.append("Ranking by Wall(work), ms (fastest → slowest):")
        for i, (name, wms) in enumerate(ranked, 1):
            lines.append(f"  {i}. {name}: {wms}")
    else:
        lines.append("Ranking: no successful runs with wall_time_ms.")

    lines.append("")
    lines.append(
        "Wall(work) = Total work time from build_uniform_grid (parse through end of RenderCPU). "
        "Construct/Render columns are aggregate build and integrator->Render only; "
        "overhead_ms = Wall - Construct - Render (parse, textures, integrator setup, etc.). "
        "Subproc = full process wall time (InitPBRT, cleanup, etc.)."
    )
    summary_text = "\n".join(lines)
    print(safe_console_text("\n" + summary_text, sys.stdout))
    return summary_text


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BVH / kdtree / uniformgrid / twoleveluniformgrid using CPU build_uniform_grid only."
    )
    parser.add_argument("scene", help="Path to scene .pbrt file")
    parser.add_argument(
        "--exe",
        default=None,
        metavar="PATH",
        help=f"Path to build_uniform_grid ({cpu_exe_name()} on Windows); auto-detect if omitted",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=72000,
        help="Timeout seconds per subprocess run (default 2 hours)",
    )
    parser.add_argument("--out", default="benchmark_results.json", help="Output JSON path")
    parser.add_argument(
        "--images-dir",
        default="benchmark_images",
        help="Directory for per-accelerator EXR outputs",
    )
    parser.add_argument(
        "--benchmark-log",
        default="benchmark_log.txt",
        help="Path to save summary text only",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Extra directory to prepend to PATH for each render (repeatable). "
            "Use if zlib1.dll or CUDA DLLs are not found (Windows)."
        ),
    )
    args = parser.parse_args()

    if args.exe:
        exe_path = Path(args.exe).expanduser().resolve()
    else:
        exe_path = find_default_cpu_executable()

    scene_file = Path(args.scene).expanduser().resolve()

    if exe_path is None or not exe_path.is_file():
        hint = (
            f"Build the CPU target, e.g.:\n"
            f"  cmake --build <your-build-dir> --config Release --target build_uniform_grid\n"
            f"Expected file name: {cpu_exe_name()}\n"
            f"Searched under: {Path.cwd()} and {SCRIPT_DIR}"
        )
        raise FileNotFoundError(
            "CPU executable build_uniform_grid not found. Pass it as the first "
            f"argument or place it under e.g. build-cuda124-customtoolset/Release/.\n{hint}"
        )

    if is_gpu_benchmark_exe(exe_path):
        print(
            "Error: This script only benchmarks CPU accelerators.\n"
            f"  Refusing GPU executable: {exe_path}\n"
            "  Use: build_uniform_grid (calls RenderCPU; respects Accelerator).\n"
            "  Not: build_uniform_grid_gpu (OptiX; ignores Accelerator).",
            file=sys.stderr,
        )
        sys.exit(2)

    if not scene_file.exists():
        raise FileNotFoundError(f"Scene not found: {scene_file}")

    print("PBRT CPU accelerator benchmark")
    print(f"Executable : {exe_path}")
    print(f"Scene      : {scene_file}")

    images_dir = Path(args.images_dir).resolve()
    extra_dirs = [Path(p).expanduser().resolve() for p in (args.extra_path or [])]
    results = benchmark(exe_path, scene_file, args.timeout, images_dir, extra_dirs)
    summary_text = print_summary(results)

    payload = {
        "benchmark_mode": "cpu",
        "executable": str(exe_path),
        "scene": str(scene_file),
        "accelerators": ACCELERATORS,
        "results": results,
    }
    out_path = Path(args.out).resolve()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    benchmark_log = Path(args.benchmark_log).resolve()
    benchmark_log.write_text(summary_text + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"Summary log: {benchmark_log}")
    print(f"Images     : {images_dir}")


if __name__ == "__main__":
    main()
