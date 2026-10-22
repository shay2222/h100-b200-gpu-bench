#!/usr/bin/env python3
"""
Characterize and benchmark NVIDIA H100 (Hopper, CC 9.0) and B200 (Blackwell, CC 10.0) GPUs.

What it measures on device 0:
  - Identity: product name, compute capability, memory, SM count, software versions
  - Dense matmul throughput in FP32, BF16, and FP8 (FP8 only where torch._scaled_mm works)
  - Memory headroom and whether an N-billion-parameter model's weights fit

Every number is measured on the machine that runs the script. Nothing is hard-coded
from datasheets, so the output is only as strong as the run that produced it.

Usage:
  python3 gpu_bench.py                       # defaults: sizes 4096,8192; models 8,70 B
  python3 gpu_bench.py --sizes 8192 --iters 100 --out results
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ARCH_BY_CC = {
    (8, 0): "Ampere (A100)",
    (9, 0): "Hopper (H100)",
    (10, 0): "Blackwell (B200)",
}
TARGET_ARCHES = {"Hopper (H100)", "Blackwell (B200)"}


def device_info(idx: int) -> dict:
    p = torch.cuda.get_device_properties(idx)
    cc = (p.major, p.minor)
    return {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "architecture": ARCH_BY_CC.get(cc, "other"),
        "memory_gib": round(p.total_memory / 1024**3, 1),
        "sm_count": p.multi_processor_count,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "fp8_dtype_available": hasattr(torch, "float8_e4m3fn"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
    }


def time_per_iter(fn, warmup: int, iters: int) -> float:
    """Average seconds per call, timed with CUDA events after a warm-up."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def dense_tflops(n: int, dtype: torch.dtype, warmup: int, iters: int) -> float:
    a = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
    sec = time_per_iter(lambda: a @ b, warmup, iters)
    return 2 * n**3 / sec / 1e12


def fp8_tflops(n: int, warmup: int, iters: int) -> float:
    """FP8 matmul via torch._scaled_mm. Raises if the op or dtype is unsupported."""
    a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
    # _scaled_mm expects the second operand column-major: transpose a contiguous tensor.
    b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t()
    scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    fn = lambda: torch._scaled_mm(a, b, scale_a=scale, scale_b=scale,
                                  out_dtype=torch.bfloat16)
    sec = time_per_iter(fn, warmup, iters)
    return 2 * n**3 / sec / 1e12


def memory_check(params_b: float, bytes_per_param: int, free_gib: float,
                 kv_headroom: float = 0.15) -> dict:
    """Weights-only estimate. Leaves kv_headroom of free memory for KV cache and activations."""
    weights_gib = params_b * 1e9 * bytes_per_param / 1024**3
    budget = free_gib * (1 - kv_headroom)
    return {
        "params_b": params_b,
        "dtype": {1: "fp8", 2: "bf16"}.get(bytes_per_param, f"{bytes_per_param}B"),
        "weights_gib": round(weights_gib, 1),
        "fits": weights_gib <= budget,
        "budget_gib": round(budget, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="4096,8192", help="square matmul sizes, comma-separated")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--models", default="8,70", help="model sizes in billions of params")
    ap.add_argument("--out", default="results", help="directory for the JSON result")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device visible. Run on a GPU runtime.", file=sys.stderr)
        return 1

    info = device_info(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3

    sizes = [int(s) for s in args.sizes.split(",")]
    results = {"fp32": {}, "bf16": {}, "fp8": {}}
    for n in sizes:
        results["fp32"][n] = round(dense_tflops(n, torch.float32, args.warmup, args.iters), 2)
        results["bf16"][n] = round(dense_tflops(n, torch.bfloat16, args.warmup, args.iters), 2)
        try:
            results["fp8"][n] = round(fp8_tflops(n, args.warmup, args.iters), 2)
        except Exception as exc:  # unsupported on this GPU or torch build
            results["fp8"][n] = f"unavailable: {type(exc).__name__}"

    memory = []
    for m in [float(x) for x in args.models.split(",")]:
        memory.append(memory_check(m, 2, free_gib))  # bf16
        memory.append(memory_check(m, 1, free_gib))  # fp8

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": info,
        "target_arch": info["architecture"] in TARGET_ARCHES,
        "memory_free_gib": round(free_gib, 1),
        "memory_total_gib": round(total_bytes / 1024**3, 1),
        "settings": {"sizes": sizes, "iters": args.iters, "warmup": args.warmup},
        "tflops": results,
        "model_fit": memory,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = info["name"].replace(" ", "_").replace("/", "_")
    out_file = out_dir / f"{slug}_{stamp}.json"
    out_file.write_text(json.dumps(report, indent=2))

    print(f"Device:       {info['name']} [{info['architecture']}], CC {info['compute_capability']}")
    print(f"Memory:       {report['memory_free_gib']} GiB free of {report['memory_total_gib']} GiB")
    print(f"Software:     torch {info['torch_version']}, CUDA {info['cuda_version']}")
    if not report["target_arch"]:
        print("Note: this GPU is not H100 or B200. Results are valid for this GPU only.")
    print("Throughput (TFLOP/s, achieved):")
    for n in sizes:
        fp8 = results["fp8"][n]
        fp8_s = f"{fp8:8.1f}" if isinstance(fp8, float) else f"{fp8}"
        print(f"  n={n:<6} FP32 {results['fp32'][n]:8.1f} | BF16 {results['bf16'][n]:8.1f} | FP8 {fp8_s}")
    print("Model weights vs. free memory:")
    for m in memory:
        verdict = "fits" if m["fits"] else "does not fit"
        print(f"  {m['params_b']:g}B {m['dtype']}: {m['weights_gib']} GiB -> {verdict} "
              f"(budget {m['budget_gib']} GiB)")
    print(f"Saved: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
