# H100 / B200 GPU Benchmark

A small, reproducible benchmark for NVIDIA Hopper (H100) and Blackwell (B200) GPUs. It identifies the device, measures dense matmul throughput in FP32, BF16 and FP8, and checks whether LLM weights fit in memory. It runs as a Python script on a GPU host or as a Colab notebook.

## Contents

| Path | Purpose |
|---|---|
| `gpu_bench.py` | Full benchmark. Writes a timestamped JSON file to `results/`. |
| `notebooks/h100_b200_gpu_bench.ipynb` | Colab version with the same measurements and a results table to fill in. |
| `results/` | JSON output from each run. One file per run. |
| `RESULTS.md` | Summary of runs on real hardware, with the hardware and settings for each. |

## Run it

```bash
pip install torch            # any CUDA-enabled build; FP8 needs a recent torch
python3 gpu_bench.py
python3 gpu_bench.py --sizes 8192,16384 --iters 100
```

Colab: open the notebook, set the runtime to GPU, and run all cells.

## What it measures, and how

- **Identity:** product name, compute capability (H100 = 9.0, B200 = 10.0), memory, SM count, torch and CUDA versions.
- **Throughput:** achieved TFLOP/s for square matmuls, timed with CUDA events after warm-up. The GPU is synchronized before and after timing, so asynchronous launches are not miscounted.
- **FP8:** measured through `torch._scaled_mm`. If that op is unavailable on the GPU or torch build, the output says so instead of reporting a number.
- **Model fit:** weights-only estimate for 8B and 70B parameter models in BF16 and FP8, against free memory with 15% kept for KV cache.

## Limits

- Single GPU. No multi-GPU, NVLink, or NCCL measurement.
- One matmul shape family. Not a model-level or serving benchmark.
- Achieved TFLOP/s, not a percentage of the datasheet peak. Peak numbers depend on clocks, power limits and the exact SKU, so they are not hard-coded.
- The model-fit check ignores KV cache and activations beyond a flat 15% reserve.
- Colab sessions may get a different GPU than requested. Always record the device name from the output.

## Results

No results are reported in this repository until they are measured on hardware. See `RESULTS.md` for the table of runs. Each row must link to a JSON file in `results/`.
