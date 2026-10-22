# Results

Each row is one run of `gpu_bench.py` or the Colab notebook on real hardware. Do not add rows for runs you did not perform. Link each row to its JSON file in `results/`.

| Date (UTC) | GPU (as reported) | Compute cap. | Environment | FP32 n=8192 | BF16 n=8192 | FP8 n=8192 | Source file |
|---|---|---|---|---|---|---|---|
| 2026-10-22 | Tesla T4 (not a target GPU) | 7.5 | Colab free tier, torch 2.11.0+cu130, CUDA 13.0 | 3.9 | 2.1 | unavailable (needs CC 9.0+) | results/ not yet saved |

Environment: Colab runtime type, cloud instance, or on-premises host. Name the provider or SKU if known.

## Notes on reading results

- Compare only runs with the same software stack (torch and CUDA version) and the same matmul size.
- A GPU that reports as H100 or B200 is not proof of the SKU. Check the name and memory size in the JSON.
- Colab does not guarantee a GPU model. A run on a T4 or A100 is valid evidence for that GPU only.
