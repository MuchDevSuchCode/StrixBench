# StrixBench

A reproducible LLM inference benchmark + config database for **AMD Strix Halo**
(Ryzen AI MAX+ 395 / Radeon 8060S / `gfx1151`).

## Why

Strix Halo's defining trait is **~128 GB of unified memory at ~120 W**. Token
generation is memory-bandwidth bound (~256 GB/s), so per-token it's slower than a
discrete GPU — but it can *hold* models a 24 GB card physically cannot. That makes
it arguably the cheapest box on earth for running frontier-class **Mixture-of-Experts**
models locally (big total params → needs capacity; small active params → low bandwidth).

The software story, however, is a mess: ROCm on `gfx1151` is immature, Vulkan vs ROCm
performance is unclear, the XDNA2 NPU sits idle, and nobody agrees on the right flags.

**StrixBench produces trustworthy, reproducible numbers + known-good configs** so the
community stops guessing.

## What it does

- **`strixbench info`** — captures a full reproducibility *fingerprint* of your stack
  (kernel, ROCm/Mesa versions, `gfx` target, memory split, BIOS GTT allocation, engine
  build commit). Every result is tied to a fingerprint so numbers are comparable.
- **`strixbench run`** — orchestrates benchmarks across models × engines × configs,
  wrapping `llama-bench` for timing and sampling **power** (amdgpu hwmon + RAPL) during
  each run.
- **`strixbench report`** — renders the result JSON into a sortable Markdown table
  ready to publish or submit to the community DB.

## Quickstart (run this on the Strix Halo box, Ubuntu 24.04)

```bash
# 0. Prereqs: a built llama.cpp with llama-bench on PATH (Vulkan and/or ROCm backend)
# 1. Capture your stack fingerprint
python -m strixbench info

# 2. Point a config at your local GGUF models, then run
cp configs/models.example.toml configs/models.toml
$EDITOR configs/models.toml
python -m strixbench run --config configs/models.toml

# 3. Build the report
python -m strixbench report
```

No third-party Python deps — stdlib only (Python 3.11+, uses `tomllib`).

## Roadmap

- [x] **Rung 1 — Measure**: fingerprint + llama-bench wrapper + power + report ← *you are here*
- [ ] Rung 2 — Fit the giants: tuned configs for 235B-class MoE models that fit in 128 GB
- [ ] Rung 3 — Implement to understand: a minimal MoE forward pass / HIP expert kernel
- [ ] Rung 4 — Schedule: route cold experts to CPU/NPU while the iGPU handles attention

See `docs/` for the learning ladder behind each rung.

## Contributing results

Run `strixbench run` then `strixbench report`, and open a PR adding your
`results/*.json` files. The fingerprint makes every submission self-describing.
