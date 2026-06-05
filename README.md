# StrixBench

**Reproducible LLM inference benchmarks + a config database for AMD Strix Halo**
(Ryzen AI MAX+ 395 · Radeon 8060S · `gfx1151` · XDNA2 NPU · 128 GB unified memory).

StrixBench answers the question the community keeps asking and nobody has clean data for:
*"What actually runs, how fast, and with which flags, on a Strix Halo box?"* — and ties
every number to a hashed fingerprint of the exact software/hardware stack that produced it,
so results are comparable across machines and reproducible over time.

---

## Table of contents

- [Why this box, why this project](#why-this-box-why-this-project)
- [The one number that matters: the roofline](#the-one-number-that-matters-the-roofline)
- [Install](#install)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [Configuration reference](#configuration-reference)
- [Output & data format](#output--data-format)
- [Supported engines](#supported-engines)
- [Power & efficiency measurement](#power--efficiency-measurement)
- [Troubleshooting](#troubleshooting)
- [The learning ladder (roadmap)](#the-learning-ladder-roadmap)
- [Contributing results](#contributing-results)
- [Project layout](#project-layout)
- [License](#license)

---

## Why this box, why this project

The AMD Ryzen AI MAX+ 395 ("Strix Halo") pairs 16 Zen 5 cores, a 40-CU RDNA 3.5 iGPU
(`gfx1151`), and an XDNA2 NPU with **~128 GB of unified LPDDR5X memory** at roughly
**256 GB/s**, all inside a ~120 W envelope.

Its defining trait isn't speed — it's **capacity per watt**. Token *generation* is
memory-bandwidth bound, so per token it's slower than a discrete GPU. But it can **hold
models a 24 GB card physically cannot load**, which makes it arguably the cheapest machine
on earth for running frontier-class **Mixture-of-Experts (MoE)** models locally:

| | Needs | Strix Halo |
|---|---|---|
| Big total params | capacity | ✅ 128 GB unified |
| Few active params/token | bandwidth | ✅ low demand → fast enough |

A 26B-total / 4B-active MoE behaves like a 4B model on the bandwidth-bound decode step
while delivering far better quality — exactly the regime this hardware was built for.

The catch is software: ROCm on `gfx1151` is young, Vulkan-vs-ROCm performance is murky,
the NPU sits idle on Linux, and there's no agreed-upon set of launch flags. **StrixBench
turns that guesswork into measured, shareable data.**

> **Memory note:** the pool is **128 GB physical and unified**, but how it's split depends
> on BIOS mode. With a *dedicated carveout* (e.g. 96 GB reserved as GPU VRAM + ~32 GB system
> RAM) the regions are disjoint, so capacity = VRAM + RAM. In *shared/UMA* mode the GPU
> borrows system RAM via GTT and the regions overlap. StrixBench detects which mode you're
> in and reports `unified_total_gb` (total physical) plus `gpu_addressable_gb` (VRAM + GTT,
> i.e. what the GPU can actually map). See [Output & data format](#output--data-format).

---

## The one number that matters: the roofline

For decode (token generation), throughput is bounded by:

```
decode_tok/s  ≈  memory_bandwidth / active_bytes_per_token
```

With ~256 GB/s:

- A dense **70B Q4** (~40 GB read per token) → **~6 tok/s** ceiling.
- A **235B-A22B MoE Q4** (~12 GB active per token) → **~20 tok/s** ceiling *and* fits in
  RAM that no consumer GPU has.

Prefill (prompt processing) is compute-bound and scales differently — which is why
StrixBench reports **prefill and decode separately**. One of the first things you'll do is
prove this roofline empirically on your own box.

---

## Install

Runs on the Strix Halo box itself (Ubuntu 24.04, Python ≥ 3.11). **No third-party Python
dependencies** — stdlib only (uses `tomllib`, `urllib`).

```bash
git clone <your-fork> StrixBench && cd StrixBench
# Run directly — nothing to build:
python3 -m strixbench --help
# (optional) install the `strixbench` entrypoint:
pip install -e .
```

### Engine prerequisites (at least one)

- **llama.cpp** — a build with `llama-bench` on `PATH`, or point the config at the binary.
  Build it once with the **Vulkan** backend and once with **ROCm/HIP** to measure the gap.
- **Ollama** — the daemon running on `http://127.0.0.1:11434` (the default).

---

## Quickstart

```bash
# 1. Capture your stack fingerprint (and save it under results/)
python3 -m strixbench info

# 2. Make a config and point it at your models
cp configs/models.example.toml configs/models.toml
find ~/models -name '*.gguf'      # llama.cpp paths (use the FIRST shard if split)
ollama list                        # ollama tags
nano configs/models.toml           # fix paths/tags

# 3. Run the benchmarks
python3 -m strixbench run --config configs/models.toml

# 4. Render the report
python3 -m strixbench report       # writes REPORT.md and prints it
```

---

## Commands

| Command | What it does |
|---|---|
| `strixbench info` | Probe and print the stack **fingerprint**; save to `results/fingerprint-<id>.json`. Add `--no-save` to print only. |
| `strixbench run --config <file>` | Run every `[[models]]` entry through its engine; save `results/run-<ts>-<id>.json`. |
| `strixbench report [--results-dir results] [--out REPORT.md]` | Aggregate all results into a sorted Markdown table. |

All commands are invokable as `python3 -m strixbench <cmd>` or, after `pip install -e .`,
as `strixbench <cmd>`.

---

## Configuration reference

Config is TOML. See [`configs/models.example.toml`](configs/models.example.toml).

```toml
engine = "llama.cpp"          # default engine for models that don't set one

[runners."llama.cpp"]
binary = "/home/ai/llama.cpp/build/bin/llama-bench"   # optional; else found on PATH

[runners.ollama]
host = "http://127.0.0.1:11434"                        # optional; this is the default

[defaults]                     # inherited by every model unless overridden
backend = "vulkan"             # a LABEL for your llama.cpp build (vulkan | rocm | cpu)
n_ctx = 4096
n_prompt = 512                 # prefill tokens to time
n_gen = 128                    # decode tokens to time
n_gpu_layers = 999             # offload everything (cheap on unified memory)
n_batch = 2048
reps = 3                       # llama-bench repetitions

[[models]]                     # llama.cpp model — referenced by file path
name = "gemma-4-26B-A4B"
path = "/home/ai/models/.../model-00001-of-00002.gguf"   # first shard if split
quant = "Q4"
is_moe = true                  # tags MoE models in the report

[[models]]                     # same logical model via Ollama — for a head-to-head
name = "gemma-4-26B-A4B"
engine = "ollama"
tag = "gemma:27b"              # from `ollama list`
quant = "Q4_K_M"
```

**Key points**
- Any per-model key (`n_ctx`, `n_gen`, `backend`, …) overrides `[defaults]`.
- `engine` may be set per model, so one config can mix llama.cpp and Ollama.
- Give the **same `name`** to the llama.cpp and Ollama variants of a model and the report
  places them adjacent for easy comparison.
- `backend` is only a label — build llama.cpp with the backend you intend to measure.

---

## Output & data format

Everything lands in `results/`:

- `fingerprint-<id>.json` — the stack fingerprint (see fields below). `<id>` is a 12-char
  hash of the comparability-relevant fields (kernel, gfx target, ROCm/Mesa, driver, mem).
- `run-<timestamp>-<id>.json` — `{ timestamp, fingerprint_id, results: [ … ] }`.
- `REPORT.md` — generated table, sorted by decode tok/s.

**Fingerprint fields:** `host, os, kernel, cpu_model, cpu_cores, gpu_name, gfx_target,
rocm_version, mesa_radv_version, amdgpu_driver, mem_total_gb, gtt_total_gb, vram_total_gb,
gpu_addressable_gb, unified_total_gb, npu_present, bios_version, fingerprint_id`.

> **Memory fields explained (Strix Halo runs two modes):**
> - *Dedicated carveout* (BIOS UMA split, e.g. 96 GB VRAM + 32 GB system): the VRAM
>   region is reserved for the GPU and hidden from the OS, so `unified_total_gb` =
>   VRAM + system RAM (~128 GB) and `gpu_addressable_gb` = VRAM + GTT.
> - *Shared/UMA*: a tiny VRAM stub; the GPU borrows system RAM via a large GTT, so
>   `unified_total_gb` = system RAM (the pools overlap).
>
> `unified_total_gb` is exact when `info` is run with `sudo` (reads `dmidecode`);
> otherwise it's derived from the pools as above.

**Result record fields (per model × engine):** `engine, engine_backend, engine_build,
model_name, model_quant, model_size_gb, model_n_params, is_moe, n_gpu_layers, n_threads,
n_batch, n_ctx, n_prompt, n_gen, prefill_tps, decode_tps, *_stddev, gpu_avg_w, gpu_max_w,
pkg_avg_w, decode_tokens_per_joule, notes, raw`.

---

## Supported engines

| Engine | How it's measured | Model reference |
|---|---|---|
| **llama.cpp** | wraps `llama-bench -o json`; folds its prompt-processing (pp) and token-generation (tg) rows into one record | `path` to a `.gguf` (first shard if split) |
| **Ollama** | calls `/api/generate` (`stream:false`) and reads `prompt_eval_*` / `eval_*` timings; pulls size/quant from `/api/show` | `tag` from `ollama list` |

Adding an engine (vLLM, MLC, raw ROCm) is a drop-in: implement a class with
`available()` and `run_model(...) -> list[BenchResult]`, then register it in
`strixbench/runners/__init__.py`.

---

## Power & efficiency measurement

During each run a background sampler records:

- **GPU rail watts** from the `amdgpu` hwmon `power1_average` node (auto-located across
  `hwmon*`).
- **Package watts** from RAPL energy counters (`/sys/class/powercap/*/energy_uj`),
  differenced over the run.

From these StrixBench derives **`decode_tokens_per_joule`** — the efficiency metric that
actually matters for an always-on local box. Sampling is best-effort: if a node is missing
or needs elevated permissions, that field is simply `null` and timing still works.

---

## Troubleshooting

**`vram_total_gb` / `gtt_total_gb` are `null`.**
StrixBench scans `/sys/class/drm/card*/device/mem_info_*` and falls back to
`rocm-smi --showmeminfo vram --json`. If both are empty, check
`ls /sys/class/drm/card*/device/ | grep mem_info` and confirm `rocm-smi` works.

**`unified_total_gb` looks off / you want the exact installed size.**
Without root it's derived from the pools (carveout: VRAM + RAM; UMA: RAM). For the precise
installed total, run `info` with `sudo` so `dmidecode -t memory` is readable. Note `dmidecode`
and `modinfo` live in `/usr/sbin`; StrixBench resolves them there even when it's off your PATH.

**`gpu_name` is empty.**
It's parsed from `rocminfo` ("Marketing Name"). Ensure `rocminfo` runs without sudo and
returns a GPU agent.

**`amdgpu_driver` is empty.**
`amdgpu` is often built into the kernel (no `/sys/module/amdgpu/version`); StrixBench then
tries `modinfo -F version amdgpu`.

**llama.cpp: "engine not available".**
`llama-bench` isn't on `PATH` — set `binary` under `[runners."llama.cpp"]`.

**Ollama: "engine not available".**
The daemon isn't reachable — start it (`ollama serve`) or fix `host`.

**Permission denied editing `configs/models.toml`.**
`.gitignore` excludes it (it holds machine-specific paths); create it fresh from the example.

---

## The learning ladder (roadmap)

StrixBench is deliberately structured so each rung ships something useful *and* forces a
deeper understanding of the layer beneath it. See [`docs/learning-ladder.md`](docs/learning-ladder.md).

- [x] **Rung 1 — Measure** *(this release)*: fingerprint + llama.cpp/Ollama runners +
  power + report. Prove the roofline; publish the first clean `gfx1151` numbers.
- [ ] **Rung 2 — Fit the giants**: tuned configs for 200B-class MoE models that fit in
  128 GB (expert offload splits, KV-cache placement, quant selection).
- [ ] **Rung 3 — Implement to understand**: a minimal MoE forward pass by hand, then a
  single expert-FFN kernel in HIP / Vulkan compute.
- [ ] **Rung 4 — Schedule**: route cold experts to CPU/NPU while the iGPU handles
  attention; exploit unified memory (no PCIe copies) and measure the win.

---

## Contributing results

1. `python3 -m strixbench run` then `python3 -m strixbench report` on your box.
2. Open a PR adding your `results/*.json` files (both the `fingerprint-*` and `run-*`).

Because every result carries its fingerprint, submissions are self-describing — we can
build a cross-machine database of "what runs how fast with which flags" on Strix Halo.

---

## Project layout

```
StrixBench/
├── README.md
├── pyproject.toml                 installable; `strixbench` entrypoint
├── configs/models.example.toml    model zoo template (llama.cpp + ollama)
├── docs/learning-ladder.md        the 4-rung roadmap
├── REPORT.md                      generated report (default --out)
├── results/                       fingerprints (fingerprint-*.json) and runs (run-*.json)
└── strixbench/
    ├── __main__.py / cli.py       info | run | report
    ├── sysinfo.py                 stack fingerprint (gfx, ROCm/Mesa, memory, NPU, BIOS)
    ├── power.py                   amdgpu hwmon + RAPL sampling
    ├── schema.py                  one comparable result record per model × engine
    ├── report.py                  Markdown report generator
    └── runners/
        ├── llamacpp.py            wraps llama-bench
        └── ollama.py              Ollama HTTP API
```

---

## License

MIT — see `LICENSE`.
