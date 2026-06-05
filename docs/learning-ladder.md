# The StrixBench learning ladder

The project is deliberately structured so each rung ships something the community can
use *and* forces you to understand the layer beneath the one before it.

## Rung 1 — Measure (this repo, v0)

**Build:** fingerprint + `llama-bench` wrapper + power sampling + report.

**Learn:**
- The **roofline**: decode tok/s ≈ memory_bandwidth / active_bytes_per_token. With
  ~256 GB/s, a 40 GB Q4 dense model tops out near ~6 tok/s. Prove this empirically.
- Why **prefill** (compute-bound) and **decode** (bandwidth-bound) scale differently.
- **Quantization** trade-offs (Q4_K_M vs Q5/Q6/Q8): size vs quality vs speed.
- The **ROCm vs Vulkan** story on `gfx1151` — measure both, publish the gap.

**Ship:** a public results table. Already valuable — nobody has clean `gfx1151` numbers.

## Rung 2 — Fit the giants

**Build:** tuned launch configs for 235B-class MoE models that fit in 128 GB.

**Learn:** MoE routing, expert offload (`--n-cpu-moe`-style splits), KV-cache placement,
why MoE is *the* model class for this hardware.

**Ship:** known-good configs + recommended quants per model.

## Rung 3 — Implement to understand

**Build:** a minimal MoE forward pass yourself — PyTorch/ROCm first, then a single
expert-FFN kernel in HIP or Vulkan compute.

**Learn:** what actually happens per token; gating, top-k expert selection, the GEMMs.

**Ship:** a teaching repo others learn from.

## Rung 4 — Schedule (the frontier)

**Build:** route cold experts to CPU/NPU while the iGPU handles attention; measure the win.

**Learn:** XDNA2 / XRT, heterogeneous scheduling, the unified-memory advantage (no PCIe copies).

**Ship:** a hybrid inference engine tuned for Strix Halo.

---

Start at rung 1. Get real numbers off your own box this week, publish them, and let the
data pull you up the ladder.
