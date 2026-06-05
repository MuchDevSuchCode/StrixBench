"""Result record schema. One record per (model × engine × config) measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class BenchResult:
    # Identity / reproducibility
    fingerprint_id: str
    timestamp: str                      # ISO 8601, passed in (no wall-clock in core logic)
    engine: str = ""                    # e.g. "llama.cpp"
    engine_backend: str = ""            # e.g. "vulkan" | "rocm" | "cpu"
    engine_build: str = ""              # build commit / version

    # Model under test
    model_name: str = ""
    model_quant: str = ""               # e.g. Q4_K_M
    model_size_gb: float | None = None
    model_n_params: int | None = None
    is_moe: bool = False

    # Run parameters
    n_gpu_layers: int | None = None
    n_threads: int | None = None
    n_batch: int | None = None
    n_ctx: int | None = None
    n_prompt: int | None = None         # prefill token count for the test
    n_gen: int | None = None            # decode token count for the test

    # Core metrics
    prefill_tps: float | None = None    # prompt-processing tokens/sec
    decode_tps: float | None = None     # generation tokens/sec
    prefill_tps_stddev: float | None = None
    decode_tps_stddev: float | None = None

    # Power / efficiency (best-effort)
    gpu_avg_w: float | None = None
    gpu_max_w: float | None = None
    pkg_avg_w: float | None = None
    decode_tokens_per_joule: float | None = None  # decode_tps / pkg_avg_w

    notes: str = ""
    raw: dict = field(default_factory=dict)

    def finalize(self) -> "BenchResult":
        """Derive efficiency metric where inputs exist."""
        if self.decode_tps and self.pkg_avg_w:
            self.decode_tokens_per_joule = round(self.decode_tps / self.pkg_avg_w, 3)
        return self

    def to_dict(self) -> dict:
        return asdict(self)
