"""Wrap llama.cpp's `llama-bench`, the community-standard timing tool.

`llama-bench -o json` emits a JSON array with one entry per test. Prompt-processing
("pp") rows carry n_prompt>0/n_gen=0; token-generation ("tg") rows carry n_gen>0.
We pair them into a single BenchResult per model and attach power sampled during the run.

StrixBench's value-add over raw llama-bench: the reproducibility fingerprint, power +
efficiency, multi-model/multi-config orchestration, and a publishable result format.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..power import PowerSampler
from ..schema import BenchResult


class LlamaCppRunner:
    name = "llama.cpp"

    def __init__(self, binary: str = "llama-bench"):
        self.binary = binary

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def _build_commit(self) -> str:
        try:
            out = subprocess.run(
                [self.binary, "--version"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            text = (out.stderr or "") + (out.stdout or "")
            for line in text.splitlines():
                if "build" in line.lower():
                    return line.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return ""

    def run_model(self, model: dict, defaults: dict, fingerprint_id: str,
                  timestamp: str) -> list[BenchResult]:
        """Run one model spec, returning a list of BenchResult (one per param combo)."""
        path = model["path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"model not found: {path}")

        backend = model.get("backend", defaults.get("backend", "vulkan"))
        n_ctx = model.get("n_ctx", defaults.get("n_ctx", 4096))
        n_prompt = model.get("n_prompt", defaults.get("n_prompt", 512))
        n_gen = model.get("n_gen", defaults.get("n_gen", 128))
        n_gpu_layers = model.get("n_gpu_layers", defaults.get("n_gpu_layers", 999))
        n_batch = model.get("n_batch", defaults.get("n_batch", 2048))
        n_threads = model.get("n_threads", defaults.get("n_threads"))
        reps = model.get("reps", defaults.get("reps", 3))

        cmd = [
            self.binary,
            "-m", path,
            "-p", str(n_prompt),
            "-n", str(n_gen),
            "-ngl", str(n_gpu_layers),
            "-b", str(n_batch),
            "-c", str(n_ctx),
            "-r", str(reps),
            "-o", "json",
        ]
        if n_threads:
            cmd += ["-t", str(n_threads)]

        with PowerSampler() as sampler:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        power = sampler.summary()

        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-bench failed ({proc.returncode}):\n{proc.stderr.strip()[-2000:]}"
            )

        entries = json.loads(proc.stdout)
        result = self._fold(entries, model, backend, fingerprint_id, timestamp)
        result.gpu_avg_w = power.gpu_avg_w
        result.gpu_max_w = power.gpu_max_w
        result.pkg_avg_w = power.pkg_avg_w
        result.n_gpu_layers = n_gpu_layers
        result.n_batch = n_batch
        result.n_ctx = n_ctx
        result.engine_build = self._build_commit()
        return [result.finalize()]

    def _fold(self, entries: list[dict], model: dict, backend: str,
              fingerprint_id: str, timestamp: str) -> BenchResult:
        """Collapse llama-bench's pp/tg rows into one record."""
        res = BenchResult(
            fingerprint_id=fingerprint_id,
            timestamp=timestamp,
            engine=self.name,
            engine_backend=backend,
            model_name=model.get("name") or Path(model["path"]).stem,
            model_quant=model.get("quant", ""),
            is_moe=model.get("is_moe", False),
            raw={"entries": entries},
        )
        for e in entries:
            size = e.get("model_size")
            if size and res.model_size_gb is None:
                res.model_size_gb = round(int(size) / 1024 ** 3, 2)
            if e.get("model_n_params"):
                res.model_n_params = int(e["model_n_params"])
            n_gen = int(e.get("n_gen", 0) or 0)
            n_prompt = int(e.get("n_prompt", 0) or 0)
            avg_ts = e.get("avg_ts")
            std_ts = e.get("stddev_ts")
            if n_gen > 0 and n_prompt == 0:           # token-generation (decode) row
                res.decode_tps = _f(avg_ts)
                res.decode_tps_stddev = _f(std_ts)
                res.n_gen = n_gen
            elif n_prompt > 0 and n_gen == 0:         # prompt-processing (prefill) row
                res.prefill_tps = _f(avg_ts)
                res.prefill_tps_stddev = _f(std_ts)
                res.n_prompt = n_prompt
        return res


def _f(x) -> float | None:
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None
