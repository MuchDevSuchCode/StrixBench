"""Benchmark models served by Ollama via its HTTP API.

Ollama reports timing in its final /api/generate response:
  prompt_eval_count / prompt_eval_duration  -> prefill tok/s
  eval_count        / eval_duration         -> decode tok/s
so we get exact numbers without an external timer. Model size/quant come from /api/show.

Reference a model by its Ollama tag (e.g. "qwen3:30b"), not a file path. This lets you
put the *same* underlying model behind both the llama.cpp and Ollama runners and compare.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..power import PowerSampler
from ..schema import BenchResult

DEFAULT_HOST = "http://127.0.0.1:11434"
# A deterministic, moderately long prompt so prefill numbers are meaningful.
_PROMPT = ("Explain, step by step and in depth, how a mixture-of-experts transformer "
           "routes tokens to experts and why that suits a memory-bandwidth-bound "
           "accelerator. ") * 6


class OllamaRunner:
    name = "ollama"

    def __init__(self, host: str = DEFAULT_HOST):
        self.host = host.rstrip("/")

    def _post(self, path: str, payload: dict, timeout: float = 1800.0) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _show(self, tag: str) -> dict:
        try:
            return self._post("/api/show", {"model": tag}, timeout=30)
        except (urllib.error.URLError, OSError, ValueError):
            return {}

    def run_model(self, model: dict, defaults: dict, fingerprint_id: str,
                  timestamp: str) -> list[BenchResult]:
        tag = model.get("tag") or model.get("name")
        if not tag:
            raise ValueError("ollama model entry needs a 'tag' (or 'name')")

        n_ctx = model.get("n_ctx", defaults.get("n_ctx", 4096))
        n_gen = model.get("n_gen", defaults.get("n_gen", 128))
        n_gpu_layers = model.get("n_gpu_layers", defaults.get("n_gpu_layers", 999))
        n_batch = model.get("n_batch", defaults.get("n_batch", 2048))
        n_threads = model.get("n_threads", defaults.get("n_threads"))

        options = {
            "num_ctx": n_ctx,
            "num_predict": n_gen,
            "num_gpu": n_gpu_layers,
            "num_batch": n_batch,
            "temperature": 0,  # determinism
        }
        if n_threads:
            options["num_thread"] = n_threads

        # keep_alive=0 unloads the model right after this request, so the next
        # benchmark (possibly a different engine) starts from a clean memory state
        # instead of contending with a model Ollama would otherwise hold for ~5 min.
        payload = {"model": tag, "prompt": _PROMPT, "stream": False,
                   "keep_alive": 0, "options": options}

        with PowerSampler() as sampler:
            resp = self._post("/api/generate", payload)
        power = sampler.summary()

        res = self._fold(resp, model, tag, fingerprint_id, timestamp)
        res.gpu_avg_w = power.gpu_avg_w
        res.gpu_max_w = power.gpu_max_w
        res.pkg_avg_w = power.pkg_avg_w
        res.n_ctx = n_ctx
        res.n_batch = n_batch
        res.n_gpu_layers = n_gpu_layers
        self._enrich_from_show(res, tag)
        return [res.finalize()]

    def _fold(self, resp: dict, model: dict, tag: str, fingerprint_id: str,
              timestamp: str) -> BenchResult:
        res = BenchResult(
            fingerprint_id=fingerprint_id,
            timestamp=timestamp,
            engine=self.name,
            engine_backend=model.get("backend", "ollama"),
            engine_build=resp.get("model", tag),
            model_name=model.get("name") or tag,
            model_quant=model.get("quant", ""),
            is_moe=model.get("is_moe", False),
            raw={"generate": {k: resp.get(k) for k in (
                "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration")}},
        )
        pe_c = resp.get("prompt_eval_count")
        pe_d = resp.get("prompt_eval_duration")  # nanoseconds
        if pe_c and pe_d:
            res.prefill_tps = round(pe_c / (pe_d / 1e9), 2)
            res.n_prompt = int(pe_c)
        ev_c = resp.get("eval_count")
        ev_d = resp.get("eval_duration")
        if ev_c and ev_d:
            res.decode_tps = round(ev_c / (ev_d / 1e9), 2)
            res.n_gen = int(ev_c)
        return res

    def _enrich_from_show(self, res: BenchResult, tag: str) -> None:
        info = self._show(tag)
        details = info.get("details", {}) or {}
        if not res.model_quant:
            res.model_quant = details.get("quantization_level", "") or ""
        size_b = info.get("size")
        if size_b and res.model_size_gb is None:
            res.model_size_gb = round(int(size_b) / 1024 ** 3, 2)
        # parameter_size like "30.5B"; record raw, leave exact n_params to llama.cpp path
        ps = details.get("parameter_size")
        if ps:
            res.notes = (res.notes + f" params={ps}").strip()
