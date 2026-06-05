"""Best-effort power sampling during a benchmark run.

On Strix Halo everything shares one package, so we sample two sources and report both:
  - GPU rail via amdgpu hwmon `power1_average` (microwatts)
  - Whole-package via RAPL energy counters (`/sys/class/powercap/...`), differenced

A background thread polls while the workload runs; call summary() afterwards.
This is intentionally simple — v0 reports average watts, not per-phase energy.
"""

from __future__ import annotations

import glob
import threading
import time
from dataclasses import dataclass

from .util import read_int


def _find_amdgpu_power_input() -> str | None:
    """Locate the amdgpu hwmon power1_average node (path varies by hwmonN index)."""
    for base in glob.glob("/sys/class/hwmon/hwmon*"):
        name = None
        try:
            with open(f"{base}/name") as f:
                name = f.read().strip()
        except OSError:
            continue
        if name == "amdgpu":
            candidate = f"{base}/power1_average"
            if read_int(candidate) is not None:
                return candidate
    return None


def _find_rapl_energy() -> str | None:
    """Locate a RAPL package energy counter (microjoules), AMD or Intel namespace."""
    for p in glob.glob("/sys/class/powercap/*/energy_uj"):
        if read_int(p) is not None:
            return p
    return None


@dataclass
class PowerSummary:
    gpu_avg_w: float | None = None
    gpu_max_w: float | None = None
    pkg_avg_w: float | None = None
    samples: int = 0


class PowerSampler:
    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self._gpu_node = _find_amdgpu_power_input()
        self._rapl_node = _find_rapl_energy()
        self._gpu_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rapl_start: int | None = None
        self._rapl_end: int | None = None
        self._t_start = 0.0
        self._t_end = 0.0

    @property
    def available(self) -> bool:
        return self._gpu_node is not None or self._rapl_node is not None

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._gpu_node:
                uw = read_int(self._gpu_node)
                if uw is not None:
                    self._gpu_samples.append(uw / 1e6)
            time.sleep(self.interval_s)

    def __enter__(self) -> "PowerSampler":
        self._t_start = time.monotonic()
        if self._rapl_node:
            self._rapl_start = read_int(self._rapl_node)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._t_end = time.monotonic()
        if self._rapl_node:
            self._rapl_end = read_int(self._rapl_node)

    def summary(self) -> PowerSummary:
        s = PowerSummary(samples=len(self._gpu_samples))
        if self._gpu_samples:
            s.gpu_avg_w = round(sum(self._gpu_samples) / len(self._gpu_samples), 1)
            s.gpu_max_w = round(max(self._gpu_samples), 1)
        if self._rapl_start is not None and self._rapl_end is not None:
            dt = self._t_end - self._t_start
            dj = (self._rapl_end - self._rapl_start) / 1e6  # uJ -> J
            if dt > 0 and dj >= 0:  # counters wrap; ignore negative deltas in v0
                s.pkg_avg_w = round(dj / dt, 1)
        return s
