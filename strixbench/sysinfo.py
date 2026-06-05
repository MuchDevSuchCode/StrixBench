"""Capture a reproducibility *fingerprint* of the Strix Halo software/hardware stack.

The whole point of StrixBench is comparable numbers. A token/sec figure is meaningless
without knowing the kernel, ROCm/Mesa version, gfx target, and how the BIOS split memory
between system RAM and GPU GTT/VRAM. This module gathers all of that, best-effort.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import asdict, dataclass, field

from .util import read_int, read_text, run


@dataclass
class Fingerprint:
    host: str = ""
    os: str = ""
    kernel: str = ""
    cpu_model: str = ""
    cpu_cores: int | None = None
    gpu_name: str = ""
    gfx_target: str = ""           # e.g. gfx1151 — critical for ROCm compatibility
    rocm_version: str = ""
    mesa_radv_version: str = ""    # Vulkan backend version
    amdgpu_driver: str = ""
    mem_total_gb: float | None = None
    gtt_total_gb: float | None = None   # how much unified mem the GPU can address
    vram_total_gb: float | None = None  # carved-out "dedicated" portion
    npu_present: bool = False
    bios_version: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable short hash of the parts that affect comparability."""
        key = "|".join(
            str(x)
            for x in (
                self.kernel, self.gfx_target, self.rocm_version,
                self.mesa_radv_version, self.amdgpu_driver,
                self.gtt_total_gb, self.vram_total_gb,
            )
        )
        return hashlib.sha256(key.encode()).hexdigest()[:12]


def _cpu() -> tuple[str, int | None]:
    info = read_text("/proc/cpuinfo") or ""
    model = ""
    cores = 0
    for line in info.splitlines():
        if line.startswith("model name") and not model:
            model = line.split(":", 1)[1].strip()
        if line.startswith("processor"):
            cores += 1
    return model, (cores or None)


def _mem_total_gb() -> float | None:
    info = read_text("/proc/meminfo") or ""
    m = re.search(r"MemTotal:\s+(\d+)\s+kB", info)
    return round(int(m.group(1)) / 1024 / 1024, 1) if m else None


def _gfx_target() -> str:
    out = run(["rocminfo"]) or ""
    m = re.search(r"gfx\d+", out)
    return m.group(0) if m else ""


def _gpu_name() -> str:
    # Prefer rocm-smi product name; fall back to the DRM card name.
    out = run(["rocm-smi", "--showproductname", "--json"])
    if out:
        try:
            data = json.loads(out)
            for card in data.values():
                name = card.get("Card series") or card.get("Card model")
                if name:
                    return str(name)
        except (json.JSONDecodeError, AttributeError):
            pass
    return read_text("/sys/class/drm/card0/device/product_name") or ""


def _rocm_version() -> str:
    v = read_text("/opt/rocm/.info/version")
    return v.splitlines()[0] if v else ""


def _radv_version() -> str:
    out = run(["vulkaninfo", "--summary"]) or run(["vulkaninfo"]) or ""
    m = re.search(r"driverInfo\s*=\s*(Mesa[^\n]+)", out)
    if m:
        return m.group(1).strip()
    m = re.search(r"(Mesa \d[\w.\-]+)", out)
    return m.group(1) if m else ""


def _amdgpu_driver() -> str:
    ver = read_text("/sys/module/amdgpu/version")
    return ver or ""


def _gtt_vram_gb() -> tuple[float | None, float | None]:
    # amdgpu exposes pool sizes via debugfs/sysfs; sizes are in bytes.
    def gb(p: str) -> float | None:
        v = read_int(p)
        return round(v / 1024 ** 3, 1) if v else None

    gtt = gb("/sys/class/drm/card0/device/mem_info_gtt_total")
    vram = gb("/sys/class/drm/card0/device/mem_info_vram_total")
    return gtt, vram


def _npu_present() -> bool:
    # XDNA2 NPU shows up as an amdxdna driver / accel device.
    if read_text("/sys/module/amdxdna/version"):
        return True
    out = run(["lsmod"]) or ""
    return "amdxdna" in out


def _bios_version() -> str:
    return read_text("/sys/class/dmi/id/bios_version") or ""


def collect() -> Fingerprint:
    cpu_model, cpu_cores = _cpu()
    gtt, vram = _gtt_vram_gb()
    return Fingerprint(
        host=platform.node(),
        os=read_text("/etc/os-release") and _pretty_os() or platform.platform(),
        kernel=platform.release(),
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        gpu_name=_gpu_name(),
        gfx_target=_gfx_target(),
        rocm_version=_rocm_version(),
        mesa_radv_version=_radv_version(),
        amdgpu_driver=_amdgpu_driver(),
        mem_total_gb=_mem_total_gb(),
        gtt_total_gb=gtt,
        vram_total_gb=vram,
        npu_present=_npu_present(),
        bios_version=_bios_version(),
    )


def _pretty_os() -> str:
    txt = read_text("/etc/os-release") or ""
    m = re.search(r'PRETTY_NAME="([^"]+)"', txt)
    return m.group(1) if m else ""


def to_dict(fp: Fingerprint) -> dict:
    d = asdict(fp)
    d["fingerprint_id"] = fp.id
    return d
