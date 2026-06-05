"""Capture a reproducibility *fingerprint* of the Strix Halo software/hardware stack.

The whole point of StrixBench is comparable numbers. A token/sec figure is meaningless
without knowing the kernel, ROCm/Mesa version, gfx target, and how the BIOS split memory
between system RAM and GPU GTT/VRAM. This module gathers all of that, best-effort.
"""

from __future__ import annotations

import glob
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
    mem_total_gb: float | None = None        # system RAM the OS sees
    gtt_total_gb: float | None = None         # GTT: system RAM the GPU may borrow (subset of mem)
    vram_total_gb: float | None = None        # dedicated GPU carveout (BIOS UMA split), hidden from OS
    gpu_addressable_gb: float | None = None   # what the GPU can map: VRAM carveout + GTT
    unified_total_gb: float | None = None     # total physical memory installed
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
    # rocminfo "Marketing Name" is the most reliable; there's one per agent (CPU + GPU),
    # so pick the one that isn't the CPU model.
    out = run(["rocminfo"]) or ""
    cpu_model = _cpu()[0].strip()
    for name in re.findall(r"Marketing Name:\s*(.+)", out):
        name = name.strip()
        if name and name != cpu_model:
            return name
    # Fall back to rocm-smi product name (text or json form).
    smi = run(["rocm-smi", "--showproductname"]) or ""
    m = re.search(r"Card (?:Series|Model|series|model)\s*:?\s*(.+)", smi)
    if m:
        return m.group(1).strip()
    return ""


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
    if ver:
        return ver
    # amdgpu is often built-in (no /sys/module/.../version); ask modinfo.
    return run(["modinfo", "-F", "version", "amdgpu"]) or ""


def _gtt_vram_gb() -> tuple[float | None, float | None]:
    """Return (gtt_gb, vram_gb). The card index isn't always 0, so scan all cards;
    fall back to rocm-smi when sysfs nodes are absent."""
    def gb(v: int | None) -> float | None:
        return round(v / 1024 ** 3, 1) if v else None

    gtt = vram = None
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        v = read_int(f"{dev}/mem_info_vram_total")
        g = read_int(f"{dev}/mem_info_gtt_total")
        if v and (vram is None or v > vram):
            vram = v
        if g and (gtt is None or g > gtt):
            gtt = g

    if vram is None:  # rocm-smi fallback
        out = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
        if out:
            try:
                for card in json.loads(out).values():
                    for k, val in card.items():
                        if "VRAM Total Memory" in k:
                            vram = int(val)
            except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
                pass

    return gb(gtt), gb(vram)


def _npu_present() -> bool:
    # XDNA2 NPU shows up as an amdxdna driver / accel device.
    if read_text("/sys/module/amdxdna/version"):
        return True
    out = run(["lsmod"]) or ""
    return "amdxdna" in out


def _bios_version() -> str:
    return read_text("/sys/class/dmi/id/bios_version") or ""


def _dmidecode_total_gb() -> float | None:
    """Exact installed memory from dmidecode (needs root; returns None otherwise)."""
    out = run(["dmidecode", "-t", "memory"])
    if not out:
        return None
    total_mb = 0
    for m in re.finditer(r"Size:\s*(\d+)\s*(MB|GB)", out):
        val = int(m.group(1))
        total_mb += val * 1024 if m.group(2) == "GB" else val
    return round(total_mb / 1024, 1) if total_mb else None


def _capacity(mem: float | None, gtt: float | None,
              vram: float | None) -> tuple[float | None, float | None]:
    """Derive (total_physical_gb, gpu_addressable_gb) from the raw pools.

    Strix Halo runs in one of two memory modes:
      - UMA / shared: tiny VRAM stub, GPU borrows system RAM via a large GTT.
        VRAM overlaps system RAM, so physical == system RAM.
      - Dedicated carveout (BIOS UMA split): a big VRAM region is reserved for
        the GPU and HIDDEN from the OS. It does NOT overlap system RAM, so
        physical == VRAM carveout + system RAM (what the OS sees).
    A VRAM region larger than a few GB indicates a real carveout.
    The GPU can map its dedicated VRAM plus whatever it borrows via GTT.
    """
    addressable = round((vram or 0) + (gtt or 0), 1) or None

    dmi = _dmidecode_total_gb()  # exact installed total, if root
    if dmi:
        return dmi, addressable

    carveout = bool(vram and vram > 4)  # >4 GB ⇒ dedicated carveout, disjoint from RAM
    if carveout and mem:
        return round(vram + mem, 1), addressable
    if mem:                              # UMA: system RAM is the whole pool
        return mem, addressable
    return (vram or None), addressable


def collect() -> Fingerprint:
    cpu_model, cpu_cores = _cpu()
    gtt, vram = _gtt_vram_gb()
    mem = _mem_total_gb()
    unified, addressable = _capacity(mem, gtt, vram)
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
        mem_total_gb=mem,
        gtt_total_gb=gtt,
        vram_total_gb=vram,
        gpu_addressable_gb=addressable,
        unified_total_gb=unified,
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
