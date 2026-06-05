"""Small shared helpers. All best-effort — this tooling runs on a possibly-bare box."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


# System tools like modinfo/dmidecode live in sbin, which often isn't on a user's PATH.
_EXTRA_BIN_DIRS = ("/usr/sbin", "/sbin", "/usr/local/sbin", "/opt/rocm/bin")


def _resolve(prog: str) -> str | None:
    """Find an executable, also checking sbin dirs that may be off the user PATH."""
    found = shutil.which(prog)
    if found:
        return found
    for d in _EXTRA_BIN_DIRS:
        cand = Path(d) / prog
        if cand.is_file():
            return str(cand)
    return None


def run(cmd: list[str], timeout: float = 30.0) -> str | None:
    """Run a command, return stdout stripped, or None on any failure.

    Used for probing system tools (rocminfo, vulkaninfo, ...) that may not exist.
    """
    prog = _resolve(cmd[0])
    if prog is None:
        return None
    cmd = [prog, *cmd[1:]]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def read_text(path: str | Path) -> str | None:
    """Read a sysfs/proc file, returning None if absent or unreadable."""
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


def read_int(path: str | Path) -> int | None:
    txt = read_text(path)
    if txt is None:
        return None
    try:
        return int(txt)
    except ValueError:
        return None
