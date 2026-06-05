"""Small shared helpers. All best-effort — this tooling runs on a possibly-bare box."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], timeout: float = 30.0) -> str | None:
    """Run a command, return stdout stripped, or None on any failure.

    Used for probing system tools (rocminfo, vulkaninfo, ...) that may not exist.
    """
    if shutil.which(cmd[0]) is None:
        return None
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
