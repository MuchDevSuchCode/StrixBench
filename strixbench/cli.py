"""StrixBench command-line interface.

  strixbench info                       capture + save the stack fingerprint
  strixbench run --config models.toml   run benchmarks, save results JSON
  strixbench report                     render results/ into REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, sysinfo
from .report import render
from .runners import RUNNERS

RESULTS_DIR = Path("results")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(ts: str) -> str:
    return ts.replace(":", "").replace("-", "")


def cmd_info(args: argparse.Namespace) -> int:
    fp = sysinfo.collect()
    data = sysinfo.to_dict(fp)
    print(json.dumps(data, indent=2))

    warnings = []
    if not fp.gfx_target:
        warnings.append("rocminfo not found — can't confirm gfx target (gfx1151 expected).")
    if not fp.rocm_version and not fp.mesa_radv_version:
        warnings.append("Neither ROCm nor Vulkan/Mesa detected — no GPU backend to bench.")
    for w in warnings:
        print(f"  ! {w}", file=sys.stderr)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"fingerprint-{fp.id}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"\nSaved fingerprint -> {out}", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = tomllib.loads(cfg_path.read_text())

    fp = sysinfo.collect()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"fingerprint-{fp.id}.json").write_text(
        json.dumps(sysinfo.to_dict(fp), indent=2)
    )

    defaults = cfg.get("defaults", {})
    models = cfg.get("models", [])
    if not models:
        print("config has no [[models]] entries.", file=sys.stderr)
        return 2

    engine_name = cfg.get("engine", "llama.cpp")
    runner_cls = RUNNERS.get(engine_name)
    if runner_cls is None:
        print(f"unknown engine '{engine_name}'. known: {list(RUNNERS)}", file=sys.stderr)
        return 2
    runner = runner_cls(**cfg.get("engine_opts", {}))
    if not runner.available():
        print(f"engine '{engine_name}' binary not on PATH.", file=sys.stderr)
        return 2

    ts = _now_iso()
    all_results = []
    for i, model in enumerate(models, 1):
        name = model.get("name") or model.get("path", "?")
        print(f"[{i}/{len(models)}] benchmarking {name} ...", file=sys.stderr)
        try:
            results = runner.run_model(model, defaults, fp.id, _now_iso())
        except Exception as e:  # one bad model shouldn't sink the whole run
            print(f"    FAILED: {e}", file=sys.stderr)
            continue
        for r in results:
            d = r.decode_tps
            print(f"    decode={d} tok/s  prefill={r.prefill_tps} tok/s  "
                  f"pkg={r.pkg_avg_w} W", file=sys.stderr)
            all_results.append(r.to_dict())

    if not all_results:
        print("no successful results.", file=sys.stderr)
        return 1

    out = RESULTS_DIR / f"run-{_slug(ts)}-{fp.id}.json"
    out.write_text(json.dumps(
        {"timestamp": ts, "fingerprint_id": fp.id, "results": all_results}, indent=2))
    print(f"\nSaved {len(all_results)} result(s) -> {out}", file=sys.stderr)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    md = render(Path(args.results_dir))
    out = Path(args.out)
    out.write_text(md)
    print(md)
    print(f"\nWrote {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="strixbench", description=__doc__)
    p.add_argument("--version", action="version", version=f"strixbench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="capture the stack fingerprint")
    pi.add_argument("--no-save", action="store_true", help="print only, don't save")
    pi.set_defaults(func=cmd_info)

    pr = sub.add_parser("run", help="run benchmarks from a config")
    pr.add_argument("--config", default="configs/models.toml")
    pr.set_defaults(func=cmd_run)

    pp = sub.add_parser("report", help="render results into Markdown")
    pp.add_argument("--results-dir", default="results")
    pp.add_argument("--out", default="REPORT.md")
    pp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
