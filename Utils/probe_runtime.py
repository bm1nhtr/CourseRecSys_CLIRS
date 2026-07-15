"""
CLI: probe host resources and optionally write ``runtime`` into Config/run.json.

Purpose
-------
Record CPU / GPU / RAM and suggested parallelization settings under the nested
config key ``runtime``. This script only updates configuration files; it does
not start training pipelines.

Usage (from repository root)
----------------------------
Dry-run (print only)::

    poetry run python Utils/probe_runtime.py --Config Config/run.json

Write resolved settings into the config file::

    poetry run python Utils/probe_runtime.py --Config Config/run.json \\
        --n-workers 3 --device cpu --write

Arguments
---------
``--Config``
    Path to nested ``run.json`` (YAML is not supported for ``--write``).
``--n-workers``
    Requested parallel trial workers. Omit for default ``1``. Value is capped
    by ``Utils.hw_profile.suggest_n_workers``.
``--device``
    ``cpu``, ``cuda``, or ``auto`` (default ``auto``).
``--n-envs``
    Suggested vectorized-env count stored as ``runtime.n_envs``.
``--write``
    Merge the built ``runtime`` object into the JSON file (replace prior block).
``--print-json``
    Also print the ``runtime`` dict as indented JSON.

See also: ``Utils.hw_profile`` for detection and capping logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Utils.hw_profile import (  # noqa: E402
    build_runtime_section,
    detect_hardware,
    format_probe_report,
)


def _load_json(path: Path) -> dict:
    """Load a UTF-8 JSON object from ``path``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    """Write ``data`` as indented JSON with a trailing newline."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def merge_runtime_into_config(
    raw: dict,
    runtime: dict,
) -> dict:
    """
    Return a shallow copy of ``raw`` with top-level ``runtime`` replaced.

    Other config sections (``experiment``, ``model``, ``data``, …) are left
    unchanged. The previous ``runtime`` value, if any, is discarded entirely
    so the probe snapshot cannot partially merge with stale fields.
    """
    out = dict(raw)
    out["runtime"] = runtime
    return out


def main() -> None:
    """Parse CLI flags, probe hardware, print the report, optionally write config."""
    parser = argparse.ArgumentParser(
        description=(
            "Probe host CPU/GPU/RAM and optionally write "
            "Config/run.json runtime settings."
        )
    )
    parser.add_argument(
        "--Config",
        default=str(_REPO_ROOT / "Config" / "run.json"),
        help="Path to nested run.json (default: Config/run.json)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=(
            "Requested parallel trial workers "
            "(default 1 when omitted; capped by n_cpu//2, CUDA max 2)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Preferred device: cpu, cuda, or auto (default: auto)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Suggested vectorized-env count stored in runtime.n_envs",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Merge suggested runtime into the Config JSON file",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the runtime block as JSON",
    )
    args = parser.parse_args()

    config_path = Path(args.Config).resolve()
    if not config_path.is_file():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    # Nested run.json only: YAML flat configs are not updated by --write.
    if config_path.suffix.lower() != ".json":
        print(
            "[ERROR] --write / probe targets nested run.json only "
            f"(got {config_path.suffix}).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        hw = detect_hardware()
        runtime = build_runtime_section(
            hw,
            n_workers=args.n_workers,
            device=args.device,
            n_envs=args.n_envs,
        )
    except ValueError as exc:
        # e.g. --device cuda on a host with no CUDA GPU
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(format_probe_report(hw, runtime, requested_workers=args.n_workers))
    if args.print_json:
        print("\nruntime JSON:")
        print(json.dumps(runtime, indent=2))

    if not args.write:
        # Echo a ready-to-copy write command using the *resolved* values.
        print(
            "\nDry-run only (config not modified). To save the values above:\n"
            f"  poetry run python Utils/probe_runtime.py --Config {config_path} "
            f"--n-workers {runtime['n_workers']} --device {runtime['device']} --write"
        )
        return

    raw = _load_json(config_path)
    updated = merge_runtime_into_config(raw, runtime)
    _write_json(config_path, updated)
    print(f"\nWrote runtime -> {config_path}")


if __name__ == "__main__":
    main()
