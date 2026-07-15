"""
Host hardware detection and ``runtime`` suggestion helpers.

This module does not train models or run pipelines. It only:

1. Inspects the current host (CPUs, GPUs, RAM).
2. Suggests a conservative ``n_workers`` value for parallel trial fan-out.
3. Formats a console report and builds the ``runtime`` dict for Config/run.json.

Worker policy (see ``suggest_n_workers``)::

    default requested = 1  (omit --n-workers)
    hard_cap = n_cpu // 2
    n_workers = min(max(1, requested), hard_cap)

Entry point for operators: ``Utils/probe_runtime.py``.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Mapping


def detect_hardware() -> dict[str, Any]:
    """
    Return a best-effort snapshot of host resources.

    Returns
    -------
    dict
        Keys:

        - ``os``: ``platform.system()`` string (e.g. ``Windows``, ``Linux``).
        - ``n_cpu``: logical CPU count (``os.cpu_count()``, at least 1).
        - ``n_gpu``: CUDA device count (0 if Torch/CUDA unavailable).
        - ``gpu_names``: list of CUDA device name strings (may be empty).
        - ``free_ram_gb`` / ``total_ram_gb``: float GiB, or ``None`` if unknown.

    Notes
    -----
    Failures in optional dependencies (``psutil``, ``torch``) are swallowed so
    the probe always returns a dict usable for suggestion logic.
    """
    n_cpu = os.cpu_count() or 1
    free_ram_gb: float | None = None
    total_ram_gb: float | None = None

    # Prefer psutil when installed; otherwise fall back to /proc on Linux.
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        free_ram_gb = round(mem.available / (1024**3), 2)
        total_ram_gb = round(mem.total / (1024**3), 2)
    except Exception:
        free_ram_gb, total_ram_gb = _ram_fallback()

    n_gpu = 0
    gpu_names: list[str] = []
    # Torch is optional here: missing torch / no CUDA => n_gpu = 0.
    try:
        import torch

        if torch.cuda.is_available():
            n_gpu = int(torch.cuda.device_count())
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    except Exception:
        pass

    return {
        "os": platform.system(),
        "n_cpu": int(n_cpu),
        "n_gpu": int(n_gpu),
        "gpu_names": gpu_names,
        "free_ram_gb": free_ram_gb,
        "total_ram_gb": total_ram_gb,
    }


def _ram_fallback() -> tuple[float | None, float | None]:
    """
    Read RAM from ``/proc/meminfo`` when ``psutil`` is unavailable.

    Returns
    -------
    tuple
        ``(free_ram_gb, total_ram_gb)`` in GiB, or ``(None, None)`` on
        non-Linux hosts or parse errors.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    info[parts[0].strip()] = parts[1].strip()
        total_kb = int(info.get("MemTotal", "0").split()[0])
        # MemAvailable is preferred over MemFree (accounts for reclaimable cache).
        avail_kb = int(info.get("MemAvailable", info.get("MemFree", "0")).split()[0])
        if total_kb <= 0:
            return None, None
        return round(avail_kb / (1024**2), 2), round(total_kb / (1024**2), 2)
    except Exception:
        return None, None


def suggest_n_workers(
    hw: Mapping[str, Any],
    *,
    requested: int | None = None,
) -> int:
    """
    Map a requested worker count onto a safe host-specific value.

    Parameters
    ----------
    hw :
        Output of ``detect_hardware()`` (needs ``n_cpu``).
    requested :
        Operator request from ``--n-workers``. ``None`` means default **1**
        (sequential / safest), not ``cap``.

    Returns
    -------
    int
        Final ``n_workers`` after capping:

        - Never below 1.
        - Never above ``n_cpu // 2`` (leaves headroom for OS / library threads).

    Examples
    --------
    20 logical CPUs, no ``--n-workers`` -> 1.
    Same host, ``requested=4`` -> 4.
    Same host, ``requested=20`` -> 10 (``20 // 2``).
    """
    n_cpu = max(1, int(hw.get("n_cpu") or 1))
    # Leave roughly half the logical cores for the OS and libraries.
    cap = max(1, n_cpu // 2)
    n = 1 if requested is None else max(1, int(requested))
    return min(n, cap)


def build_runtime_section(
    hw: Mapping[str, Any],
    *,
    n_workers: int | None = None,
) -> dict[str, Any]:
    """
    Build the top-level ``runtime`` object for nested ``Config/run.json``.

    Parameters
    ----------
    hw :
        ``detect_hardware()`` result.
    n_workers :
        Optional requested parallel trial workers (passed to ``suggest_n_workers``).

    Returns
    -------
    dict
        Structure mirrored in Config::

            {
              "_note": "...",
              "n_workers": int,
              "detected": { ... hardware snapshot ... }
            }

    Notes
    -----
    ``detected`` is informational (audit of the host at probe time). Callers
    that persist this dict typically overwrite any previous ``runtime`` key.
    Only ``n_workers`` is consumed by pipelines (parallel trial fan-out).
    """
    resolved_workers = suggest_n_workers(hw, requested=n_workers)
    return {
        "_note": (
            "Host runtime settings (n_workers drives parallel trial fan-out). Update with: "
            "poetry run python Utils/probe_runtime.py --Config Config/run.json --write"
        ),
        "n_workers": resolved_workers,
        "detected": {
            "n_cpu": hw.get("n_cpu"),
            "n_gpu": hw.get("n_gpu"),
            "gpu_names": hw.get("gpu_names") or [],
            "free_ram_gb": hw.get("free_ram_gb"),
            "total_ram_gb": hw.get("total_ram_gb"),
            "os": hw.get("os"),
        },
    }


def format_probe_report(
    hw: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    requested_workers: int | None = None,
) -> str:
    """
    Format a multi-section console report for operators.

    Sections
    --------
    1. Hardware probe — OS, CPU, GPU, RAM.
    2. Suggested runtime — resolved ``n_workers`` and cap.
    3. How to raise n_workers — host-specific ladder and example ``--write`` commands.

    The ladder is ``1 -> 2 -> n_cpu//4 -> hard_cap`` (duplicates removed), so
    operators can step up without jumping straight to the maximum.

    Parameters
    ----------
    hw :
        Hardware snapshot.
    runtime :
        Output of ``build_runtime_section``.
    requested_workers :
        Raw ``--n-workers`` value (may be ``None``); shown for transparency.

    Returns
    -------
    str
        Plain-text report safe for Windows consoles (ASCII arrows only).
    """
    n_cpu = max(1, int(hw.get("n_cpu") or 1))
    cap = max(1, n_cpu // 2)

    # Conservative ladder toward the hard cap (unique, ascending).
    ladder: list[int] = []
    for candidate in (1, 2, max(2, n_cpu // 4), cap):
        c = min(max(1, candidate), cap)
        if c not in ladder:
            ladder.append(c)

    lines = [
        "=== Hardware probe ===",
        f"OS:            {hw.get('os')}",
        f"Logical CPUs:  {hw.get('n_cpu')}",
        f"GPUs:          {hw.get('n_gpu')} "
        f"({', '.join(hw.get('gpu_names') or []) or 'none'})",
        f"RAM free/total:{hw.get('free_ram_gb')} / {hw.get('total_ram_gb')} GB",
        "",
        "=== Suggested runtime (Config/run.json -> runtime) ===",
        f"n_workers:     {runtime.get('n_workers')}"
        + (
            f"  (requested={requested_workers}, capped by n_cpu//2)"
            if requested_workers is not None
            else "  (default=1)"
        ),
        f"n_workers cap: {cap}  (rule: min(requested, n_cpu//2))",
        "",
        "=== How to raise n_workers ===",
        "Start at 1, then step up only if the host stays stable (CPU/RAM OK).",
        f"Suggested ladder for this host: {' -> '.join(str(x) for x in ladder)}",
        "Example commands:",
    ]
    # Show at most two steps above the current suggestion.
    current = int(runtime.get("n_workers") or 1)
    examples = [w for w in ladder if w > current][:2] or ([cap] if current < cap else [])
    if not examples and current >= cap:
        lines.append(f"  (already at or above useful steps; hard cap is {cap})")
    else:
        for w in examples:
            lines.append(
                f"  poetry run python Utils/probe_runtime.py --Config Config/run.json "
                f"--n-workers {w} --write"
            )
    lines.append(
        "If processes thrash or RAM drops sharply, lower n_workers and re-run --write."
    )
    return "\n".join(lines)
