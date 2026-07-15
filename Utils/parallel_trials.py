"""
Parallel trial fan-out for pipeline scripts — without changing train/eval trial logic.

When ``runtime.n_workers`` > 1, the **parent** process:

1. Runs the first pending trial alone (freeze ``manifest.json`` / clusters / split).
2. Splits the remaining trial ids into contiguous chunks.
3. Spawns child processes of the **same** pipeline script with
   ``--from-trial`` / ``--to-trial`` / ``--parallel-worker`` / ``--skip-eval``.
4. Returns ``True`` so the parent skips the in-process trial loop and only runs
   ``run_sweep_eval`` (if requested).

Children set ``--parallel-worker`` (and ``CLIRS_PARALLEL_WORKER=1``) so they execute
the existing sequential trial loop for their slice only.

This module does not import Dataset, Reinforce, or env code.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def resolved_n_workers(config: Mapping[str, Any]) -> int:
    """Read flat ``n_workers`` from config (default 1)."""
    try:
        return max(1, int(config.get("n_workers", 1) or 1))
    except (TypeError, ValueError):
        return 1


def is_parallel_worker(config: Mapping[str, Any] | None = None) -> bool:
    """True when this process must NOT fan out again."""
    if os.environ.get("CLIRS_PARALLEL_WORKER", "").strip() in {"1", "true", "yes"}:
        return True
    if config is not None and config.get("_parallel_worker"):
        return True
    return False


def chunk_trial_ids(trial_ids: Sequence[int], n_workers: int) -> list[list[int]]:
    """
    Split sorted trial ids into up to ``n_workers`` contiguous chunks.

    Contiguous chunks map cleanly onto ``--from-trial`` / ``--to-trial`` ranges.
    """
    ids = sorted(int(t) for t in trial_ids)
    if not ids:
        return []
    n = max(1, min(int(n_workers), len(ids)))
    base, rem = divmod(len(ids), n)
    chunks: list[list[int]] = []
    index = 0
    for worker in range(n):
        take = base + (1 if worker < rem else 0)
        if take <= 0:
            continue
        chunks.append(ids[index : index + take])
        index += take
    return chunks


def _range_for_chunk(chunk: Sequence[int]) -> tuple[int, int]:
    """Inclusive trial ids ``[a, ..., b]`` -> ``--from-trial a --to-trial b+1``."""
    return int(chunk[0]), int(chunk[-1]) + 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _worker_command(
    script_path: Path,
    config_path: str,
    from_trial: int,
    to_trial: int,
    *,
    force: bool,
    no_resume: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--Config",
        config_path,
        "--from-trial",
        str(from_trial),
        "--to-trial",
        str(to_trial),
        "--parallel-worker",
        "--skip-eval",
    ]
    if force:
        cmd.append("--force")
    if no_resume:
        cmd.append("--no-resume")
    return cmd


def _run_worker(
    script_path: Path,
    config_path: str,
    from_trial: int,
    to_trial: int,
    *,
    force: bool,
    no_resume: bool,
) -> None:
    cmd = _worker_command(
        script_path,
        config_path,
        from_trial,
        to_trial,
        force=force,
        no_resume=no_resume,
    )
    env = os.environ.copy()
    # Avoid oversubscription when many SB3 processes share one host.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env["CLIRS_PARALLEL_WORKER"] = "1"
    print(f"[parallel] spawn: from_trial={from_trial} to_trial={to_trial}")
    subprocess.run(cmd, check=True, cwd=str(_repo_root()), env=env)


def try_fan_out_trials(
    *,
    script_path: str | Path,
    config_path: str,
    config: Mapping[str, Any],
    trial_ids: Sequence[int],
    force: bool = False,
    no_resume: bool = False,
) -> bool:
    """
    Fan out trials across worker processes when ``n_workers`` > 1.

    Returns
    -------
    bool
        ``True`` if this parent finished spawning workers (caller should skip the
        in-process trial loop and typically only run sweep eval).
        ``False`` if the caller should run the existing sequential trial loop
        (``n_workers == 1``, single trial, or already a worker).
    """
    if is_parallel_worker(config):
        return False

    n_workers = resolved_n_workers(config)
    pending = [int(t) for t in trial_ids]
    if n_workers <= 1 or len(pending) <= 1:
        return False

    script = Path(script_path).resolve()
    print(
        f"[parallel] Fan-out enabled: n_workers={n_workers}, "
        f"pending_trials={pending}"
    )

    # First pending trial alone: Create Algorithm manifest / clusters / split pins.
    first = pending[0]
    print(f"[parallel] Seed trial (freeze artifacts): trial_id={first}")
    _run_worker(
        script,
        config_path,
        first,
        first + 1,
        force=force,
        no_resume=no_resume,
    )

    remaining = pending[1:]
    if not remaining:
        return True

    chunks = chunk_trial_ids(remaining, n_workers)
    # Launch subsequent chunks together; wait for each (ordered join).
    # Sequential wait keeps peak RAM lower than fully detached pools; still
    # overlaps only as much as we start sequentially — use Popen for true overlap.
    procs: list[subprocess.Popen] = []
    for chunk in chunks:
        from_trial, to_trial = _range_for_chunk(chunk)
        cmd = _worker_command(
            script,
            config_path,
            from_trial,
            to_trial,
            force=force,
            no_resume=no_resume,
        )
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env["CLIRS_PARALLEL_WORKER"] = "1"
        print(
            f"[parallel] spawn chunk trials={list(chunk)} "
            f"(from={from_trial}, to={to_trial})"
        )
        procs.append(
            subprocess.Popen(cmd, cwd=str(_repo_root()), env=env)
        )

    failed = False
    for proc in procs:
        code = proc.wait()
        if code != 0:
            failed = True
            print(f"[parallel] worker exited with code {code}", file=sys.stderr)

    if failed:
        raise SystemExit("One or more parallel trial workers failed")
    print("[parallel] All worker chunks finished")
    return True
