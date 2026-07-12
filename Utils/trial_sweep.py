"""T independent trials: seed policy, resume, and sweep bookkeeping."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np
from stable_baselines3.common.utils import set_random_seed

from Utils.results_paths import completed_trial_ids, rl_seed_for_trial

RL_ALGORITHMS = frozenset({"dqn", "ppo"})


def apply_rl_seed(rl_seed: int) -> None:
    """Seed SB3, numpy, torch, and python ``random`` for one trial."""
    set_random_seed(rl_seed)
    random.seed(rl_seed)
    np.random.seed(rl_seed)


def validate_trial_config(config: Mapping[str, Any]) -> list[str]:
    """Return non-fatal warnings about trial / seed setup."""
    warnings: list[str] = []
    nb_runs = int(config.get("nb_runs", 1))
    if nb_runs < 1:
        raise ValueError("experiment.nb_runs must be >= 1")
    if nb_runs < 5:
        warnings.append(
            f"nb_runs={nb_runs} is below 5 — bootstrap CI in sweep summary will be noisy"
        )
    algorithm = str(config.get("model", "")).lower()
    if algorithm in RL_ALGORITHMS and int(config.get("total_steps", 0)) <= 0:
        warnings.append(f"total_steps=0 for RL algorithm {algorithm!r}")
    return warnings


def trials_to_run(
    config: Mapping[str, Any],
    *,
    from_trial: int = 0,
    to_trial: int | None = None,
    resume: bool = True,
) -> list[int]:
    """Trial ids to execute (respecting resume and optional slice)."""
    nb_runs = int(config["nb_runs"])
    end = nb_runs if to_trial is None else min(int(to_trial), nb_runs)
    start = max(0, int(from_trial))
    if start >= end:
        return []
    completed = completed_trial_ids(config) if resume else set()
    return [trial_id for trial_id in range(start, end) if trial_id not in completed]


def trial_plan_summary(
    config: Mapping[str, Any],
    trial_ids: list[int],
    *,
    resume: bool,
) -> str:
    nb_runs = int(config["nb_runs"])
    completed = completed_trial_ids(config)
    rl_base = int(config.get("rl_seed_base", config.get("seed", 42)))
    lines = [
        f"Trial plan: {len(trial_ids)}/{nb_runs} to run "
        f"(resume={'on' if resume else 'off'})",
        f"RL seed policy: rl_base({rl_base}) + trial_id",
    ]
    if completed:
        lines.append(f"Already in sweep CSV: trial_ids {sorted(completed)}")
    if trial_ids:
        seeds = [rl_seed_for_trial(config, t) for t in trial_ids[:5]]
        preview = ", ".join(str(s) for s in seeds)
        if len(trial_ids) > 5:
            preview += ", ..."
        lines.append(f"Next rl_seeds: {preview}")
    return "\n".join(lines)
