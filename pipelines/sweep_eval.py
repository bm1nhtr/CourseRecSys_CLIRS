"""Post-sweep summary — called at the end of each pipeline run."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np
import pandas as pd

from Utils.experiment_log import ExperimentRunLog
from Utils.results_paths import method_slug, sweep_csv_path


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "std": None}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = sample.mean()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))),
    }


def summarize_sweep_csv(
    csv_path: str,
    *,
    expected_trials: int,
    metrics: tuple[str, ...] = ("end", "life"),
) -> dict[str, Any]:
    """Build a compact report dict from one sweep CSV."""
    if not os.path.isfile(csv_path):
        return {
            "csv_path": csv_path,
            "expected_trials": expected_trials,
            "completed_trials": 0,
            "missing_trial_ids": list(range(expected_trials)),
            "metrics": {},
        }

    df = pd.read_csv(csv_path)
    present_ids = sorted(int(x) for x in df["trial_id"].unique()) if "trial_id" in df.columns else []
    expected_ids = list(range(expected_trials))
    missing = [i for i in expected_ids if i not in set(present_ids)]

    metric_stats: dict[str, Any] = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            metric_stats[metric] = _bootstrap_mean_ci(np.array([], dtype=float))
        else:
            metric_stats[metric] = _bootstrap_mean_ci(series.to_numpy(dtype=float))

    return {
        "csv_path": csv_path,
        "expected_trials": expected_trials,
        "completed_trials": len(present_ids),
        "trial_ids": present_ids,
        "missing_trial_ids": missing,
        "metrics": metric_stats,
    }


def write_sweep_report(
    config: Mapping[str, Any],
    experiment_root: str,
    report: Mapping[str, Any],
) -> str:
    reports_dir = os.path.join(experiment_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"{method_slug(config)}_sweep_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def print_sweep_summary(report: Mapping[str, Any]) -> None:
    print("\n--- Sweep summary ---")
    print(
        f"Trials in CSV: {report.get('completed_trials')}/"
        f"{report.get('expected_trials')}"
    )
    missing = report.get("missing_trial_ids") or []
    if missing:
        print(f"Missing trial_ids: {missing}")
    for metric, stats in (report.get("metrics") or {}).items():
        if stats.get("mean") is None:
            print(f"  {metric}: no data")
            continue
        print(
            f"  {metric}: mean={stats['mean']:.6f} "
            f"95% CI [{stats['ci_low']:.6f}, {stats['ci_high']:.6f}] "
            f"(std={stats['std']:.6f})"
        )


def run_sweep_eval(
    config: Mapping[str, Any],
    experiment_root: str,
    run_log: ExperimentRunLog | None = None,
) -> dict[str, Any] | None:
    """Summarize sweep CSV and write ``reports/{method}_sweep_summary.json``."""
    csv_path = sweep_csv_path(config)
    if not os.path.isfile(csv_path):
        if run_log is not None:
            run_log.warn("No sweep CSV found — skipping sweep summary")
        return None

    report = summarize_sweep_csv(
        csv_path,
        expected_trials=int(config["nb_runs"]),
    )
    report["method"] = method_slug(config)
    report["data_seed"] = config.get("seed")
    report["rl_seed_base"] = config.get("rl_seed_base")

    missing = report.get("missing_trial_ids") or []
    if missing and run_log is not None:
        run_log.warn(
            f"Sweep incomplete: missing trial_ids {missing} "
            f"({report['completed_trials']}/{report['expected_trials']} in CSV)"
        )

    out_path = write_sweep_report(config, experiment_root, report)
    print_sweep_summary(report)
    print(f"Sweep report: {out_path}")
    return report
