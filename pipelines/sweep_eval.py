"""Post-sweep summary — Dataset D load + bootstrap CI (mean/median, multi-method)."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from Utils.experiment_log import ExperimentRunLog
from Utils.results_paths import SWEEP_CSV_COLUMNS, method_slug, sweep_csv_path

DEFAULT_METRICS = ("end", "life")


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | None]:
    if values.size == 0:
        return {"point": None, "ci_low": None, "ci_high": None, "std": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = stat_fn(sample)
    point = float(stat_fn(values))
    return {
        "point": point,
        "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "n": int(n),
    }


def _bootstrap_mean_ci(values: np.ndarray, **kwargs) -> dict[str, float | None]:
    stats = _bootstrap_ci(values, np.mean, **kwargs)
    return {
        "mean": stats["point"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "std": stats["std"],
        "n": stats["n"],
    }


def _bootstrap_median_ci(values: np.ndarray, **kwargs) -> dict[str, float | None]:
    stats = _bootstrap_ci(values, np.median, **kwargs)
    return {
        "median": stats["point"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "std": stats["std"],
        "n": stats["n"],
    }


def list_sweep_csvs(experiment_root: str) -> list[str]:
    """All sweep CSV files under ``sweeps/`` for this experiment cell."""
    sweeps_dir = os.path.join(experiment_root, "sweeps")
    if not os.path.isdir(sweeps_dir):
        return []
    return sorted(glob.glob(os.path.join(sweeps_dir, "*_data*.csv")))


def load_trial_metrics_long(experiment_root: str) -> pd.DataFrame:
    """
    All sweep CSVs in one experiment cell — long-form trial metrics.

    One row per trial per method (Jordan et al. eval layer input).
    """
    frames: list[pd.DataFrame] = []
    for path in list_sweep_csvs(experiment_root):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_csv"] = os.path.basename(path)
        if "method" not in df.columns or df["method"].isna().all():
            stem = os.path.basename(path).split("_data", 1)[0]
            df["method"] = stem
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(SWEEP_CSV_COLUMNS) + ["source_csv"])
    out = pd.concat(frames, ignore_index=True)
    if "trial_id" in out.columns:
        out["trial_id"] = pd.to_numeric(out["trial_id"], errors="coerce").astype("Int64")
    return out


def dataset_d_summary(
    dataset_d: pd.DataFrame,
    *,
    expected_trials: int,
) -> dict[str, Any]:
    """Metadata for dataset D (methods, trial coverage)."""
    if dataset_d.empty:
        return {
            "n_methods": 0,
            "methods": [],
            "expected_trials": expected_trials,
            "methods_detail": {},
        }
    methods_detail: dict[str, Any] = {}
    for method, group in dataset_d.groupby("method", sort=True):
        ids = sorted(int(x) for x in group["trial_id"].dropna().unique())
        expected = list(range(expected_trials))
        missing = [i for i in expected if i not in set(ids)]
        methods_detail[str(method)] = {
            "completed_trials": len(ids),
            "trial_ids": ids,
            "missing_trial_ids": missing,
            "source_csv": group["source_csv"].iloc[0] if "source_csv" in group else None,
        }
    return {
        "n_methods": len(methods_detail),
        "methods": sorted(methods_detail.keys()),
        "expected_trials": expected_trials,
        "methods_detail": methods_detail,
    }


def summarize_sweep_csv(
    csv_path: str,
    *,
    expected_trials: int,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Bootstrap mean + median CI for one sweep CSV."""
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
            metric_stats[metric] = {
                "mean": _bootstrap_mean_ci(np.array([], dtype=float)),
                "median": _bootstrap_median_ci(np.array([], dtype=float)),
            }
        else:
            arr = series.to_numpy(dtype=float)
            metric_stats[metric] = {
                "mean": _bootstrap_mean_ci(arr),
                "median": _bootstrap_median_ci(arr),
            }

    return {
        "csv_path": csv_path,
        "expected_trials": expected_trials,
        "completed_trials": len(present_ids),
        "trial_ids": present_ids,
        "missing_trial_ids": missing,
        "metrics": metric_stats,
    }


BOOTSTRAP_EXPORT_COLUMNS = (
    "method",
    "metric",
    "statistic",
    "value",
    "ci_low",
    "ci_high",
    "n_trials",
)


def bootstrap_aggregate_table(
    dataset_d: pd.DataFrame,
    metric: str,
    *,
    stat: str = "mean",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Multi-method bootstrap summary for one metric and statistic (mean or median).

    Columns: method, metric, statistic, value, ci_low, ci_high, n_trials.
    """
    if dataset_d.empty or metric not in dataset_d.columns:
        return pd.DataFrame(columns=list(BOOTSTRAP_EXPORT_COLUMNS))

    stat_fn = np.mean if stat == "mean" else np.median
    rng = np.random.default_rng(seed)
    methods = sorted(dataset_d["method"].dropna().unique())
    rows: list[dict[str, Any]] = []

    for method in methods:
        series = pd.to_numeric(
            dataset_d.loc[dataset_d["method"] == method, metric],
            errors="coerce",
        ).dropna()
        arr = series.to_numpy(dtype=float)
        n_trials = int(arr.size)
        if n_trials == 0:
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "statistic": stat,
                    "value": None,
                    "ci_low": None,
                    "ci_high": None,
                    "n_trials": 0,
                }
            )
            continue

        boots = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = rng.choice(arr, size=n_trials, replace=True)
            boots[i] = stat_fn(sample)

        rows.append(
            {
                "method": method,
                "metric": metric,
                "statistic": stat,
                "value": float(stat_fn(arr)),
                "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
                "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))),
                "n_trials": n_trials,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty and out["value"].notna().any():
        out = out.sort_values("value", ascending=False, na_position="last").reset_index(
            drop=True
        )
    return out


def write_trial_metrics_long(dataset_d: pd.DataFrame, experiment_root: str) -> str | None:
    if dataset_d.empty:
        return None
    reports_dir = os.path.join(experiment_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "trial_metrics_long.csv")
    dataset_d.to_csv(out_path, index=False)
    return out_path


def write_bootstrap_aggregates(
    dataset_d: pd.DataFrame,
    experiment_root: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, str]:
    """Write ``bootstrap_{metric}_{stat}.csv`` under ``reports/``."""
    reports_dir = os.path.join(experiment_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    written: dict[str, str] = {}
    for metric in metrics:
        if metric not in dataset_d.columns:
            continue
        for stat in ("mean", "median"):
            table = bootstrap_aggregate_table(dataset_d, metric, stat=stat)
            if table.empty:
                continue
            out_path = os.path.join(reports_dir, f"bootstrap_{metric}_{stat}.csv")
            table[list(BOOTSTRAP_EXPORT_COLUMNS)].to_csv(out_path, index=False)
            written[f"{metric}_{stat}"] = out_path
    return written


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
        mean_stats = stats.get("mean") or {}
        med_stats = stats.get("median") or {}
        if mean_stats.get("mean") is None:
            print(f"  {metric}: no data")
            continue
        print(
            f"  {metric} mean: {mean_stats['mean']:.6f} "
            f"95% CI [{mean_stats['ci_low']:.6f}, {mean_stats['ci_high']:.6f}]"
        )
        if med_stats.get("median") is not None:
            print(
                f"  {metric} median: {med_stats['median']:.6f} "
                f"95% CI [{med_stats['ci_low']:.6f}, {med_stats['ci_high']:.6f}]"
            )


def print_trial_metrics_summary(d_summary: Mapping[str, Any], agg_paths: Mapping[str, str]) -> None:
    print("\n--- Trial metrics (long) ---")
    print(f"Methods: {d_summary.get('n_methods', 0)} — {d_summary.get('methods', [])}")
    for path in agg_paths.values():
        print(f"Bootstrap summary: {path}")


def run_sweep_eval(
    config: Mapping[str, Any],
    experiment_root: str,
    run_log: ExperimentRunLog | None = None,
) -> dict[str, Any] | None:
    """
    Summarize sweeps: single-method JSON, trial metrics CSV, bootstrap aggregates.

    Writes:
    - ``reports/{method}_sweep_summary.json``
    - ``reports/trial_metrics_long.csv`` (all methods in cell)
    - ``reports/bootstrap_{end|life}_{mean|median}.csv``
    - ``reports/performance_analysis.json`` (combined index)
    """
    csv_path = sweep_csv_path(config)
    expected_trials = int(config["nb_runs"])

    dataset_d = load_trial_metrics_long(experiment_root)
    d_summary = dataset_d_summary(dataset_d, expected_trials=expected_trials)
    d_path = write_trial_metrics_long(dataset_d, experiment_root)
    agg_paths = write_bootstrap_aggregates(dataset_d, experiment_root)

    if not os.path.isfile(csv_path):
        if run_log is not None:
            run_log.warn("No sweep CSV for current method — skipping method summary")
        if dataset_d.empty:
            return None
    else:
        method_report = summarize_sweep_csv(csv_path, expected_trials=expected_trials)
        method_report["method"] = method_slug(config)
        method_report["data_seed"] = config.get("seed")
        method_report["rl_seed_base"] = config.get("rl_seed_base")

        missing = method_report.get("missing_trial_ids") or []
        if missing and run_log is not None:
            run_log.warn(
                f"Sweep incomplete: missing trial_ids {missing} "
                f"({method_report['completed_trials']}/{expected_trials} in CSV)"
            )
        write_sweep_report(config, experiment_root, method_report)
        print_sweep_summary(method_report)

    analysis = {
        "trial_metrics_long_csv": d_path,
        "trial_metrics_summary": d_summary,
        "bootstrap_aggregates": agg_paths,
        "current_method": method_slug(config),
    }
    if os.path.isfile(csv_path):
        analysis["current_method_summary"] = os.path.join(
            experiment_root,
            "reports",
            f"{method_slug(config)}_sweep_summary.json",
        )

    reports_dir = os.path.join(experiment_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    analysis_path = os.path.join(reports_dir, "performance_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print_trial_metrics_summary(d_summary, agg_paths)
    print(f"Performance analysis: {analysis_path}")
    if d_path:
        print(f"Trial metrics: {d_path}")

    return analysis
