"""
Cross-lineage comparison only — all pairs under ``Results/compare/``.

Layout::

    Results/compare/steps_*/data_*/courses_*/k_*/{algo}/
    ├── cross_lineage_index.json
    ├── clirs_vs_jcrec_fair/
    └── clirs_vs_jcrec_author/

Each algorithm (dqn, ppo) has its own compare folder — runs do not overwrite.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLIRS_SCRIPTS = _REPO_ROOT / "CLIRS" / "Scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.sweep_eval import (  # noqa: E402
    DEFAULT_METRICS,
    bootstrap_aggregate_table,
    summarize_sweep_csv,
)
from Utils.results_paths import (
    compare_pair_dir,
    compare_root,
    experiment_root,
    method_slug,
    sweep_csv_path,
)

RL_ALGORITHMS = frozenset({"dqn", "ppo"})

PAIRWISE_EXPORT_COLUMNS = (
    "method_a",
    "method_b",
    "metric",
    "statistic",
    "value",
    "ci_low",
    "ci_high",
    "n_pairs",
    "n_wins",
    "n_ties",
    "n_losses",
    "evaluation_split_a",
    "evaluation_split_b",
)


def _load_config(config_path: str) -> dict[str, Any]:
    path = _CLIRS_SCRIPTS / "load_config.py"
    spec = importlib.util.spec_from_file_location("clirs_load_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load load_config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config(config_path)


def config_for_pipeline(config: Mapping[str, Any], pipeline: str) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    if pipeline == "clirs":
        cfg["pipeline"] = "clirs"
        cfg["results_lineage"] = cfg.get("results_lineage", "CLIRS")
        cfg["use_clustering"] = cfg.get("use_clustering", True)
    elif pipeline == "jcrec":
        cfg["pipeline"] = "jcrec"
        cfg["results_lineage"] = cfg.get("jcrec_results_lineage", "JCRec")
        cfg["use_clustering"] = False
    elif pipeline == "jcrec_fair":
        cfg["pipeline"] = "jcrec_fair"
        cfg["results_lineage"] = cfg.get("jcrec_fair_results_lineage", "JCRecFair")
        cfg["use_clustering"] = False
    else:
        raise ValueError(f"Unknown pipeline {pipeline!r}")
    return cfg


def load_lineage_sweep(config: Mapping[str, Any], pipeline: str) -> pd.DataFrame | None:
    cfg = config_for_pipeline(config, pipeline)
    path = sweep_csv_path(cfg)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df = df.copy()
    df["source_lineage"] = cfg["results_lineage"]
    df["source_csv"] = os.path.basename(path)
    if "method" not in df.columns or df["method"].isna().all():
        df["method"] = method_slug(cfg)
    return df


def _ensure_trial_wall_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``trial_wall_minutes`` exists (migrate legacy seconds if needed)."""
    out = df.copy()
    if "trial_wall_minutes" not in out.columns:
        out["trial_wall_minutes"] = np.nan
    if "trial_wall_seconds" in out.columns:
        minutes = pd.to_numeric(out["trial_wall_minutes"], errors="coerce")
        seconds = pd.to_numeric(out["trial_wall_seconds"], errors="coerce")
        fill = minutes.isna() & seconds.notna()
        out.loc[fill, "trial_wall_minutes"] = (seconds[fill] / 60.0).round(3)
    return out


def build_compare_trial_metrics(
    clirs_df: pd.DataFrame,
    jcrec_df: pd.DataFrame,
) -> pd.DataFrame:
    """Stacked per-trial rows for both lineages (includes ``trial_wall_minutes``)."""
    return pd.concat(
        [_ensure_trial_wall_minutes(clirs_df), _ensure_trial_wall_minutes(jcrec_df)],
        ignore_index=True,
    )


def _paired_metric_values(
    clirs_df: pd.DataFrame,
    jcrec_df: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    left = clirs_df[["trial_id", metric, "method", "evaluation_split"]].copy()
    right = jcrec_df[["trial_id", metric, "method", "evaluation_split"]].copy()
    merged = left.merge(right, on="trial_id", suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return np.array([], dtype=float), np.array([], dtype=float), merged
    a = pd.to_numeric(merged[f"{metric}_a"], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(merged[f"{metric}_b"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    return a[valid], b[valid], merged.loc[valid]


def _paired_scores(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    return np.where(
        values_a > values_b,
        1.0,
        np.where(values_a == values_b, 0.5, 0.0),
    )


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = float(sample.mean())
    point = float(values.mean())
    return (
        point,
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def paired_win_rate_stats(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    scores = _paired_scores(values_a, values_b)
    n_pairs = int(scores.size)
    if n_pairs == 0:
        return {
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "n_pairs": 0,
            "n_wins": 0,
            "n_ties": 0,
            "n_losses": 0,
        }
    n_wins = int(np.sum(values_a > values_b))
    n_ties = int(np.sum(values_a == values_b))
    n_losses = int(np.sum(values_a < values_b))
    value, ci_low, ci_high = _bootstrap_ci(scores, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_pairs": n_pairs,
        "n_wins": n_wins,
        "n_ties": n_ties,
        "n_losses": n_losses,
    }


def pbpt_stats(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    n_pairs = int(min(values_a.size, values_b.size))
    if n_pairs == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "n_pairs": 0}
    sorted_a = np.sort(values_a)
    sorted_b = np.sort(values_b)
    pts = np.searchsorted(sorted_b, sorted_a, side="left") / sorted_b.size
    mu = float(pts.mean())
    if pts.size <= 1:
        return {"value": mu, "ci_low": None, "ci_high": None, "n_pairs": n_pairs}
    sigma = float(pts.std(ddof=1))
    tstar = float(student_t.ppf(1 - alpha / 2, df=pts.size - 1))
    se = sigma / np.sqrt(pts.size)
    return {
        "value": mu,
        "ci_low": mu - tstar * se,
        "ci_high": mu + tstar * se,
        "n_pairs": n_pairs,
    }


def pairwise_comparison_table(
    clirs_df: pd.DataFrame,
    jcrec_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    method_a: str,
    method_b: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_split_a = (
        str(clirs_df["evaluation_split"].iloc[0])
        if "evaluation_split" in clirs_df.columns and not clirs_df.empty
        else None
    )
    eval_split_b = (
        str(jcrec_df["evaluation_split"].iloc[0])
        if "evaluation_split" in jcrec_df.columns and not jcrec_df.empty
        else None
    )

    for metric in metrics:
        values_a, values_b, _ = _paired_metric_values(clirs_df, jcrec_df, metric)
        win = paired_win_rate_stats(values_a, values_b)
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "statistic": "win_rate",
                "value": win["value"],
                "ci_low": win["ci_low"],
                "ci_high": win["ci_high"],
                "n_pairs": win["n_pairs"],
                "n_wins": win["n_wins"],
                "n_ties": win["n_ties"],
                "n_losses": win["n_losses"],
                "evaluation_split_a": eval_split_a,
                "evaluation_split_b": eval_split_b,
            }
        )
        pbpt = pbpt_stats(values_a, values_b)
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "statistic": "pbpt",
                "value": pbpt["value"],
                "ci_low": pbpt["ci_low"],
                "ci_high": pbpt["ci_high"],
                "n_pairs": pbpt["n_pairs"],
                "n_wins": None,
                "n_ties": None,
                "n_losses": None,
                "evaluation_split_a": eval_split_a,
                "evaluation_split_b": eval_split_b,
            }
        )
    return pd.DataFrame(rows, columns=list(PAIRWISE_EXPORT_COLUMNS))


def write_ecdf_plot(
    clirs_df: pd.DataFrame,
    jcrec_df: pd.DataFrame,
    *,
    metric: str,
    method_a: str,
    method_b: str,
    out_path: str,
) -> str | None:
    series: list[tuple[str, pd.Series]] = []
    for label, df in ((method_a, clirs_df), (method_b, jcrec_df)):
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not values.empty:
            series.append((label, values.sort_values()))
    if len(series) < 2:
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, values in series:
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values.to_numpy(), y, where="post", label=label, linewidth=2)

    ax.set_xlabel(metric)
    ax.set_ylabel("ECDF")
    ax.set_title(f"ECDF — {metric} ({method_a} vs {method_b})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_final_report(
    *,
    compare_dir: str,
    config: Mapping[str, Any],
    method_a: str,
    method_b: str,
    compare_csv: str,
    pairwise_csv: str,
    bootstrap_paths: Mapping[str, str],
    plot_paths: Mapping[str, str],
    clirs_root: str,
    baseline_root: str,
    report_filename: str = "clirs_vs_jcrec_report.md",
    title: str = "CLIRS vs JCRec comparison report",
    protocol_rows: list[tuple[str, str, str]] | None = None,
    protocol_note: str = "",
) -> str:
    algorithm = config.get("model", "dqn")
    if protocol_rows is None:
        protocol_rows = [
            ("CLIRS", "test", "70/30 hold-out test split"),
            ("JCRec", "all_learners", "Full learner pool (author protocol)"),
        ]
    lines = [
        f"# {title}",
        "",
        "## Cell",
        "",
        f"- Algorithm: `{algorithm}`",
        f"- Steps: `{config.get('total_steps')}`",
        f"- Data seed: `{config.get('seed')}`",
        f"- Courses: `{config.get('nb_courses')}`",
        f"- k: `{config.get('k')}`",
        f"- Compare dir: `{compare_dir}`",
        "",
        "## Lineage roots",
        "",
        f"- CLIRS: `{clirs_root}`",
        f"- Baseline: `{baseline_root}`",
        "",
        "## Protocol",
        "",
        "| Method | `evaluation_split` | `end` population |",
        "|--------|--------------------|--------------------|",
    ]
    for name, split, population in protocol_rows:
        lines.append(f"| {name} | `{split}` | {population} |")
    if protocol_note:
        lines.extend(["", protocol_note, ""])
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Trial metrics: `{compare_csv}`",
            f"- Pairwise: `{pairwise_csv}`",
        ]
    )
    for name, path in sorted(bootstrap_paths.items()):
        lines.append(f"- Bootstrap `{name}`: `{path}`")
    for name, path in sorted(plot_paths.items()):
        lines.append(f"- Plot `{name}`: `{path}`")

    if os.path.isfile(pairwise_csv):
        lines.extend(["", "## Pairwise summary", ""])
        table = pd.read_csv(pairwise_csv)
        for metric in table["metric"].dropna().unique():
            lines.append(f"### `{metric}`")
            lines.append("")
            lines.append("| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |")
            lines.append("|-----------|-------|--------|---------|---------|------|------|--------|")
            subset = table[table["metric"] == metric]
            for _, row in subset.iterrows():
                wins = row.get("n_wins")
                ties = row.get("n_ties")
                losses = row.get("n_losses")
                lines.append(
                    f"| {row['statistic']} | {row['value']} | {row['ci_low']} | "
                    f"{row['ci_high']} | {row['n_pairs']} | {wins} | {ties} | {losses} |"
                )
            lines.append("")

    out_path = os.path.join(compare_dir, report_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _run_lineage_compare(
    config: Mapping[str, Any],
    *,
    baseline_pipeline: str,
    compare_dir: str,
    report_filename: str,
    title: str,
    protocol_rows: list[tuple[str, str, str]],
    protocol_note: str,
    print_label: str,
) -> dict[str, Any] | None:
    algorithm = str(config.get("model", "dqn")).lower()
    if algorithm not in RL_ALGORITHMS:
        print(
            f"{print_label} skipped: algorithm {algorithm!r} "
            f"(only {sorted(RL_ALGORITHMS)})."
        )
        return None

    clirs_cfg = config_for_pipeline(config, "clirs")
    baseline_cfg = config_for_pipeline(config, baseline_pipeline)
    clirs_root = experiment_root(clirs_cfg)
    baseline_root = experiment_root(baseline_cfg)
    clirs_csv = sweep_csv_path(clirs_cfg)
    baseline_csv = sweep_csv_path(baseline_cfg)

    if not os.path.isfile(clirs_csv):
        print(f"{print_label} skipped: missing CLIRS sweep {clirs_csv}")
        return None
    if not os.path.isfile(baseline_csv):
        print(f"{print_label} skipped: missing baseline sweep {baseline_csv}")
        return None

    clirs_df = load_lineage_sweep(config, "clirs")
    baseline_df = load_lineage_sweep(config, baseline_pipeline)
    if clirs_df is None or baseline_df is None:
        print(f"{print_label} skipped: empty sweep data.")
        return None

    method_a = method_slug(clirs_cfg)
    method_b = method_slug(baseline_cfg)
    plots_dir = os.path.join(compare_dir, "plots")
    os.makedirs(compare_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    compare_df = build_compare_trial_metrics(clirs_df, baseline_df)
    compare_csv = os.path.join(compare_dir, "compare_trial_metrics.csv")
    compare_df.to_csv(compare_csv, index=False)

    bootstrap_paths: dict[str, str] = {}
    for metric in DEFAULT_METRICS:
        if metric not in compare_df.columns:
            continue
        for stat in ("mean", "median"):
            table = bootstrap_aggregate_table(compare_df, metric, stat=stat)
            if table.empty:
                continue
            out_path = os.path.join(compare_dir, f"bootstrap_{metric}_{stat}.csv")
            table.to_csv(out_path, index=False)
            bootstrap_paths[f"{metric}_{stat}"] = out_path

    pairwise_df = pairwise_comparison_table(
        clirs_df,
        baseline_df,
        method_a=method_a,
        method_b=method_b,
    )
    pairwise_csv = os.path.join(compare_dir, "pairwise_comparison.csv")
    pairwise_df.to_csv(pairwise_csv, index=False)

    plot_paths: dict[str, str] = {}
    for metric in DEFAULT_METRICS:
        plot_path = os.path.join(plots_dir, f"ecdf_{metric}_{method_a}_vs_{method_b}.png")
        written = write_ecdf_plot(
            clirs_df,
            baseline_df,
            metric=metric,
            method_a=method_a,
            method_b=method_b,
            out_path=plot_path,
        )
        if written:
            plot_paths[metric] = written

    expected_trials = int(config.get("nb_runs", 0))
    analysis = {
        "algorithm": algorithm,
        "method_a": method_a,
        "method_b": method_b,
        "baseline_pipeline": baseline_pipeline,
        "compare_dir": compare_dir,
        "compare_trial_metrics_csv": compare_csv,
        "pairwise_comparison_csv": pairwise_csv,
        "bootstrap_aggregates": bootstrap_paths,
        "plots": plot_paths,
        "clirs_summary": summarize_sweep_csv(clirs_csv, expected_trials=expected_trials),
        "baseline_summary": summarize_sweep_csv(
            baseline_csv, expected_trials=expected_trials
        ),
        "protocol_note": protocol_note,
    }
    analysis_path = os.path.join(compare_dir, "compare_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    report_path = write_final_report(
        compare_dir=compare_dir,
        config=config,
        method_a=method_a,
        method_b=method_b,
        compare_csv=compare_csv,
        pairwise_csv=pairwise_csv,
        bootstrap_paths=bootstrap_paths,
        plot_paths=plot_paths,
        clirs_root=clirs_root,
        baseline_root=baseline_root,
        report_filename=report_filename,
        title=title,
        protocol_rows=protocol_rows,
        protocol_note=protocol_note,
    )

    print(f"\n--- {print_label} ---")
    print(f"Compare dir: {compare_dir}")
    print(f"Trial metrics: {compare_csv}")
    print(f"Pairwise: {pairwise_csv}")
    print(f"Report: {report_path}")
    for path in plot_paths.values():
        print(f"ECDF plot: {path}")

    analysis["report_md"] = report_path
    analysis["compare_analysis_json"] = analysis_path
    return analysis


def run_fair_cross_lineage_eval(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Cross-lineage: CLIRS vs JCRecFair (both ``end`` on test split)."""
    return _run_lineage_compare(
        config,
        baseline_pipeline="jcrec_fair",
        compare_dir=compare_pair_dir(config, "clirs_vs_jcrec_fair"),
        report_filename="report.md",
        title="CLIRS vs JCRec fair (test split)",
        protocol_rows=[
            ("CLIRS", "test", "70/30 hold-out test split"),
            ("JCRecFair", "test", "Same hold-out test split (JCRec algo + CLIRS split)"),
        ],
        protocol_note=(
            "Both methods report `end` on the **same test_indices** "
            "(paired trials, same `data_seed` / `rl_seed` policy). "
            "Primary thesis comparison."
        ),
        print_label="Cross-lineage: CLIRS vs JCRecFair",
    )


def run_author_cross_lineage_eval(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Cross-lineage: CLIRS test vs JCRec author all_learners (replication)."""
    return _run_lineage_compare(
        config,
        baseline_pipeline="jcrec",
        compare_dir=compare_pair_dir(config, "clirs_vs_jcrec_author"),
        report_filename="report.md",
        title="CLIRS vs JCRec author reproduction",
        protocol_rows=[
            ("CLIRS", "test", "70/30 hold-out test split"),
            ("JCRec", "all_learners", "Full learner pool (author protocol)"),
        ],
        protocol_note=(
            "Different `end` populations — author replication context only."
        ),
        print_label="Cross-lineage: CLIRS vs JCRec (author)",
    )


def run_cross_lineage_eval(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Run all cross-lineage compares under ``Results/compare/...``."""
    return run_cross_lineage_compares(config)


def run_cross_lineage_compares(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Run every configured cross-lineage pair; write index at compare cell root."""
    fair = run_fair_cross_lineage_eval(config)
    author = run_author_cross_lineage_eval(config)
    pairs = {k: v for k, v in (("clirs_vs_jcrec_fair", fair), ("clirs_vs_jcrec_author", author)) if v}

    if not pairs:
        return None

    cell_root = compare_root(config)
    os.makedirs(cell_root, exist_ok=True)
    index = {
        "compare_cell": cell_root,
        "algorithm": str(config.get("model", "dqn")).lower(),
        "pairs": {
            slug: {
                "compare_dir": result["compare_dir"],
                "method_a": result["method_a"],
                "method_b": result["method_b"],
                "report_md": result.get("report_md"),
                "pairwise_comparison_csv": result.get("pairwise_comparison_csv"),
            }
            for slug, result in pairs.items()
        },
    }
    index_path = os.path.join(cell_root, "cross_lineage_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\nCross-lineage index: {index_path}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-lineage CLIRS comparison reports."
    )
    parser.add_argument("--Config", default=r"Config/run.json")
    parser.add_argument(
        "--fair-only",
        action="store_true",
        help="Run only CLIRS vs JCRecFair (primary)",
    )
    parser.add_argument(
        "--author-only",
        action="store_true",
        help="Run only CLIRS vs JCRec author reproduction",
    )
    args = parser.parse_args()
    config = _load_config(args.Config)

    if args.fair_only:
        result = run_fair_cross_lineage_eval(config)
    elif args.author_only:
        result = run_author_cross_lineage_eval(config)
    else:
        result = run_cross_lineage_compares(config)

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
