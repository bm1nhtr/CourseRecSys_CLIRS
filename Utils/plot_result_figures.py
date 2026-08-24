"""
Publication-style figures from Results/: boxplots, CI forests, multi-seed curves.

Usage (repo root)::

    poetry run python Utils/plot_result_figures.py all \\
        --algo ppo --k 2 3 4 5 --lineages CLIRS JCRecFair --metric both

    poetry run python Utils/plot_result_figures.py boxplot --metric end --show-points
    poetry run python Utils/plot_result_figures.py ci --stat mean --ci-style forest
    poetry run python Utils/plot_result_figures.py curves --algo ppo --k 4 --shade-std
    # Also writes curves_compare_clirs_vs_jcrec_fair_{algo}_k{K}.png when ≥2 lineages
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LINEAGE_DIR = {
    "CLIRS": "CLIRS",
    "JCRec": "JCRec",
    "JCRecFair": "JCRecFair",
}

METHOD_PREFIX = {
    "CLIRS": "clirs",
    "JCRec": "jcrec",
    "JCRecFair": "jcrec_fair",
}

TRAINING_NAME_RE = re.compile(
    r"^(?P<method>.+)_(?P<algo>dqn|ppo)_data(?P<data>\d+)_rl(?P<rl>\d+)_k(?P<k>\d+)_training\.txt$"
)


def read_eval_freq(cell: Path) -> int | None:
    """Read EvaluateCallback frequency from cell manifest.json."""
    manifest = cell / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get("eval_freq")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def infer_eval_freq(steps: np.ndarray) -> int | None:
    """Fallback: median positive gap between logged training steps."""
    if steps is None or len(steps) < 2:
        return None
    diffs = np.diff(np.asarray(steps, dtype=float))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    return int(round(float(np.median(diffs))))


def format_eval_freq(freq: int | None) -> str:
    if freq is None:
        return "EvaluateCallback freq: unknown"
    return f"EvaluateCallback every {freq:,} steps"


def style_training_curve_axes(ax: plt.Axes) -> None:
    """Axis labels for training-log curves (EvaluateCallback over time)."""
    ax.set_xlabel("Training steps")
    # Full curve = train-split applicable jobs at each callback; "life" is only the final point.
    ax.set_ylabel("Applicable jobs (train split)")


def load_metric_means_for_cell(
    results_root: Path,
    lineage: str,
    steps: int,
    data_seed: int,
    courses: int,
    k: int,
    algo: str,
) -> dict[str, float]:
    """Mean life/end from trial_metrics_long for one lineage×algo×k cell."""
    path = (
        cell_root(results_root, lineage, steps, data_seed, courses, k)
        / "reports"
        / "trial_metrics_long.csv"
    )
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    method = f"{METHOD_PREFIX[lineage]}_{algo}"
    sub = df[(df["method"] == method) & (df["algorithm"] == algo)]
    if sub.empty:
        sub = df[df["method"] == method]
    if sub.empty:
        return {}
    out: dict[str, float] = {}
    if "life" in sub.columns:
        out["life"] = float(sub["life"].mean())
    if "end" in sub.columns:
        out["end"] = float(sub["end"].mean())
    out["n"] = float(len(sub))
    return out


def mark_curve_endpoint(
    ax: plt.Axes,
    *,
    steps: np.ndarray,
    mean_curve: np.ndarray,
    color: str,
) -> float:
    """Mark final callback point (life); return its y-value."""
    x_last = float(steps[-1])
    y_last = float(mean_curve[-1])
    ax.scatter(
        [x_last],
        [y_last],
        color=color,
        s=42,
        zorder=5,
        edgecolors="0.15",
        linewidths=0.8,
        label="_nolegend_",
    )
    return y_last


def add_end_mean_line(
    ax: plt.Axes,
    *,
    end_mean: float | None,
    color: str,
) -> float | None:
    """Dashed end-mean line only (value goes in the end legend, not on the line)."""
    if end_mean is None:
        return None
    ax.axhline(
        end_mean,
        color=color,
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label="_nolegend_",
    )
    return float(end_mean)


def _legend_above(
    fig: plt.Figure,
    ax: plt.Axes,
    below_legend: Any,
    *,
    handles: Sequence[Any],
    title: str,
    fontsize: float = 7.5,
) -> Any:
    """Stack a new legend just above an existing one (lower-right stack)."""
    if below_legend is None or not handles:
        return None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = below_legend.get_window_extent(renderer).transformed(ax.transAxes.inverted())
    new_legend = ax.legend(
        handles=list(handles),
        title=title,
        loc="lower right",
        bbox_to_anchor=(bbox.x1, bbox.y1 + 0.012),
        fontsize=fontsize,
        title_fontsize=8,
        framealpha=0.92,
        borderaxespad=0.0,
    )
    ax.add_artist(below_legend)
    ax.add_artist(new_legend)
    return new_legend


def add_life_and_end_legends(
    fig: plt.Figure,
    ax: plt.Axes,
    curve_legend: Any,
    life_entries: Sequence[tuple[str, float, str]],
    end_entries: Sequence[tuple[str, float, str]],
) -> None:
    """
    Stack legends above the curve legend (lower-right):
      curve legend → life legend → end legend (top).
    Colors match the endpoint markers / dashed end lines.
    """
    life_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="0.15",
            markersize=8,
            linestyle="None",
            label=f"{(name or 'mean')} = {value:.3f}",
        )
        for name, value, color in life_entries
    ]
    end_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle="--",
            linewidth=1.6,
            label=f"{(name or 'mean')} = {value:.3f}",
        )
        for name, value, color in end_entries
    ]

    life_legend = _legend_above(
        fig,
        ax,
        curve_legend,
        handles=life_handles,
        title="life mean (final eval, train split)",
    )
    base = life_legend if life_legend is not None else curve_legend
    _legend_above(
        fig,
        ax,
        base,
        handles=end_handles,
        title="end mean (single eval, test split)",
    )
    # Keep curve legend visible if life was skipped.
    if life_legend is None and curve_legend is not None:
        ax.add_artist(curve_legend)


# ---------------------------------------------------------------------------
# Paths / IO
# ---------------------------------------------------------------------------


def cell_root(
    results_root: Path,
    lineage: str,
    steps: int,
    data_seed: int,
    courses: int,
    k: int,
) -> Path:
    return (
        results_root
        / LINEAGE_DIR[lineage]
        / f"steps_{steps}"
        / f"data_{data_seed}"
        / f"courses_{courses}"
        / f"k_{k}"
    )


def figures_root(
    results_root: Path,
    steps: int,
    data_seed: int,
    courses: int,
    out: Path | None,
) -> Path:
    if out is not None:
        return out
    return (
        results_root
        / "figures"
        / f"steps_{steps}"
        / f"data_{data_seed}"
        / f"courses_{courses}"
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, path: Path, *, dpi: int, overwrite: bool) -> Path | None:
    if path.exists() and not overwrite:
        print(f"  skip (exists): {path}")
        plt.close(fig)
        return None
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {path}")
    return path


def parse_metrics(metric: str) -> list[str]:
    if metric == "both":
        return ["end", "life"]
    if metric in ("end", "life"):
        return [metric]
    raise ValueError(f"Unknown metric: {metric}")


def load_trial_metrics(
    results_root: Path,
    lineages: Sequence[str],
    steps: int,
    data_seed: int,
    courses: int,
    ks: Sequence[int],
    algos: Sequence[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for lineage in lineages:
        for k in ks:
            path = (
                cell_root(results_root, lineage, steps, data_seed, courses, k)
                / "reports"
                / "trial_metrics_long.csv"
            )
            if not path.is_file():
                print(f"  warn: missing {path}")
                continue
            df = pd.read_csv(path)
            df["lineage"] = lineage
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["algorithm"].isin(algos)]
    return out


def load_bootstrap(
    results_root: Path,
    lineages: Sequence[str],
    steps: int,
    data_seed: int,
    courses: int,
    ks: Sequence[int],
    algos: Sequence[str],
    metrics: Sequence[str],
    stat: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for lineage in lineages:
        for k in ks:
            reports = (
                cell_root(results_root, lineage, steps, data_seed, courses, k)
                / "reports"
            )
            for metric in metrics:
                path = reports / f"bootstrap_{metric}_{stat}.csv"
                if not path.is_file():
                    print(f"  warn: missing {path}")
                    continue
                df = pd.read_csv(path)
                df["lineage"] = lineage
                df["k"] = k
                # filter algorithms via method suffix
                mask = df["method"].str.endswith(tuple(f"_{a}" for a in algos))
                df = df[mask]
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_training_curves_for_cell(
    results_root: Path,
    lineage: str,
    steps: int,
    data_seed: int,
    courses: int,
    k: int,
    algo: str,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {rl_seed: (steps, metric)} for one cell."""
    raw = cell_root(results_root, lineage, steps, data_seed, courses, k) / "raw"
    if not raw.is_dir():
        return {}
    prefix = METHOD_PREFIX[lineage]
    pattern = f"{prefix}_{algo}_data{data_seed}_rl*_k{k}_training.txt"
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(raw.glob(pattern)):
        m = TRAINING_NAME_RE.match(path.name)
        if not m:
            continue
        rl = int(m.group("rl"))
        try:
            data = np.loadtxt(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: cannot read {path}: {exc}")
            continue
        if data.ndim < 2 or data.shape[1] < 2:
            print(f"  warn: bad shape in {path}")
            continue
        curves[rl] = (data[:, 0], data[:, 1])
    return curves


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------


def _method_label(method: str) -> str:
    return method.replace("_", "\n")


def plot_boxplots(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    out_dir: Path,
    dpi: int,
    overwrite: bool,
    show_points: bool,
) -> list[Path]:
    written: list[Path] = []
    if df.empty:
        print("  boxplot: no trial metrics loaded")
        return written

    from matplotlib.patches import Patch

    for metric in metrics:
        for algo in sorted(df["algorithm"].unique()):
            sub = df[df["algorithm"] == algo].copy()
            if sub.empty or metric not in sub.columns:
                continue
            ks = sorted(sub["k"].unique())
            methods = sorted(sub["method"].unique())
            n_k = len(ks)
            fig, axes = plt.subplots(
                1,
                n_k,
                figsize=(max(4.2 * n_k, 9), 6.2),
                sharey=True,
                squeeze=False,
            )
            for ax, k in zip(axes[0], ks):
                chunk = sub[sub["k"] == k]
                data = []
                labels = []
                for method in methods:
                    vals = (
                        chunk.loc[chunk["method"] == method, metric]
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    if len(vals) == 0:
                        continue
                    data.append(vals)
                    labels.append(method)
                if not data:
                    ax.set_visible(False)
                    continue
                bp = ax.boxplot(
                    data,
                    labels=labels,
                    patch_artist=True,
                    showfliers=True,
                    showmeans=True,
                    meanline=False,
                    widths=0.55,
                    meanprops={
                        "marker": "D",
                        "markerfacecolor": "white",
                        "markeredgecolor": "0.15",
                        "markersize": 6.5,
                        "markeredgewidth": 1.0,
                    },
                )
                colors = plt.cm.tab10(np.linspace(0, 0.9, len(data)))
                # Match boxplot defaults: median = thick horizontal line, mean = diamond.
                for med_line in bp["medians"]:
                    med_line.set_color("0.15")
                    med_line.set_linewidth(2.0)
                legend_handles: list[Any] = []
                for vals, method, patch, color in zip(
                    data, labels, bp["boxes"], colors
                ):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                    med = float(np.percentile(vals, 50))
                    mean = float(np.mean(vals))
                    legend_handles.append(
                        Patch(
                            facecolor=color,
                            edgecolor="0.3",
                            alpha=0.55,
                            label=method,
                        )
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="0.15",
                            linewidth=2.0,
                            solid_capstyle="butt",
                            label=f"  Med={med:.3f}",
                        )
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="D",
                            color="w",
                            markerfacecolor="white",
                            markeredgecolor="0.15",
                            markersize=7,
                            linestyle="None",
                            label=f"  Mean={mean:.3f}",
                        )
                    )
                if show_points:
                    for i, vals in enumerate(data, start=1):
                        jitter = np.random.default_rng(0).uniform(
                            -0.12, 0.12, size=len(vals)
                        )
                        ax.scatter(
                            np.full(len(vals), i) + jitter,
                            vals,
                            s=18,
                            color="black",
                            alpha=0.55,
                            zorder=3,
                        )
                ax.legend(
                    handles=legend_handles,
                    loc="best",
                    fontsize=9,
                    framealpha=0.95,
                    borderpad=0.45,
                    labelspacing=0.4,
                    handlelength=1.6,
                    handletextpad=0.6,
                )
                ax.set_title(f"k={k}", fontsize=12)
                ax.tick_params(axis="x", labelrotation=30, labelsize=9)
                ax.tick_params(axis="y", labelsize=9)
                ax.grid(True, axis="y", alpha=0.3)
            axes[0][0].set_ylabel(metric, fontsize=11)
            seed_counts = (
                sub.groupby("method")["rl_seed"].nunique()
                if "rl_seed" in sub.columns
                else sub.groupby("method").size()
            )
            n_min = int(seed_counts.min()) if len(seed_counts) else 0
            n_max = int(seed_counts.max()) if len(seed_counts) else 0
            n_label = (
                f"n={n_min} seeds/method"
                if n_min == n_max
                else f"n={n_min}–{n_max} seeds/method"
            )
            fig.suptitle(
                f"Seed distribution — {metric} — {algo.upper()} ({n_label})",
                fontsize=12,
            )
            fig.tight_layout()
            path = out_dir / f"boxplot_{metric}_{algo}.png"
            saved = save_fig(fig, path, dpi=dpi, overwrite=overwrite)
            if saved:
                written.append(saved)
    return written


def plot_ci(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    stat: str,
    style: str,
    out_dir: Path,
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    written: list[Path] = []
    if df.empty:
        print("  ci: no bootstrap rows loaded")
        return written

    for metric in metrics:
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            continue
        # One panel per algorithm
        algos = sorted(
            {m.rsplit("_", 1)[-1] for m in sub["method"].unique() if "_" in m}
        )
        for algo in algos:
            chunk = sub[sub["method"].str.endswith(f"_{algo}")].copy()
            if chunk.empty:
                continue
            chunk = chunk.sort_values(["k", "method"])
            labels = [
                f"k={int(r.k)} · {r.method}" for r in chunk.itertuples(index=False)
            ]
            y = np.arange(len(chunk))
            values = chunk["value"].to_numpy(dtype=float)
            lo = chunk["ci_low"].to_numpy(dtype=float)
            hi = chunk["ci_high"].to_numpy(dtype=float)
            err_low = values - lo
            err_high = hi - values

            fig_h = max(3.5, 0.42 * len(chunk) + 1.8)
            fig, ax = plt.subplots(figsize=(9.5, fig_h))
            if style == "forest":
                ax.errorbar(
                    values,
                    y,
                    xerr=[err_low, err_high],
                    fmt="o",
                    color="C0",
                    ecolor="0.4",
                    elinewidth=1.4,
                    capsize=3,
                    markersize=5,
                )
                # Annotate CI endpoints (+ mean) so the figure is self-reading.
                x_pad = max((hi - lo).max() * 0.04, (values.max() - values.min()) * 0.02, 0.02)
                for yi, vlo, vmean, vhi in zip(y, lo, values, hi):
                    ax.text(
                        vlo - x_pad,
                        yi,
                        f"{vlo:.3f}",
                        va="center",
                        ha="right",
                        fontsize=7,
                        color="0.25",
                    )
                    ax.text(
                        vhi + x_pad,
                        yi,
                        f"{vhi:.3f}",
                        va="center",
                        ha="left",
                        fontsize=7,
                        color="0.25",
                    )
                    ax.text(
                        vmean,
                        yi + 0.28,
                        f"{vmean:.3f}",
                        va="bottom",
                        ha="center",
                        fontsize=6.5,
                        color="C0",
                    )
                ax.set_yticks(y)
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_xlabel(f"{metric} ({stat}) with 95% CI  [left=ci_low · dot=mean · right=ci_high]")
                span = max(hi.max() - lo.min(), 1e-6)
                ax.set_xlim(lo.min() - 0.12 * span, hi.max() + 0.12 * span)
                ax.invert_yaxis()
            else:  # errorbar vertical grouped-ish
                ax.errorbar(
                    y,
                    values,
                    yerr=[err_low, err_high],
                    fmt="o",
                    color="C0",
                    ecolor="0.4",
                    elinewidth=1.4,
                    capsize=3,
                    markersize=5,
                )
                y_pad = max((hi - lo).max() * 0.04, 0.02)
                for xi, vlo, vmean, vhi in zip(y, lo, values, hi):
                    ax.text(
                        xi,
                        vlo - y_pad,
                        f"{vlo:.3f}",
                        va="top",
                        ha="center",
                        fontsize=7,
                        color="0.25",
                    )
                    ax.text(
                        xi,
                        vhi + y_pad,
                        f"{vhi:.3f}",
                        va="bottom",
                        ha="center",
                        fontsize=7,
                        color="0.25",
                    )
                ax.set_xticks(y)
                ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
                ax.set_ylabel(f"{metric} ({stat}) with 95% CI  [bottom=ci_low · top=ci_high]")

            ax.grid(True, alpha=0.3)
            ax.set_title(f"Bootstrap CI — {metric}/{stat} — {algo.upper()}")
            fig.tight_layout()
            path = out_dir / f"ci_{metric}_{stat}_{style}_{algo}.png"
            saved = save_fig(fig, path, dpi=dpi, overwrite=overwrite)
            if saved:
                written.append(saved)
    return written


def _smooth_series(y: np.ndarray, smooth: int) -> np.ndarray:
    if smooth <= 1:
        return y
    kernel = np.ones(smooth) / smooth
    return np.convolve(y, kernel, mode="same")


def _curves_to_matrix(
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    smooth: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align seed curves on common steps → (steps, matrix[n_seeds, n_steps])."""
    if not curves:
        return None
    step_sets = [set(s.tolist()) for s, _ in curves.values()]
    common = sorted(set.intersection(*step_sets)) if step_sets else []
    if not common:
        return None
    rows = []
    for _, (s, y) in sorted(curves.items()):
        lookup = {int(sv): yv for sv, yv in zip(s, y)}
        yy = np.asarray([lookup[int(t)] for t in common], dtype=float)
        rows.append(_smooth_series(yy, smooth))
    return np.asarray(common, dtype=float), np.vstack(rows)


def plot_training_curves(
    results_root: Path,
    lineages: Sequence[str],
    steps: int,
    data_seed: int,
    courses: int,
    ks: Sequence[int],
    algos: Sequence[str],
    *,
    out_dir: Path,
    dpi: int,
    overwrite: bool,
    shade_std: bool,
    smooth: int,
) -> list[Path]:
    written: list[Path] = []

    for lineage in lineages:
        for algo in algos:
            for k in ks:
                curves = load_training_curves_for_cell(
                    results_root, lineage, steps, data_seed, courses, k, algo
                )
                aligned = _curves_to_matrix(curves, smooth=smooth)
                if aligned is None:
                    print(
                        f"  warn: no training curves for "
                        f"{lineage}/{algo}/k={k}"
                    )
                    continue
                common_arr, mat = aligned
                cell = cell_root(
                    results_root, lineage, steps, data_seed, courses, k
                )
                eval_freq = read_eval_freq(cell) or infer_eval_freq(common_arr)
                metrics = load_metric_means_for_cell(
                    results_root, lineage, steps, data_seed, courses, k, algo
                )
                fig, ax = plt.subplots(figsize=(8, 5.0))
                for yy in mat:
                    ax.plot(
                        common_arr,
                        yy,
                        color="0.65",
                        alpha=0.35,
                        linewidth=1.0,
                        label="_nolegend_",
                    )
                mean = mat.mean(axis=0)
                ax.plot(
                    common_arr,
                    mean,
                    color="C0",
                    linewidth=2.4,
                    label=f"mean (n={len(mat)})",
                )
                if shade_std and len(mat) > 1:
                    std = mat.std(axis=0, ddof=1)
                    ax.fill_between(
                        common_arr,
                        mean - std,
                        mean + std,
                        color="C0",
                        alpha=0.18,
                        label="±1 std",
                    )
                life_final = mark_curve_endpoint(
                    ax, steps=common_arr, mean_curve=mean, color="C0"
                )
                end_mean = add_end_mean_line(
                    ax, end_mean=metrics.get("end"), color="C0"
                )
                ax.set_title(
                    f"Training curves — {lineage} · {algo.upper()} · k={k}\n"
                    f"{format_eval_freq(eval_freq)}"
                )
                style_training_curve_axes(ax)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                curve_legend = ax.legend(loc="lower right", fontsize=8)
                end_entries: list[tuple[str, float, str]] = []
                if end_mean is not None:
                    end_entries.append(("", end_mean, "C0"))
                add_life_and_end_legends(
                    fig,
                    ax,
                    curve_legend,
                    [("", life_final, "C0")],
                    end_entries,
                )
                slug = METHOD_PREFIX[lineage]
                path = out_dir / f"curves_{slug}_{algo}_k{k}.png"
                saved = save_fig(fig, path, dpi=dpi, overwrite=overwrite)
                if saved:
                    written.append(saved)
    return written


def plot_training_curves_compare(
    results_root: Path,
    lineages: Sequence[str],
    steps: int,
    data_seed: int,
    courses: int,
    ks: Sequence[int],
    algos: Sequence[str],
    *,
    out_dir: Path,
    dpi: int,
    overwrite: bool,
    shade_std: bool,
    smooth: int,
    show_seed_curves: bool,
) -> list[Path]:
    """Overlay multiple lineages/methods on one training-curve plot per algo×k."""
    written: list[Path] = []
    if len(lineages) < 2:
        print("  compare-curves: need ≥2 lineages; skipped")
        return written

    colors = [f"C{i}" for i in range(10)]
    for algo in algos:
        for k in ks:
            loaded: list[tuple[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = []
            common_steps: set[int] | None = None
            for lineage in lineages:
                curves = load_training_curves_for_cell(
                    results_root, lineage, steps, data_seed, courses, k, algo
                )
                if not curves:
                    print(
                        f"  warn: compare skip missing {lineage}/{algo}/k={k}"
                    )
                    continue
                loaded.append((lineage, curves))
                step_sets = [
                    {int(x) for x in s.tolist()} for s, _ in curves.values()
                ]
                cell_common = set.intersection(*step_sets) if step_sets else set()
                common_steps = (
                    cell_common
                    if common_steps is None
                    else (common_steps & cell_common)
                )

            if len(loaded) < 2 or not common_steps:
                print(
                    f"  warn: compare needs ≥2 methods with shared steps "
                    f"({algo}/k={k})"
                )
                continue

            common = sorted(common_steps)
            common_arr = np.asarray(common, dtype=float)
            # Prefer manifest eval_freq from first available lineage; else infer.
            eval_freq = None
            for lineage, _ in loaded:
                cell = cell_root(
                    results_root, lineage, steps, data_seed, courses, k
                )
                eval_freq = read_eval_freq(cell)
                if eval_freq is not None:
                    break
            if eval_freq is None:
                eval_freq = infer_eval_freq(common_arr)

            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            labels_short = " vs ".join(lineages)
            life_entries: list[tuple[str, float, str]] = []
            end_entries: list[tuple[str, float, str]] = []

            for idx, (lineage, curves) in enumerate(loaded):
                color = colors[idx % len(colors)]
                rows = []
                for _, (s, y) in sorted(curves.items()):
                    lookup = {int(sv): float(yv) for sv, yv in zip(s, y)}
                    yy = np.asarray([lookup[t] for t in common], dtype=float)
                    yy = _smooth_series(yy, smooth)
                    rows.append(yy)
                    if show_seed_curves:
                        ax.plot(
                            common_arr,
                            yy,
                            color=color,
                            alpha=0.18,
                            linewidth=0.9,
                            label="_nolegend_",
                        )
                mat = np.vstack(rows)
                mean = mat.mean(axis=0)
                method = f"{METHOD_PREFIX[lineage]}_{algo}"
                ax.plot(
                    common_arr,
                    mean,
                    color=color,
                    linewidth=2.4,
                    label=f"{method} mean (n={len(mat)})",
                )
                if shade_std and len(mat) > 1:
                    std = mat.std(axis=0, ddof=1)
                    ax.fill_between(
                        common_arr,
                        mean - std,
                        mean + std,
                        color=color,
                        alpha=0.15,
                        label=f"{method} ±1 std",
                    )
                life_final = mark_curve_endpoint(
                    ax, steps=common_arr, mean_curve=mean, color=color
                )
                life_entries.append((method, life_final, color))
                metrics = load_metric_means_for_cell(
                    results_root, lineage, steps, data_seed, courses, k, algo
                )
                end_mean = add_end_mean_line(
                    ax, end_mean=metrics.get("end"), color=color
                )
                if end_mean is not None:
                    end_entries.append((method, end_mean, color))

            ax.set_title(
                f"Training curves — {algo.upper()} · k={k} ({labels_short})\n"
                f"{format_eval_freq(eval_freq)}"
            )
            style_training_curve_axes(ax)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            curve_legend = ax.legend(loc="lower right", fontsize=7)
            add_life_and_end_legends(
                fig, ax, curve_legend, life_entries, end_entries
            )
            slug = "_vs_".join(METHOD_PREFIX[L] for L in lineages)
            path = out_dir / f"curves_compare_{slug}_{algo}_k{k}.png"
            saved = save_fig(fig, path, dpi=dpi, overwrite=overwrite)
            if saved:
                written.append(saved)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate boxplot / CI / multi-seed training-curve figures from Results/",
    )
    p.add_argument(
        "command",
        choices=["boxplot", "ci", "curves", "all"],
        help="Which figure family to generate",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=_REPO_ROOT / "Results",
        help="Results directory (default: Results/)",
    )
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--courses", type=int, default=100)
    p.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--algo", type=str, nargs="+", default=["dqn", "ppo"])
    p.add_argument(
        "--lineages",
        type=str,
        nargs="+",
        default=["CLIRS", "JCRecFair"],
        choices=list(LINEAGE_DIR),
    )
    p.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["end", "life", "both"],
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: Results/figures/steps_*/data_*/courses_*)",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files",
    )
    # boxplot
    p.add_argument(
        "--show-points",
        action="store_true",
        help="Overlay individual seed points on boxplots",
    )
    # ci
    p.add_argument("--stat", type=str, default="mean", choices=["mean", "median"])
    p.add_argument(
        "--ci-style",
        type=str,
        default="forest",
        choices=["forest", "errorbar"],
    )
    # curves
    p.add_argument(
        "--shade-std",
        action="store_true",
        help="Shade ±1 std around the mean training curve",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Optional moving-average window for training curves (0 = off)",
    )
    p.add_argument(
        "--show-seed-curves",
        action="store_true",
        help="On compare plots, also draw faint per-seed curves (default: mean±std only)",
    )
    p.add_argument(
        "--no-compare-curves",
        action="store_true",
        help="Skip multi-method overlay training-curve plots",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    metrics = parse_metrics(args.metric)
    algos = [a.lower() for a in args.algo]
    for a in algos:
        if a not in ("dqn", "ppo"):
            print(f"Unsupported algo: {a}", file=sys.stderr)
            return 2

    out_dir = figures_root(
        args.results_root, args.steps, args.data_seed, args.courses, args.out
    )
    ensure_dir(out_dir)
    print(f"Output: {out_dir}")

    written: list[Path] = []
    do_box = args.command in ("boxplot", "all")
    do_ci = args.command in ("ci", "all")
    do_curves = args.command in ("curves", "all")

    if do_box:
        print("\n[boxplot]")
        trials = load_trial_metrics(
            args.results_root,
            args.lineages,
            args.steps,
            args.data_seed,
            args.courses,
            args.k,
            algos,
        )
        written += plot_boxplots(
            trials,
            metrics=metrics,
            out_dir=out_dir,
            dpi=args.dpi,
            overwrite=args.overwrite,
            show_points=args.show_points,
        )

    if do_ci:
        print("\n[ci]")
        boot = load_bootstrap(
            args.results_root,
            args.lineages,
            args.steps,
            args.data_seed,
            args.courses,
            args.k,
            algos,
            metrics,
            args.stat,
        )
        written += plot_ci(
            boot,
            metrics=metrics,
            stat=args.stat,
            style=args.ci_style,
            out_dir=out_dir,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )

    if do_curves:
        print("\n[curves]")
        written += plot_training_curves(
            args.results_root,
            args.lineages,
            args.steps,
            args.data_seed,
            args.courses,
            args.k,
            algos,
            out_dir=out_dir,
            dpi=args.dpi,
            overwrite=args.overwrite,
            shade_std=args.shade_std,
            smooth=args.smooth,
        )
        if not args.no_compare_curves and len(args.lineages) >= 2:
            print("\n[curves-compare]")
            written += plot_training_curves_compare(
                args.results_root,
                args.lineages,
                args.steps,
                args.data_seed,
                args.courses,
                args.k,
                algos,
                out_dir=out_dir,
                dpi=args.dpi,
                overwrite=args.overwrite,
                shade_std=args.shade_std,
                smooth=args.smooth,
                show_seed_curves=args.show_seed_curves,
            )

    print(f"\nDone. {len(written)} figure(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
