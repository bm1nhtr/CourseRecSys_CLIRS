"""Central path layout and artifact naming for experiment Results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

SWEEP_CSV_COLUMNS = (
    "trial_id",
    "data_seed",
    "rl_seed",
    "method",
    "algorithm",
    "total_steps",
    "nb_courses",
    "k",
    "threshold",
    "clustering_reward_shaping",
    "evaluation_split",  # test (CLIRS hold-out) | all_learners (JCRec, no hold-out)
    "life",  # train-split proxy — see METRIC_DEFINITIONS["life"]
    "end",  # primary report metric — see METRIC_DEFINITIONS["end"]
    "original_applicable_jobs",
    "train_size",  # CLIRS: train split n; JCRec RL: n (full pool, same as test_size)
    "test_size",  # CLIRS: held-out test n; JCRec: n (full pool final eval)
)


def repo_root() -> Path:
    """Repository root (parent of Utils/)."""
    return Path(__file__).resolve().parents[1]


def method_slug(config: Mapping[str, Any]) -> str:
    """e.g. clirs_dqn, jcrec_fair_dqn, jcrec_dqn."""
    pipeline = config.get("pipeline")
    algorithm = config.get("model", "dqn")
    if pipeline == "jcrec":
        return f"jcrec_{algorithm}"
    if pipeline == "jcrec_fair":
        return f"jcrec_fair_{algorithm}"
    prefix = "clirs" if config.get("use_clustering") else "baseline"
    return f"{prefix}_{algorithm}"


def rl_seed_for_trial(config: Mapping[str, Any], trial_id: int) -> int:
    """``rl_seed = rl_seed_base + trial_id`` (see manifest ``rl_seed_policy``)."""
    base = int(config.get("rl_seed_base", config.get("seed", 42)))
    return base + int(trial_id)


def completed_trial_ids(config: Mapping[str, Any]) -> set[int]:
    """Trial ids already present in the sweep CSV."""
    path = sweep_csv_path(config)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return set()
    completed: set[int] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "trial_id" not in reader.fieldnames:
            return set()
        for row in reader:
            try:
                completed.add(int(row["trial_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def courses_dir_slug(config: Mapping[str, Any]) -> str:
    """e.g. courses_100, courses_all when nb_courses is -1."""
    nb_courses = config.get("nb_courses", -1)
    if nb_courses == -1:
        return "courses_all"
    return f"courses_{int(nb_courses)}"


def k_dir_slug(config: Mapping[str, Any]) -> str:
    """e.g. k_2 — one Complete Algorithm cell per recommendation horizon."""
    return f"k_{int(config.get('k', 1))}"


def experiment_root(config: Mapping[str, Any]) -> str:
    """
    Results/{lineage}/steps_{total_steps}/data_{data_seed}/courses_{nb}/k_{k}/

    ``config["results_path"]`` must already be absolute (see load_config).
    """
    lineage = config.get("results_lineage", "CLIRS")
    total_steps = config.get("total_steps", 0)
    data_seed = config.get("seed", 42)
    base = config.get("results_path", os.path.join(str(repo_root()), "Results"))
    return os.path.normpath(
        os.path.join(
            base,
            lineage,
            f"steps_{total_steps}",
            f"data_{data_seed}",
            courses_dir_slug(config),
            k_dir_slug(config),
        )
    )


def compare_root(config: Mapping[str, Any]) -> str:
    """Cross-lineage compare cell: ``Results/compare/steps_*/.../k_*/{algo}/``."""
    total_steps = config.get("total_steps", 0)
    data_seed = config.get("seed", 42)
    algorithm = str(config.get("model", "dqn")).lower()
    base = config.get("results_path", os.path.join(str(repo_root()), "Results"))
    return os.path.normpath(
        os.path.join(
            base,
            "compare",
            f"steps_{total_steps}",
            f"data_{data_seed}",
            courses_dir_slug(config),
            k_dir_slug(config),
            algorithm,
        )
    )


def compare_pair_dir(config: Mapping[str, Any], pair_slug: str) -> str:
    """One cross-lineage pair under the compare cell, e.g. ``.../clirs_vs_jcrec_fair/``."""
    return os.path.join(compare_root(config), pair_slug)


def experiment_log_path(config: Mapping[str, Any]) -> str:
    """Per-experiment console log: ``{experiment_root}/run.log``."""
    return os.path.join(experiment_root(config), "run.log")


def experiment_dirs(config: Mapping[str, Any], save_raw: bool | None = None) -> dict[str, str]:
    """Return standard subdirectories under the experiment root."""
    root = experiment_root(config)
    if save_raw is None:
        save_raw = bool(config.get("save_raw", True))
    dirs = {
        "root": root,
        "sweeps": os.path.join(root, "sweeps"),
        "reports": os.path.join(root, "reports"),
        "plots": os.path.join(root, "plots"),
        "clustering_plots": os.path.join(root, "plots", "clustering"),
    }
    if save_raw:
        dirs["raw"] = os.path.join(root, "raw")
    return dirs


def ensure_experiment_dirs(
    config: Mapping[str, Any],
    *,
    save_raw: bool | None = None,
    write_manifest: bool = False,
) -> dict[str, str]:
    """Create experiment directory tree.

    Manifest writing is handled by ``CompleteAlgorithmStage`` after the SB3
    model exists (see ``Utils.complete_algorithm``).
    """
    dirs = experiment_dirs(config, save_raw=save_raw)
    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)
    if write_manifest:
        import warnings

        warnings.warn(
            "ensure_experiment_dirs(write_manifest=True) is deprecated; "
            "use CompleteAlgorithmStage.ensure() after model init.",
            DeprecationWarning,
            stacklevel=2,
        )
        manifest_path = os.path.join(dirs["root"], "manifest.json")
        if not os.path.exists(manifest_path):
            manifest = {
                "results_lineage": config.get("results_lineage", "CLIRS"),
                "data_seed": config.get("seed"),
                "total_steps": config.get("total_steps"),
                "nb_courses": config.get("nb_courses"),
                "model": config.get("model"),
                "k": config.get("k"),
                "threshold": config.get("threshold"),
                "use_clustering_in_config": config.get("use_clustering"),
                "eval_freq": config.get("eval_freq"),
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
    return dirs


def sweep_csv_path(config: Mapping[str, Any]) -> str:
    """sweeps/{method}_data{data_seed}.csv"""
    data_seed = config.get("seed", 42)
    filename = f"{method_slug(config)}_data{data_seed}.csv"
    return os.path.join(experiment_dirs(config)["sweeps"], filename)


def trial_artifact_paths(config: Mapping[str, Any], trial_id: int) -> dict[str, str]:
    """Per-trial raw paths under raw/ when save_raw is enabled."""
    dirs = experiment_dirs(config)
    raw_dir = dirs.get("raw", os.path.join(dirs["root"], "raw"))
    stem = (
        f"{method_slug(config)}_data{config.get('seed', 42)}"
        f"_rl{rl_seed_for_trial(config, trial_id)}_k{config.get('k', 1)}"
    )
    return {
        "training": os.path.join(raw_dir, f"{stem}_training.txt"),
        "eval": os.path.join(raw_dir, f"{stem}_eval.json"),
    }


def upsert_trial_csv_row(config: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """Write one trial row, replacing any existing row with the same ``trial_id``."""
    path = sweep_csv_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trial_id = int(row["trial_id"])
    rows: dict[int, dict[str, Any]] = {}
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for existing in reader:
                    try:
                        rows[int(existing["trial_id"])] = existing
                    except (KeyError, TypeError, ValueError):
                        continue
    rows[trial_id] = {col: row.get(col) for col in SWEEP_CSV_COLUMNS}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for tid in sorted(rows):
            writer.writerow(rows[tid])
    return path


def append_trial_csv_row(config: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """Upsert one trial row into the sweep CSV."""
    return upsert_trial_csv_row(config, row)


def read_training_life_proxy(training_path: str) -> float | None:
    """Last train-split mean jobs from a training log (metric ``life``).

    See ``Utils.complete_algorithm.METRIC_DEFINITIONS["life"]``.
    """
    if not training_path or not os.path.isfile(training_path):
        return None
    last_line = None
    with open(training_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        return None
    parts = last_line.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None
