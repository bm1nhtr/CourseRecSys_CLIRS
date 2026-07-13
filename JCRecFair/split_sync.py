"""Pin JCRecFair to the CLIRS experiment cell train/test split."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np

from Utils.results_paths import experiment_root


class ClirsSplitNotFoundError(SystemExit):
    """CLIRS cell has no split_indices.json — run CLIRS first."""


def clirs_experiment_root(config: Mapping[str, Any]) -> str:
    cfg = dict(config)
    cfg["pipeline"] = "clirs"
    cfg["results_lineage"] = "CLIRS"
    cfg["use_clustering"] = True
    return experiment_root(cfg)


def clirs_split_indices_path(config: Mapping[str, Any]) -> str:
    return os.path.join(clirs_experiment_root(config), "split_indices.json")


def load_clirs_split_indices(config: Mapping[str, Any]) -> dict[str, list[int]] | None:
    path = clirs_split_indices_path(config)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    if "train_indices" not in saved or "test_indices" not in saved:
        return None
    return {
        "train_indices": [int(i) for i in saved["train_indices"]],
        "test_indices": [int(i) for i in saved["test_indices"]],
    }


def _clirs_split_missing_message(config: Mapping[str, Any], path: str) -> str:
    clirs_root = clirs_experiment_root(config)
    return (
        "jcrec_fair requires CLIRS split_indices.json (run CLIRS first).\n"
        f"  Expected: {path}\n"
        f"  CLIRS cell: {clirs_root}\n"
        "  Run: poetry run python pipelines/run_clirs_pipeline.py --Config Config/run.json\n"
        "  Or full orchestration with clirs before jcrec_fair:\n"
        "    poetry run python pipelines/run_pipeline.py --Config Config/run.json"
    )


def require_clirs_split_file(config: Mapping[str, Any]) -> str:
    """Fail fast if CLIRS ``split_indices.json`` is missing or invalid."""
    path = clirs_split_indices_path(config)
    if load_clirs_split_indices(config) is None:
        raise ClirsSplitNotFoundError(_clirs_split_missing_message(config, path))
    return path


def publish_clirs_split_artifact(config: Mapping[str, Any], dataset: Any) -> str:
    """
    Persist CLIRS train/test split for downstream jcrec_fair.

    Called by the CLIRS pipeline after dataset load so ``split_indices.json``
    exists even if RL trials fail later.
    """
    path = clirs_split_indices_path(config)
    if load_clirs_split_indices(config) is not None:
        return path

    train_indices = getattr(dataset, "train_indices", None)
    test_indices = getattr(dataset, "test_indices", None)
    if train_indices is None or test_indices is None or len(train_indices) == 0:
        raise SystemExit(
            "CLIRS dataset has no train/test split — cannot publish split_indices.json"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "data_seed": config.get("seed"),
        "train_indices": [int(i) for i in train_indices],
        "test_indices": [int(i) for i in test_indices],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def ensure_clirs_split(dataset: Any, config: Mapping[str, Any]) -> str:
    """Apply train/test indices from CLIRS ``split_indices.json``."""
    path = require_clirs_split_file(config)
    saved = load_clirs_split_indices(config)
    assert saved is not None

    n_learners = len(dataset.learners)
    for name in ("train_indices", "test_indices"):
        for idx in saved[name]:
            if idx < 0 or idx >= n_learners:
                raise SystemExit(
                    f"CLIRS split_indices.json invalid: {name} index {idx} "
                    f"out of range for n_learners={n_learners}"
                )

    dataset.train_indices = np.array(saved["train_indices"], dtype=int)
    dataset.test_indices = np.array(saved["test_indices"], dtype=int)
    return path
