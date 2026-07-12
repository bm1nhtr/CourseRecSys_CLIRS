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


def ensure_clirs_split(dataset: Any, config: Mapping[str, Any]) -> str:
    """Apply train/test indices from CLIRS ``split_indices.json``."""
    path = clirs_split_indices_path(config)
    saved = load_clirs_split_indices(config)
    if saved is None:
        clirs_root = clirs_experiment_root(config)
        raise ClirsSplitNotFoundError(
            "jcrec_fair requires CLIRS split_indices.json (run CLIRS first).\n"
            f"  Expected: {path}\n"
            f"  CLIRS cell: {clirs_root}\n"
            "  Run: poetry run python pipelines/run_clirs_pipeline.py --Config Config/run.json\n"
            "  Or full orchestration: poetry run python pipelines/run_pipeline.py --Config Config/run.json"
        )

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
