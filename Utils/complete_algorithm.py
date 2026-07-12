"""
Complete Algorithm stage — freeze, document, and validate one experiment cell.

Workflow (Jordan et al. eval pipeline — Stage 3)
-------------------------------------------------
1. First trial in a cell: snapshot config + SB3 hyperparameters → ``manifest.json``.
2. Same file records metric definitions (``life`` vs ``end``) for humans and tools.
3. ``split_indices.json`` pins the 70/30 learner split for this ``data_seed``.
4. Later trials / later runs: **validate only** — never overwrite the manifest (rule A.2).

Experiment cell path::

    Results/{lineage}/steps_{total_steps}/data_{data_seed}/courses_{nb_courses}/

If you change algorithm, steps, k, clustering, etc. and reuse that folder, the pipeline
raises ``ManifestValidationError`` instead of mixing two experiments in one CSV.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Mapping

from Utils.results_paths import method_slug

# ---------------------------------------------------------------------------
# Metric contract (also copied into manifest.json → "metrics" for offline reading)
# ---------------------------------------------------------------------------
# ``end``  = test split, after training  → PRIMARY metric for papers / compare.py
# ``life`` = train split, last callback  → training proxy only (optimistic)
METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "end": {
        "name": "end",
        "csv_column": "end",
        "eval_json_field": "end",
        "aliases": ["new_applicable_jobs"],
        "summary": "Primary report metric (test split, after training).",
        "definition": (
            "Mean applicable jobs per learner on the held-out test split after "
            "the RL policy recommends k courses. Raw reward (no clustering "
            "shaping). This is the main metric for CLIRS vs baseline comparisons."
        ),
        "population": "test_indices",
        "when": "once per trial, after model.learn() completes",
    },
    "life": {
        "name": "life",
        "csv_column": "life",
        "eval_json_field": "life",
        "aliases": ["life"],
        "summary": "Training proxy (train split, last callback log).",
        "definition": (
            "Mean applicable jobs on the train split at the final EvaluateCallback "
            "checkpoint: the second column of the last line in "
            "raw/*_training.txt (training steps, mean jobs, elapsed time). "
            "Reflects in-training progress on train learners — not generalization."
        ),
        "population": "train_indices",
        "when": "end of training log; do not use as the primary conclusion metric",
    },
    "original_applicable_jobs": {
        "name": "original_applicable_jobs",
        "csv_column": "original_applicable_jobs",
        "eval_json_field": "original_applicable_jobs",
        "aliases": ["original_applicable_jobs"],
        "summary": "Test-split baseline before any recommendation.",
        "definition": (
            "Mean applicable jobs on the test split using each learner's profile "
            "before the policy recommends courses."
        ),
        "population": "test_indices",
        "when": "once per trial, before model.learn()",
    },
}

# Fields that define the Complete Algorithm. Config must match manifest on every re-run.
# Tuple: (key in flat config from load_config, key in manifest.json)
_FROZEN_FIELDS: tuple[tuple[str, str], ...] = (
    ("results_lineage", "results_lineage"),
    ("seed", "data_seed"),
    ("total_steps", "total_steps"),
    ("nb_courses", "nb_courses"),
    ("nb_jobs", "nb_jobs"),
    ("model", "algorithm"),
    ("k", "k"),
    ("threshold", "threshold"),
    ("use_clustering", "use_clustering_in_config"),
    ("eval_freq", "eval_freq"),
    ("train_ratio", "train_ratio"),
    ("test_ratio", "test_ratio"),
)

# SB3 attributes to archive — only those present on the concrete algorithm class are kept.
_SB3_HPARAM_KEYS = (
    "learning_rate",
    "gamma",
    "batch_size",
    "buffer_size",
    "tau",
    "train_freq",
    "gradient_steps",
    "n_steps",
    "n_epochs",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
)


class ManifestValidationError(Exception):
    """Config or learner split no longer matches the frozen ``manifest.json``."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(
            "Complete Algorithm manifest mismatch:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


class Sb3HyperparameterSnapshot:
    """
    Read hyperparameters from a live SB3 model after ``get_model()``.

    We snapshot the *actual* defaults on the instance, not a hand-written table,
    so the manifest reflects what was really trained.
    """

    @staticmethod
    def from_model(model: Any) -> dict[str, Any]:
        import stable_baselines3

        hyperparameters = {
            key: _json_safe(getattr(model, key))
            for key in _SB3_HPARAM_KEYS
            if hasattr(model, key)
        }
        policy_name = (
            model.policy.__class__.__name__
            if getattr(model, "policy", None) is not None
            else None
        )
        return {
            "package_version": stable_baselines3.__version__,
            "python_version": _python_version_tag(),
            "algorithm_class": model.__class__.__name__,
            "policy": policy_name,
            "hyperparameters": hyperparameters,
        }


class CompleteAlgorithmManifest:
    """Assemble the JSON document written once per experiment cell."""

    @staticmethod
    def build(
        config: Mapping[str, Any],
        *,
        dataset: Any | None = None,
        sb3_snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "complete_algorithm_id": _complete_algorithm_id(config),
            "results_lineage": config.get("results_lineage", "CLIRS"),
            "method": method_slug(config),
            "data_seed": config.get("seed"),
            "algorithm": config.get("model"),
            "total_steps": config.get("total_steps"),
            "eval_freq": config.get("eval_freq"),
            "nb_courses": config.get("nb_courses"),
            "nb_jobs": config.get("nb_jobs"),
            "nb_cvs": config.get("nb_cvs"),
            "k": config.get("k"),
            "threshold": config.get("threshold"),
            "use_clustering_in_config": bool(config.get("use_clustering")),
            # Human-readable note: shaping is train_env-only (see CourseRecEnv).
            "clustering_reward_shaping": (
                "train_env only when use_clustering_in_config is true"
            ),
            "train_ratio": config.get("train_ratio"),
            "test_ratio": config.get("test_ratio"),
            # Metric contract: canonical names (life/end) + where they appear per artifact.
            # Source of truth: METRIC_DEFINITIONS in this module (not run.json).
            "metrics": {
                name: {
                    "summary": spec["summary"],
                    "definition": spec["definition"],
                    "population": spec["population"],
                    "when": spec["when"],
                    "csv_column": spec["csv_column"],
                    "eval_json_field": spec["eval_json_field"],
                    "aliases": spec.get("aliases", [name]),
                }
                for name, spec in METRIC_DEFINITIONS.items()
            },
            "metrics_source": "Utils/complete_algorithm.METRIC_DEFINITIONS",
        }
        if dataset is not None:
            manifest["split"] = {
                "train_size": int(len(dataset.train_indices)),
                "test_size": int(len(dataset.test_indices)),
            }
        if sb3_snapshot is not None:
            manifest["sb3"] = dict(sb3_snapshot)
        return manifest


class CompleteAlgorithmValidator:
    """Check that a new run belongs to the same Complete Algorithm as the manifest."""

    @staticmethod
    def validate(
        config: Mapping[str, Any],
        manifest: Mapping[str, Any],
        *,
        dataset: Any | None = None,
        split_indices_path: str | None = None,
    ) -> list[str]:
        errors: list[str] = []

        for config_key, manifest_key in _FROZEN_FIELDS:
            expected = manifest.get(manifest_key)
            actual = config.get(config_key)
            if config_key == "use_clustering":
                actual = bool(actual)
            if not _values_equal(expected, actual):
                errors.append(
                    f"{manifest_key}: manifest={expected!r} config={actual!r}"
                )

        # method_slug derives from use_clustering + algorithm (clirs_dqn vs baseline_dqn).
        expected_method = manifest.get("method")
        actual_method = method_slug(config)
        if expected_method is not None and expected_method != actual_method:
            errors.append(
                f"method: manifest={expected_method!r} config={actual_method!r}"
            )

        # Same data_seed must reproduce the same train/test row indices.
        if dataset is not None and split_indices_path and os.path.isfile(
            split_indices_path
        ):
            errors.extend(
                _validate_split_indices(dataset, split_indices_path)
            )

        return errors


class CompleteAlgorithmStage:
    """
    Entry point for pipeline: write or validate the Complete Algorithm contract.

    Call after ``Reinforce.__init__`` so the SB3 model exists for hyperparameter
    snapshot. Call after ``Dataset`` load so split sizes and indices are known.
    """

    def __init__(self, config: Mapping[str, Any], experiment_root: str):
        self.config = config
        self.experiment_root = experiment_root
        self.manifest_path = os.path.join(experiment_root, "manifest.json")
        self.split_indices_path = os.path.join(
            experiment_root, "split_indices.json"
        )

    def ensure(self, model: Any, dataset: Any) -> dict[str, Any]:
        """
        First run in cell: write manifest + split_indices.
        Later runs: load manifest and validate config/split; never overwrite.
        """
        sb3_snapshot = Sb3HyperparameterSnapshot.from_model(model)
        candidate = CompleteAlgorithmManifest.build(
            self.config,
            dataset=dataset,
            sb3_snapshot=sb3_snapshot,
        )

        if os.path.isfile(self.manifest_path):
            # --- Rule A.2: manifest is immutable; append trials only if config matches ---
            with open(self.manifest_path, encoding="utf-8") as f:
                existing = json.load(f)
            errors = CompleteAlgorithmValidator.validate(
                self.config,
                existing,
                dataset=dataset,
                split_indices_path=self.split_indices_path,
            )
            if errors:
                raise ManifestValidationError(errors)
            return existing

        # --- First trial in this cell: freeze the Complete Algorithm ---
        os.makedirs(self.experiment_root, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(candidate, f, indent=2)
        _write_split_indices(
            self.split_indices_path,
            self.config.get("seed"),
            dataset,
        )
        return candidate


def _complete_algorithm_id(config: Mapping[str, Any]) -> str:
    """Human-readable id; mirrors Results folder semantics."""
    return (
        f"{method_slug(config)}_steps{config.get('total_steps')}"
        f"_data{config.get('seed')}_courses{config.get('nb_courses')}"
    )


def _write_split_indices(path: str, data_seed: Any, dataset: Any) -> None:
    """Persist learner row indices so split_learners() can be audited later."""
    payload = {
        "data_seed": data_seed,
        "train_indices": [int(i) for i in dataset.train_indices],
        "test_indices": [int(i) for i in dataset.test_indices],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _validate_split_indices(dataset: Any, path: str) -> list[str]:
    """Detect data_seed or split_ratio changes that would invalidate old trials."""
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    errors: list[str] = []
    train_saved = [int(i) for i in saved.get("train_indices", [])]
    test_saved = [int(i) for i in saved.get("test_indices", [])]
    train_actual = [int(i) for i in dataset.train_indices]
    test_actual = [int(i) for i in dataset.test_indices]
    if train_saved != train_actual:
        errors.append(
            "train_indices: saved split differs from current Dataset "
            "(data_seed or split logic may have changed)"
        )
    if test_saved != test_actual:
        errors.append(
            "test_indices: saved split differs from current Dataset "
            "(data_seed or split logic may have changed)"
        )
    return errors


def _values_equal(a: Any, b: Any) -> bool:
    """Compare manifest vs config values (tolerant for float threshold)."""
    if a is None and b is None:
        return True
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars / exotic SB3 types to JSON-serializable values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _python_version_tag() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
