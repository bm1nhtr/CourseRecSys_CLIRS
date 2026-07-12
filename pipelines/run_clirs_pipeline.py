"""
CLIRS experiment pipeline.

Run from repo root::

    poetry run python pipelines/run_clirs_pipeline.py --Config Config/run.json
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLIRS_SCRIPTS = _REPO_ROOT / "CLIRS" / "Scripts"
for path in (_CLIRS_SCRIPTS, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Dataset import Dataset
from Reinforce import Reinforce
from load_config import load_config
from Utils.complete_algorithm import CompleteAlgorithmStage, ManifestValidationError
from Utils.experiment_log import ExperimentRunLog
from Utils.results_paths import (
    ensure_experiment_dirs,
    read_training_life_proxy,
    rl_seed_for_trial,
    trial_artifact_paths,
)


def create_and_print_dataset(config):
    dataset = Dataset(config)
    print(dataset)
    return dataset


def _freeze_manifest(
    config,
    dirs,
    run: int,
    recommender,
    dataset,
    run_log: ExperimentRunLog,
) -> None:
    if run != 0:
        return
    try:
        CompleteAlgorithmStage(config, dirs["root"]).ensure(
            recommender.model,
            dataset,
        )
        print("Complete Algorithm manifest OK (frozen or validated).")
    except ManifestValidationError as exc:
        run_log.warn(str(exc))
        print(
            "Hint: use a new experiment cell or delete the existing Results folder."
        )
        sys.exit(1)
    except Exception as exc:
        run_log.record_exception(exc, trial_id=run, phase="manifest_freeze")
        raise


def _run_clirs_trial(config, dirs, run: int, run_log: ExperimentRunLog) -> None:
    rl_seed = rl_seed_for_trial(config, run)
    set_random_seed(rl_seed)
    config["current_rl_seed"] = rl_seed
    config["current_trial_id"] = run
    print(f"\n--- Trial {run + 1}/{config['nb_runs']} (rl_seed={rl_seed}) ---")

    try:
        dataset = create_and_print_dataset(config)
    except Exception as exc:
        run_log.record_exception(exc, trial_id=run, phase="dataset_load")
        raise

    try:
        recommender = Reinforce(
            dataset,
            config["model"],
            config["k"],
            config["threshold"],
            run,
            config["total_steps"],
            config["eval_freq"],
        )
    except Exception as exc:
        run_log.record_exception(exc, trial_id=run, phase="model_init")
        raise

    _freeze_manifest(config, dirs, run, recommender, dataset, run_log)

    try:
        recommender.reinforce_recommendation()
    except Exception as exc:
        run_log.record_exception(exc, trial_id=run, phase="train_and_eval")
        raise

    paths = trial_artifact_paths(config, run)
    life = None
    end = None
    try:
        life = read_training_life_proxy(paths["training"])
        if config.get("save_raw", True) and os.path.isfile(paths["eval"]):
            with open(paths["eval"], encoding="utf-8") as f:
                end = json.load(f).get("end")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        run_log.record_exception(exc, trial_id=run, phase="read_trial_artifacts")

    run_log.check_trial_artifacts(
        trial_id=run,
        eval_path=paths["eval"],
        save_raw=bool(config.get("save_raw", True)),
    )
    run_log.record_trial(
        trial_id=run,
        algorithm=config["model"],
        life=life,
        end=end,
        training_path=paths["training"],
    )

    if config["use_clustering"]:
        try:
            clusterer = recommender.train_env.clusterer
            if hasattr(clusterer, "optimal_k"):
                print(f"Optimal number of clusters: {clusterer.optimal_k}")
            if hasattr(clusterer, "inertia_"):
                print(f"Clustering inertia: {clusterer.inertia_}")
        except Exception as exc:
            run_log.record_exception(exc, trial_id=run, phase="clustering_summary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLIRS / baseline RL pipeline.")
    parser.add_argument(
        "--Config",
        default=r"Config/run.json",
        help="Path to run.json or run.yaml",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.Config)
    except Exception as exc:
        print(f"[ERROR] Failed to load config {args.Config}: {exc}")
        traceback.print_exc()
        sys.exit(1)

    config["pipeline"] = "clirs"

    try:
        dirs = ensure_experiment_dirs(config)
    except Exception as exc:
        print(f"[ERROR] Failed to create experiment directories: {exc}")
        traceback.print_exc()
        sys.exit(1)

    config["clustering_plots_dir"] = dirs["clustering_plots"]

    with ExperimentRunLog(
        config,
        dirs["root"],
        config_path=args.Config,
        pipeline="clirs",
        repo_root=str(_REPO_ROOT),
    ) as run_log:
        print(f"Experiment cell: {dirs['root']}")
        print(
            f"Trials to run: {config['nb_runs']} "
            f"(method={config.get('model')}, clustering={config.get('use_clustering')})"
        )

        for run in range(config["nb_runs"]):
            _run_clirs_trial(config, dirs, run, run_log)

        print(f"\nDone. Results under: {dirs['root']}")


if __name__ == "__main__":
    main()
