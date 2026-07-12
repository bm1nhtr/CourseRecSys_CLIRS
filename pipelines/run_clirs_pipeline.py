"""
CLIRS experiment pipeline.

Run from repo root::

    poetry run python pipelines/run_clirs_pipeline.py --Config Config/run.json
"""

import argparse
import sys
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
from Utils.results_paths import ensure_experiment_dirs, rl_seed_for_trial


def create_and_print_dataset(config):
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLIRS / baseline RL pipeline.")
    parser.add_argument(
        "--Config",
        default=r"Config/run.json",
        help="Path to run.json or run.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.Config)

    dirs = ensure_experiment_dirs(config)
    config["clustering_plots_dir"] = dirs["clustering_plots"]

    print(f"Experiment cell: {dirs['root']}")
    print(
        f"Trials to run: {config['nb_runs']} "
        f"(method={config.get('model')}, clustering={config.get('use_clustering')})"
    )

    for run in range(config["nb_runs"]):
        rl_seed = rl_seed_for_trial(config, run)
        set_random_seed(rl_seed)
        config["current_rl_seed"] = rl_seed
        config["current_trial_id"] = run
        print(f"\n--- Trial {run + 1}/{config['nb_runs']} (rl_seed={rl_seed}) ---")

        dataset = create_and_print_dataset(config)
        recommender = Reinforce(
            dataset,
            config["model"],
            config["k"],
            config["threshold"],
            run,
            config["total_steps"],
            config["eval_freq"],
        )

        if run == 0:
            try:
                CompleteAlgorithmStage(config, dirs["root"]).ensure(
                    recommender.model,
                    dataset,
                )
                print("Complete Algorithm manifest OK (frozen or validated).")
            except ManifestValidationError as exc:
                print(exc)
                print(
                    "Hint: use a new experiment cell or delete the existing Results folder."
                )
                sys.exit(1)

        recommender.reinforce_recommendation()

        if config["use_clustering"]:
            clusterer = recommender.train_env.clusterer
            if hasattr(clusterer, "optimal_k"):
                print(f"Optimal number of clusters: {clusterer.optimal_k}")
            if hasattr(clusterer, "inertia_"):
                print(f"Clustering inertia: {clusterer.inertia_}")

    print(f"\nDone. Results under: {dirs['root']}")


if __name__ == "__main__":
    main()
