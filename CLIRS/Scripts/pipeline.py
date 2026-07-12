"""
CLIRS training pipeline — entry point for RL course recommendation experiments.

Run from repo root::

    poetry run python CLIRS/Scripts/pipeline.py --Config Config/run.json

High-level flow
---------------
1. Load config (``Config/run.json`` is source of truth).
2. Create the Results/ directory tree for this experiment *cell* (one frozen setup).
3. For each trial ``run`` in ``0 .. nb_runs-1``:
   a. Fix ``data_seed`` → same learners / jobs / courses / train-test split every trial.
   b. Draw ``rl_seed`` → SB3 + NumPy randomness differs per trial (variance estimate).
   c. Build ``Dataset`` and ``Reinforce`` (train env + eval env + DQN/PPO/A2C).
   d. On trial 0 only: freeze Complete Algorithm → ``manifest.json`` + ``split_indices.json``.
   e. Train, evaluate on test split, append one row to ``sweeps/*.csv``.

Outputs (under ``Results/{lineage}/steps_*/data_*/courses_*/``)
-----------------------------------------------------------------
- ``manifest.json``     — frozen hyperparameters + metric definitions (written once).
- ``split_indices.json``— train/test learner indices for this ``data_seed``.
- ``sweeps/{method}_data{seed}.csv`` — one row per trial (columns ``end``, ``life``, …).
- ``raw/*_training.txt`` — callback log on train split (feeds metric ``life``).
- ``raw/*_eval.json``    — per-trial test metrics (primary: ``end``).
- ``plots/clustering/``  — only when ``use_clustering: true``.

Key config fields (see ``Config/run.json``)
-------------------------------------------
- ``seeds.data``  → subsample + 70/30 split (fixed across trials).
- ``seeds.rl``    → one RL seed per trial (or ``rl_seed_base + trial_id``).
- ``use_clustering`` → ``true`` = CLIRS (``clirs_*``), ``false`` = baseline (``baseline_*``).
- ``nb_runs``     → number of independent trials T for statistical analysis later.

Do not change algorithm / steps / k / clustering mid-cell: ``CompleteAlgorithmStage``
validates trial 0 manifest on every re-run and exits if config drifts.
"""

import argparse
import sys
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed

from Dataset import Dataset
from Reinforce import Reinforce
from load_config import load_config

# Repo root on sys.path so ``from Utils...`` works when this script is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Utils.results_paths import ensure_experiment_dirs, rl_seed_for_trial
from Utils.complete_algorithm import CompleteAlgorithmStage, ManifestValidationError


def create_and_print_dataset(config):
    """
    Load learners, jobs, courses; apply subsample; split 70/30 train/test.

    Uses ``config["seed"]`` (data seed) for reproducible subsample and split.
    Prints a one-line summary (learner counts, catalog sizes).
    """
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main():
    # -------------------------------------------------------------------------
    # 1. Parse CLI and load experiment config
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Train CLIRS or baseline RL recommender. "
            "Reads Config/run.json by default."
        )
    )
    parser.add_argument(
        "--Config",
        help="Path to run.json (nested) or run.yaml (flat override)",
        default=r"Config/run.json",
    )
    args = parser.parse_args()
    config = load_config(args.Config)

    # -------------------------------------------------------------------------
    # 2. Prepare Results/ layout for this experiment cell
    #    Path pattern: Results/CLIRS/steps_{total_steps}/data_{data_seed}/courses_{nb}/
    #    manifest.json is NOT written here — see step 4d (needs SB3 model first).
    # -------------------------------------------------------------------------
    dirs = ensure_experiment_dirs(config)
    config["clustering_plots_dir"] = dirs["clustering_plots"]

    print(f"Experiment cell: {dirs['root']}")
    print(f"Trials to run: {config['nb_runs']} (method={config.get('model')}, "
          f"clustering={config.get('use_clustering')})")

    # -------------------------------------------------------------------------
    # 3. Trial loop — each iteration is one independent RL run (one CSV row)
    # -------------------------------------------------------------------------
    for run in range(config["nb_runs"]):

        # 3a. Seeds: data fixed, RL varies (Jordan et al. complete-algorithm design).
        rl_seed = rl_seed_for_trial(config, run)
        set_random_seed(rl_seed)
        config["current_rl_seed"] = rl_seed
        config["current_trial_id"] = run
        print(f"\n--- Trial {run + 1}/{config['nb_runs']} (rl_seed={rl_seed}) ---")

        # 3b. Data: same catalog and train/test split every trial (data_seed).
        dataset = create_and_print_dataset(config)

        # 3c. Agent: train_env (shaping on) + eval_env (raw jobs) + SB3 model.
        recommender = Reinforce(
            dataset,
            config["model"],
            config["k"],
            config["threshold"],
            run,
            config["total_steps"],
            config["eval_freq"],
        )

        # 3d. Complete Algorithm (trial 0 only): freeze manifest + split indices.
        #     Later trials in this process only append CSV rows; manifest unchanged.
        #     Re-running pipeline in same cell: validates config vs manifest (rule A.2).
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
                    "Hint: use a new experiment cell (change total_steps / data_seed / "
                    "nb_courses in config) or delete the existing Results folder."
                )
                sys.exit(1)

        # 3e. Train → log life (train split) → eval end (test split) → write artifacts.
        recommender.reinforce_recommendation()

        # 3f. Optional: print clustering diagnostics when CLIRS shaping is enabled.
        if config["use_clustering"]:
            clusterer = recommender.train_env.clusterer
            if hasattr(clusterer, "optimal_k"):
                print(f"Optimal number of clusters: {clusterer.optimal_k}")
            if hasattr(clusterer, "inertia_"):
                print(f"Clustering inertia: {clusterer.inertia_}")

    print(f"\nDone. Results under: {dirs['root']}")


if __name__ == "__main__":
    main()
