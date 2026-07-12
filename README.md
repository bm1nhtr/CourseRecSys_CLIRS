# CLIRS — Course Recommendation with Cluster-Informed Reward Shaping RL

Reinforcement learning system that recommends courses to learners using **CLuster-Informed Reward Shaping** and mastery-level skill profiles.

## Overview

- K-means clustering on course features; reward adjusted by cluster transitions during training
- RL algorithms: DQN, PPO (Stable-Baselines3)
- Learner train / test split (70/30); metrics reported on held-out test CVs
- Primary metric: number of applicable jobs (configurable threshold)

## Project structure

```
CLIRS-Recsys/
├── CLIRS/
│   └── Scripts/              # env, RL, dataset, clustering (no pipeline — see pipelines/)
├── jcrec/                    # author baselines (Greedy, Optimal, Reinforce)
├── pipelines/                # run_pipeline.py, run_clirs_pipeline.py, run_jcrec_pipeline.py
├── Config/
│   ├── run.json              # primary config (pipeline reads this)
│   └── run.yaml              # flat reference / documentation
├── Data/                     # dataset (taxonomy, CVs, jobs, courses)
├── Docs/
│   └── README_DEVELOPMENT.md # architecture, clustering, results
├── Utils/
│   ├── results_paths.py              # canonical Results/ layout (used by pipeline + plots)
│   ├── visualize_learning_curves.py  # learning-curve plots from raw logs
│   └── general_utils.py              # shared helpers (placeholder)
├── Results/                  # training outputs (gitignored)
└── pyproject.toml            # Poetry dependencies
```

## Quick start

### 1. Install (Poetry)

```bash
poetry lock
poetry install
```

Use `poetry shell` or prefix commands with `poetry run`.

### 2. Run training

```bash
poetry run python pipelines/run_pipeline.py --Config Config/run.json
```

Or a single backend:

```bash
poetry run python pipelines/run_clirs_pipeline.py --Config Config/run.json
poetry run python pipelines/run_jcrec_pipeline.py --Config Config/run.json
```

### 3. Plot learning curves (optional)

```bash
poetry run python Utils/visualize_learning_curves.py
```

## Results layout

All training outputs go under `Results/` (gitignored). Path resolution is centralized in `Utils/results_paths.py` (`ensure_experiment_dirs`, `trial_artifact_paths`, `append_trial_csv_row`).

```
Results/
├── CLIRS/steps_{steps}/data_{seed}/courses_{nb}/k_{k}/   # run_clirs_pipeline
└── JCRec/steps_{steps}/data_{seed}/courses_{nb}/k_{k}/   # run_jcrec_pipeline
    ├── manifest.json
    ├── split_indices.json
    ├── sweeps/{method}_data{seed}.csv
    └── raw/{method}_data{seed}_rl{rl}_k{k}_eval.json
```

**Naming:** CLIRS → `clirs_{algo}` / `baseline_{algo}`; JCRec → `jcrec_{algo}`. Both pipelines read `model.algorithm` in `Config/run.json` (e.g. `dqn`, `ppo`, `greedy`).

**Complete Algorithm:** `Utils/complete_algorithm.py` writes `manifest.json` on the first trial of a cell (SB3 hyperparameters + metric definitions). Later runs validate config against that manifest and refuse to mix experiments in the same folder. See [`Docs/README_DEVELOPMENT.md`](Docs/README_DEVELOPMENT.md#evaluation-metrics-life-vs-end).

**Manage outputs:**

```bash
poetry run python CLIRS/Scripts/manage_results.py list
poetry run python CLIRS/Scripts/manage_results.py backup --config Config/run.json
```

## Dependencies

Managed in `pyproject.toml` (Python ^3.10): stable-baselines3, gymnasium, scikit-learn, numpy, pandas, matplotlib, seaborn, PyYAML, tqdm.

Legacy list: `requirements.txt`.

