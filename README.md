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
│   └── Scripts/              # pipeline, env, RL, dataset, clustering
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
poetry run python CLIRS/Scripts/pipeline.py --Config Config/run.json
```

### 3. Plot learning curves (optional)

```bash
poetry run python Utils/visualize_learning_curves.py
```

## Results layout

All training outputs go under `Results/` (gitignored). Path resolution is centralized in `Utils/results_paths.py` (`ensure_experiment_dirs`, `trial_artifact_paths`, `append_trial_csv_row`).

```
Results/
└── CLIRS/                          # results_lineage (Config/run.json)
    └── steps_{total_steps}/        # e.g. steps_5000000
        └── data_{data_seed}/       # e.g. data_42
            └── courses_{nb_courses}/   # e.g. courses_100 (courses_all if nb_courses=-1)
                ├── manifest.json          # frozen Complete Algorithm contract
                ├── split_indices.json     # train/test row indices for data_seed
                ├── sweeps/
                │   └── {method}_data{data_seed}.csv   # 1 row per trial (T = nb_runs)
                ├── reports/                           # compare.py output (future)
                ├── plots/
                │   └── clustering/                    # elbow / PCA plots from clustering
                └── raw/                               # optional (save_raw: true)
                    ├── {method}_data{d}_rl{r}_k{k}_training.txt
                    └── {method}_data{d}_rl{r}_k{k}_eval.json
```

**Naming:** `method` is `clirs_{algo}` or `baseline_{algo}` depending on `use_clustering`. Each trial uses `rl_seed` from `seeds.rl[trial_id]` or `seed + trial_id`. `nb_courses` scopes the experiment cell (folder); it is also stored in `manifest.json` and sweep CSV columns. Mean ± std across trials is computed downstream (e.g. `eval/compare.py`), not in the sweep CSV.

**Complete Algorithm:** `Utils/complete_algorithm.py` writes `manifest.json` on the first trial of a cell (SB3 hyperparameters + metric definitions). Later runs validate config against that manifest and refuse to mix experiments in the same folder. See [`Docs/README_DEVELOPMENT.md`](Docs/README_DEVELOPMENT.md#evaluation-metrics-life-vs-end).

**Manage outputs:**

```bash
poetry run python CLIRS/Scripts/manage_results.py list
poetry run python CLIRS/Scripts/manage_results.py backup --config Config/run.json
```

## Dependencies

Managed in `pyproject.toml` (Python ^3.10): stable-baselines3, gymnasium, scikit-learn, numpy, pandas, matplotlib, seaborn, PyYAML, tqdm.

Legacy list: `requirements.txt`.

