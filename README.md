# CIRS — Course Recommendation with Clustering-Informed Reward Shaping RL

Reinforcement learning system that recommends courses to learners using **clustering-based reward shaping** and mastery-level skill profiles.

## Overview

- K-means clustering on course features; reward adjusted by cluster transitions during training
- RL algorithms: DQN, A2C, PPO (Stable-Baselines3)
- Learner train / validation / test split; metrics reported on held-out test CVs
- Primary metric: number of applicable jobs (configurable threshold)

## Project structure

```
wuir_class/
├── CLASS/
│   ├── Scripts/          # pipeline, env, RL, dataset, clustering
│   ├── config/
│   │   └── run.yaml.example   # copy → run.yaml (local, gitignored)
│   ├── results/          # training outputs (gitignored)
│   └── README_DEVELOPMENT.md
├── Data - Collection/Final/   # local dataset (gitignored)
├── pyproject.toml        # Poetry dependencies
```

## Quick start

### 1. Install (Poetry)

```bash
poetry lock
poetry install
```

Use `poetry shell` or prefix commands with `poetry run`.

### 2. Config

Copy the example config and edit as needed:

```bash
copy CLASS\config\run.yaml.example CLASS\config\run.yaml
```

(`run.yaml` and `Data - Collection/` are gitignored — keep them locally.)

### 3. Run

```bash
poetry run python CLASS/Scripts/pipeline.py --config CLASS/config/run.yaml
```

## Dependencies

Managed in `pyproject.toml` (Python ^3.10): stable-baselines3, gymnasium, scikit-learn, numpy, pandas, matplotlib, seaborn, PyYAML, tqdm.

Legacy list: `requirements.txt`.

## Documentation

- `CLASS/README_DEVELOPMENT.md` — architecture, clustering, results

## Acknowledgements

Based on [JCRec](https://github.com/Jibril-Frej/JCRec) by [Jibril Frej](https://github.com/Jibril-Frej).
