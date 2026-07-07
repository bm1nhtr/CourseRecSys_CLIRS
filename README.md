# CLIRS — Course Recommendation with Clustering-Informed Reward Shaping RL

Reinforcement learning system that recommends courses to learners using **CLustering-Informed Reward Shaping** and mastery-level skill profiles.

## Overview

- K-means clustering on course features; reward adjusted by cluster transitions during training
- RL algorithms: DQN, A2C, PPO (Stable-Baselines3)
- Learner train / test split (70/30); metrics reported on held-out test CVs
- Primary metric: number of applicable jobs (configurable threshold)

## Project structure

```
wuir_class/
├── CLIRS/
│   ├── Scripts/          # pipeline, env, RL, dataset, clustering
│   ├── config/
│   │   ├── run.json      # primary config (pipeline reads this)
│   │   └── run.yaml      # flat reference / documentation
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


### 2. Run

```bash
poetry run python CLIRS/Scripts/pipeline.py --config CLIRS/config/run.json
```

## Dependencies

Managed in `pyproject.toml` (Python ^3.10): stable-baselines3, gymnasium, scikit-learn, numpy, pandas, matplotlib, seaborn, PyYAML, tqdm.

Legacy list: `requirements.txt`.

## Documentation

- `CLIRS/README_DEVELOPMENT.md` — architecture, clustering, results

## Acknowledgements

Based on [JCRec](https://github.com/Jibril-Frej/JCRec) by [Jibril Frej](https://github.com/Jibril-Frej).
