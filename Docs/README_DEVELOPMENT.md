# Course Recommendation System - Development Guide (Clustering-based Approach)

This document provides detailed information for developers working on the course recommendation system with mastery levels and clustering-based reward adjustment.

## System Architecture

### Core Components

1. **RL Environment** (`CLIRS/Scripts/CourseRecEnv.py`):
   - Implements Gymnasium environment for course recommendations
   - Handles state representation with mastery levels (0-3)
   - Supports CLIRS clustering-based reward adjustment (persistent adjusted reference)

2. **Data Management** (`CLIRS/Scripts/Dataset.py`):
   - Handles data loading and preprocessing
   - Manages learner, job, and course data with mastery levels
   - Counts applicable jobs via inverted-index candidates + batch matching (see below)

3. **Skill matching** (`CLIRS/Scripts/matchings.py`; mirrored in `jcrec/matchings.py`):
   - Scalar APIs: `matching`, `learner_job_matching`, course required/provided matching
   - Batch API: `learner_jobs_matching(learner, jobs)` — same scores as row-wise calls
   - Used by `Dataset.get_nb_applicable_jobs` (hot path every env step / eval)

4. **RL Implementation** (`CLIRS/Scripts/Reinforce.py`):
   - Implements DQN, A2C, and PPO algorithms
   - Manages model training and evaluation
   - Handles hyperparameter tuning
   - Supports clustering-based reward adjustment

5. **Pipeline** (`pipelines/run_clirs_pipeline.py`, `pipelines/run_jcrec_pipeline.py`, `pipelines/run_pipeline.py`):
   - Orchestrates the training process
   - Manages configuration and logging
   - Handles results storage and visualization
   - Supports multiple k values (1,2,3,...)

6. **Utilities** (`Utils/`):
   - `visualize_learning_curves.py` — plot learning curves from `Results/`
   - `general_utils.py` — shared helpers (placeholder)
   - `results_paths.py` — Results layout, sweep CSV, `trial_wall_minutes`
## Clustering Implementation

The system uses K-means clustering to group similar courses based on their skill profiles. This helps improve the RL performance by adjusting rewards based on course cluster membership.

### Clustering Features
The system extracts 5 key features for each course:

1. **Coverage**: Overall skill coverage ratio (average of required and provided skill coverage)
2. **Required Entropy**: Diversity measure of required skills distribution
3. **Provided Entropy**: Diversity measure of provided skills distribution  
4. **Average Level Gap**: Average difference between required and provided skill levels
5. **Maximum Level Gap**: Maximum difference between required and provided skill levels

### Reward Adjustment Rules
CLIRS shapes rewards in **train_env only** using a persistent adjusted reference
(`R'_adjusted,ref`, tracked as `best_reward_so_far` in `clustering.py`):

1. **First recommendation (C₁)**: Always apply `first_recommendation` multiplier (default ×1.3); sets initial ref
2. **Progress (C₂…Cₖ)**: If `R_base > R'_adjusted,ref`, apply `progress_increase` (default ×1.3) and update ref
3. **No improvement**: If `R_base ≤ R'_adjusted,ref`, apply `no_improvement` (×1.0); ref unchanged

**Persistent reference mechanism**:
- Compare each step's **base** reward (applicable jobs) against the last **bonused adjusted** value
- Skipped steps do not reset the benchmark (avoids reward reset drift)
- Eval / test always use raw reward (no shaping)

Config keys (`Config/run.json` → `clustering.reward_multipliers`):
`first_recommendation`, `progress_increase`, `no_improvement`

### Clustering Process
1. **Feature Extraction** (once per experiment cell):
   - Calculate skill coverage for each course
   - Compute entropy for required and provided skills (active skills only for level gaps)
   - Analyze level gaps between required and provided skills
2. **Clustering** (fit once, frozen in ``course_clusters.json``):
   - Normalize features using StandardScaler
   - Select k via ``selection`` (`silhouette` or `elbow`, k ≥ 2) when ``auto_clusters`` is true
   - Apply K-means; write ``reports/cluster_selection.json`` and ``reports/cluster_quality.json``
3. **Reward Adjustment**:
   - Apply CLIRS persistent-reference multipliers during training (``train_env`` only)
   - Cluster labels are frozen for all T trials in the cell

## Configuration Guide

The system is configured through `Config/run.yaml` (reference) and `Config/run.json` (source of truth) with the following parameters:

### Model Configuration
```yaml
model: "ppo"  # or "dqn", "a2c"
total_steps: 500000
eval_freq: 1000
```

### Environment Configuration
```yaml
threshold: 0.8  # Matching threshold
k: 4  # Number of recommendations
use_clustering: true  # Enable clustering
```

### Clustering Configuration
```yaml
use_clustering: true
auto_clusters: true
max_clusters: 10
selection: silhouette
min_cluster_size: 5
# run.json → clustering.reward_multipliers
clustering:
  first_recommendation: 1.3
  progress_increase: 1.3
  no_improvement: 1.0
```

## Results Management

### Directory structure

Paths are resolved by `Utils/results_paths.py` (used by `pipeline.py`, `Reinforce.py`, and `visualize_learning_curves.py`).

```
Results/
└── {results_lineage}/              # default: CLIRS
    └── steps_{total_steps}/
        └── data_{data_seed}/
            └── courses_{nb_courses}/   # courses_all when nb_courses=-1
                └── k_{k}/              # one Complete Algorithm cell per k (2..6)
                ├── manifest.json
                ├── split_indices.json
                ├── course_clusters.json  # frozen course labels (when use_clustering)
                ├── sweeps/             # {method}_data{seed}.csv — one row per trial
                ├── reports/            # cluster_selection.json, cluster_quality.json, sweep summary
                ├── plots/
                │   └── clustering/
                └── raw/                # when save_raw: true
                    ├── *_training.txt
                    └── *_eval.json
```

### Management commands

1. List experiment cells:
```bash
python CLIRS/Scripts/manage_results.py list
```

2. Backup one experiment cell (from active config):
```bash
python CLIRS/Scripts/manage_results.py backup --config Config/run.json
```

### Learning curve plots

From repo root (reads raw logs for the experiment cell in `Config/run.json`):

```bash
python Utils/visualize_learning_curves.py
python Utils/visualize_learning_curves.py --config Config/run.json
```

Plots are written to `Results/.../plots/`. No manual `BRANCH_NAME` setting is required.

## Complete Algorithm (frozen experiment cell)

Each folder under `Results/.../courses_*/k_*/` is one **Complete Algorithm**: fixed method, budget, env, **k**, and data subsample. Implementation: `Utils/complete_algorithm.py`.

| Artifact | Purpose |
|----------|---------|
| `manifest.json` | Written on **first** trial only. Records config, SB3 defaults, metric definitions. Never overwritten (A.2). |
| `split_indices.json` | Train/test learner row indices for this `data_seed`. Validated on later runs. |

If you change `total_steps`, `k`, `use_clustering`, algorithm, etc. and re-run the same cell path, the pipeline **exits with an error** instead of mixing trials. Use a new cell path or delete the old cell.

## Evaluation metrics: `life` vs `end`

Canonical definitions live in `Utils/complete_algorithm.METRIC_DEFINITIONS` (source of truth in code). On the first trial of an experiment cell, they are **copied** into `manifest.json` under `"metrics"` so `Results/` is readable without the repo.

**Do not duplicate full metric text in `Config/run.json`** — run.json holds hyperparameters; manifest holds the frozen contract for that cell.

### `end` — primary metric (report this)

| | |
|--|--|
| **CSV column** | `end` |
| **eval.json** | `end` (legacy name: `new_applicable_jobs`) |
| **Population** | Held-out **test** learners (`test_indices`) |
| **When** | Once per trial, **after** `model.learn()` |
| **Meaning** | Mean applicable jobs after the policy recommends `k` courses. Raw jobs only (no clustering reward shaping). |

Use `end` for CLIRS vs `baseline_*` comparisons and thesis tables.

### `life` — training proxy (secondary)

| | |
|--|--|
| **CSV column** | `life` |
| **eval.json** | `life` |
| **Source** | Last line of `raw/*_training.txt` (column 2) |
| **Population** | **Train** learners (`train_indices`) |
| **When** | Last `EvaluateCallback` checkpoint during training |
| **Meaning** | In-training progress on the train split — optimistic vs `end`. |

Do **not** treat `life` as the main conclusion metric. Use it for learning-curve-style analysis alongside `Utils/visualize_learning_curves.py`.

### `original_applicable_jobs`

Test-split mean applicable jobs **before** any recommendation (baseline per trial).

### `trial_wall_minutes`

Wall-clock time for one trial (train + final eval), stored in sweep CSV / eval JSON / compare tables.

- **Unit:** minutes only (legacy `trial_wall_seconds` is migrated on read).
- **Precision:** 5 decimal places.

## Job applicability scoring (performance)

Reward and metrics depend on counting jobs whose learner–job match score is `≥ threshold`.

| Step | Behavior |
|------|----------|
| Candidates | Inverted index: jobs that share ≥1 skill with the learner (unchanged) |
| Scores | `matchings.learner_jobs_matching` on the candidate job rows (batch) |
| Count | `count(scores >= threshold)` |

Semantics match the previous per-job Python loop (including float edge cases at the threshold). Prefer this path over reintroducing a row-wise loop in `get_nb_applicable_jobs`.

Parity tests: `python -m unittest tests.test_matchings_vectorized -v`


## Important Notes

1. **Results Management**:
   - Always backup before deleting experiment cells
   - Results are keyed by `results_lineage`, `total_steps`, and `data_seed`
   - Do not commit `Results/` to git
   - Sweep CSV holds one row per trial; aggregate stats are computed separately


2. **Backup and Version Control**:
   - Backups are stored in `backups/` with timestamp
   - Each branch has its own results directory
   - Do not commit results to git
   - Document mastery level changes 