# Course Recommendation System - Development Guide (Clustering-based Approach)

This document provides detailed information for developers working on the course recommendation system with mastery levels and clustering-based reward adjustment.

## System Architecture

### Core Components

1. **RL Environment** (`CLIRS/Scripts/CourseRecEnv.py`):
   - Implements Gymnasium environment for course recommendations
   - Handles state representation with mastery levels (0-3)
   - Supports clustering-based reward adjustment:
     - Same cluster & reward increase: x1.1
     - Different cluster & reward increase: x1.3
   

2. **Data Management** (`CLIRS/Scripts/Dataset.py`):
   - Handles data loading and preprocessing
   - Manages learner, job, and course data with mastery levels
   - Provides skill matching functionality considering mastery levels

3. **RL Implementation** (`CLIRS/Scripts/Reinforce.py`):
   - Implements DQN, A2C, and PPO algorithms
   - Manages model training and evaluation
   - Handles hyperparameter tuning
   - Supports clustering-based reward adjustment

4. **Pipeline** (`CLIRS/Scripts/pipeline.py`):
   - Orchestrates the training process
   - Manages configuration and logging
   - Handles results storage and visualization
   - Supports multiple k values (1,2,3,...)

5. **Utilities** (`Utils/`):
   - `visualize_learning_curves.py` — plot learning curves from `Results/`
   - `general_utils.py` — shared helpers (placeholder)
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
The clustering mechanism modifies rewards based on cluster transitions and tracks the best adjusted reward so far:

1. **First Recommendation**: Applies diff_cluster_increase multiplier (x1.3) to encourage exploration
2. **Same Cluster & Reward Increase**: Moderate encouragement (x1.1) when current reward > best reward so far
3. **Different Cluster & Reward Increase**: Strong encouragement (x1.3) when current reward > best reward so far
4. **Reward Decrease**: Neutral multiplier (x1.0) when current reward ≤ best reward so far

**Best Reward So Far Mechanism**:
- The system tracks the best adjusted reward achieved in the current recommendation sequence
- Only applies positive multipliers when the current reward exceeds the best reward so far
- This ensures that only genuinely improving actions receive encouragement
- The best reward is updated whenever a better reward is achieved

### Clustering Process
1. **Feature Extraction**:
   - Calculate skill coverage for each course
   - Compute entropy for required and provided skills
   - Analyze level gaps between required and provided skills
2. **Clustering**:
   - Normalize features using StandardScaler
   - Apply K-means clustering
   - Optionally use elbow method to determine optimal k
3. **Reward Adjustment**:
   - Track cluster transitions during training
   - Apply reward multipliers based on transition rules
   - Store cluster information for analysis

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
n_clusters: 4
auto_clusters: true
max_clusters: 10
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
                ├── manifest.json
                ├── sweeps/             # {method}_data{seed}.csv — one row per trial
                ├── reports/            # statistical compare output (future)
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