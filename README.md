# CLIRS — Course Recommendation with Cluster-Informed Reward Shaping RL

Reinforcement learning system that recommends courses to learners using **CLuster-Informed Reward Shaping** and mastery-level skill profiles.

## Overview

- K-means clustering on course features; reward adjusted by cluster transitions during training
- RL algorithms: DQN, PPO (Stable-Baselines3)
- Learner train / test split (70/30); metrics reported on held-out test CVs
- Primary metric: number of applicable jobs (configurable threshold)

## CLIRS pipeline (proposed method)

End-to-end flow for `run_clirs_pipeline.py` — from config and data split through Complete Algorithm freeze, RL training, held-out evaluation, and sweep statistics.

```mermaid
flowchart TB
    subgraph entry [Entry]
        CFG["Config/run.json\ndata_seed, train_ratio 70/30\nk, threshold, DQN|PPO\nnb_runs T, total_steps, eval_freq\nuse_clustering + reward multipliers"]
        RUN["pipelines/run_clirs_pipeline.py"]
        CELL["Experiment cell\nResults/CLIRS/steps_S/data_D/courses_C/k_K/"]
        CFG --> RUN --> CELL
    end

    subgraph data [1. Dataset D — one cell, shared across T trials]
        LOAD["CLIRS/Scripts/Dataset\nload taxonomy, CVs, jobs, courses"]
        SUB["Subsample nb_courses / nb_jobs / nb_cvs\nrng ← data_seed"]
        SPLIT["split_learners\nshuffle learner row indices"]
        TRAINIDX["train_indices ~70%\nRL episodes + callback monitor"]
        TESTIDX["test_indices ~30%\nheld out until final eval"]
        PIN["split_indices.json\npin train/test for this data_seed"]
        LOAD --> SUB --> SPLIT
        SPLIT --> TRAINIDX
        SPLIT --> TESTIDX
        SPLIT --> PIN
    end

    RUN --> LOAD

    subgraph cluster [2. Clustering — once per cell]
        KM["ensure_course_clusterer\nK-means, silhouette k selection"]
        CLJSON["course_clusters.json\nfrozen labels reused every trial"]
        KM --> CLJSON
    end

    SPLIT --> KM

    subgraph trials [3. Trial loop — T independent RL runs]
        PLAN["trials_to_run + resume\nskip trial_id already in sweep CSV"]
        SEED["trial t: rl_seed = rl_base + t\napply_rl_seed before each trial"]
        PLAN --> SEED
    end

    PIN --> PLAN

    subgraph complete [4. Complete Algorithm — trial 0 only]
        CA["CompleteAlgorithmStage.ensure"]
        MAN["manifest.json\nfrozen config + SB3 hparams\nmetric defs: life, end, original"]
        RULE["later trials / reruns: validate only\nManifestValidationError if config drifts"]
        CA --> MAN
        CA --> RULE
    end

    SEED --> CA

    subgraph train [5. Train — train_env only]
        TE["train_env CourseRecEnv\nepisode reset → sample train_indices CV"]
        SHAPE["cluster-informed reward shaping\ntrain_env only; eval_env raw reward"]
        LEARN["SB3 model.learn total_steps"]
        CB["EvaluateCallback every eval_freq\nrun policy on train_indices via eval_env"]
        TRLOG["raw/clirs_{algo}_dataD_rlR_kK_training.txt\nstep mean_jobs elapsed"]
        TE --> SHAPE --> LEARN --> CB --> TRLOG
    end

    CA --> TE
    CLJSON -.-> SHAPE

    subgraph eval [6. Held-out eval — test split once per trial]
        BASE["original_applicable_jobs\nmean jobs on test before recommend"]
        EVAL["eval_env: deterministic policy\nrecommend ≤k courses per test learner"]
        ENDM["end PRIMARY METRIC\nmean applicable jobs on test_indices\nafter course sequence"]
        BASE --> LEARN
        LEARN --> EVAL --> ENDM
    end

    TESTIDX -.-> EVAL

    subgraph row [7. Persist trial row]
        LIFE["life = last callback mean_jobs\nfrom *_training.txt\ntrain-split proxy — not primary"]
        EJSON["raw/clirs_{algo}_..._eval.json\nfull trial record"]
        CSV["upsert sweeps/clirs_{algo}_dataD.csv\none row per trial_id"]
        TRLOG --> LIFE
        ENDM --> EJSON
        LIFE --> CSV
        ENDM --> CSV
    end

    CSV --> PLAN

    subgraph sweep [8. sweep_eval — end of pipeline run]
        LONG["reports/trial_metrics_long.csv"]
        BOOT["reports/bootstrap_{end|life}_{mean|median}.csv\n10k bootstrap, 95% CI"]
        SUM["reports/clirs_{algo}_sweep_summary.json"]
        LONG --> BOOT --> SUM
    end

    CSV --> LONG
```

| Phase | What happens | Key output |
|-------|----------------|------------|
| **Split** | Learner CV rows shuffled with `data_seed`; 70% train / 30% test | `split_indices.json` |
| **Clustering** | Course features clustered once; labels frozen for all trials | `course_clusters.json` |
| **Complete Algorithm** | Trial 0 freezes experiment definition; later runs validate | `manifest.json` |
| **Train** | SB3 learns on `train_env` with optional shaping; callback monitors train split | `*_training.txt` → **`life`** |
| **Eval** | Policy runs once on held-out test learners (no shaping) | **`end`** (primary) |
| **Sweep stats** | Aggregate T trials with bootstrap CI | `bootstrap_*.csv`, sweep summary |

**Metric contract:** report conclusions on **`end`** (test split, after training). **`life`** is a training-progress proxy on the train split only. See `Utils/complete_algorithm.py` for definitions copied into `manifest.json`.

## Pipeline overview (orchestration)

One command runs all lineages in order, then cross-lineage comparison. Change `model.algorithm` in `Config/run.json` (`dqn` or `ppo`) between runs; compare outputs are keyed by algorithm so they do not overwrite.

```mermaid
flowchart TB
    CFG["Config/run.json\norchestration: clirs → jcrec_fair → jcrec\nmodel.algorithm: dqn | ppo"]

    CFG --> ORCH["pipelines/run_pipeline.py"]

    ORCH --> CLIRS["1. CLIRS\nrun_clirs_pipeline.py"]
    CLIRS --> SPLIT["Results/CLIRS/.../split_indices.json\n70/30 hold-out"]
    CLIRS --> R1["sweeps/clirs_{algo}_data{seed}.csv\n+ sweep_eval bootstrap"]

    SPLIT -.->|required| FAIR["2. JCRec fair\nrun_jcrec_fair_pipeline.py"]
    ORCH --> FAIR
    FAIR --> R2["Results/JCRecFair/.../sweeps/jcrec_fair_{algo}_*.csv\n+ sweep_eval bootstrap"]

    ORCH --> AUTHOR["3. JCRec author\nrun_jcrec_pipeline.py"]
    AUTHOR --> R3["Results/JCRec/.../sweeps/jcrec_{algo}_*.csv\n+ sweep_eval bootstrap"]

    R1 --> COMPARE
    R2 --> COMPARE
    R3 --> COMPARE

    COMPARE["cross_lineage_eval.py\nResults/compare/.../{algo}/"]
    COMPARE --> PRIMARY["clirs_vs_jcrec_fair\nprimary thesis compare\ntest split vs test split"]
    COMPARE --> REPL["clirs_vs_jcrec_author\nauthor replication\ntest vs all_learners"]
```

**Per-lineage cells:** `Results/{CLIRS|JCRecFair|JCRec}/steps_{S}/data_{D}/courses_{C}/k_{K}/`

**Compare artifacts:** `pairwise_comparison.csv`, `bootstrap_*.csv`, `plots/ecdf_*.png`, `report.md`

## Project structure

```
CLIRS-Recsys/
├── CLIRS/
│   └── Scripts/              # env, RL, dataset, clustering (no pipeline — see pipelines/)
├── JCRecFair/                # fair baseline (CLIRS split + jcrec env)
├── jcrec/                    # author baselines (Greedy, Optimal, Reinforce)
├── pipelines/                # run_pipeline.py, run_*_pipeline.py, cross_lineage_eval.py
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
poetry run python pipelines/run_jcrec_fair_pipeline.py --Config Config/run.json
poetry run python pipelines/run_jcrec_pipeline.py --Config Config/run.json
```

Regenerate compare only (after sweeps exist):

```bash
poetry run python pipelines/cross_lineage_eval.py --Config Config/run.json
```

### 3. Plot learning curves (optional)

```bash
poetry run python Utils/visualize_learning_curves.py
```

## Results layout

All training outputs go under `Results/` (gitignored). Path resolution is centralized in `Utils/results_paths.py` (`ensure_experiment_dirs`, `trial_artifact_paths`, `append_trial_csv_row`).

```
Results/
├── CLIRS/steps_{steps}/data_{seed}/courses_{nb}/k_{k}/       # proposed method
├── JCRecFair/steps_{steps}/data_{seed}/courses_{nb}/k_{k}/   # fair baseline (test split)
├── JCRec/steps_{steps}/data_{seed}/courses_{nb}/k_{k}/       # author reproduction
└── compare/steps_{steps}/data_{seed}/courses_{nb}/k_{k}/{algo}/
    ├── clirs_vs_jcrec_fair/
    └── clirs_vs_jcrec_author/
```

Each lineage cell:

```
Results/{CLIRS|JCRecFair|JCRec}/steps_*/data_*/courses_*/k_*/
    ├── run.log                 # only if warnings/errors (omit = run OK)
    ├── manifest.json
    ├── split_indices.json      # CLIRS + JCRecFair: train/test; JCRec: all_learners
    ├── sweeps/{method}_data{seed}.csv
    └── raw/{method}_data{seed}_rl{rl}_k{k}_eval.json
```

**Naming:** CLIRS → `clirs_{algo}`; JCRec fair → `jcrec_fair_{algo}`; JCRec author → `jcrec_{algo}`. All pipelines read `model.algorithm` in `Config/run.json` (e.g. `dqn`, `ppo`).

**Complete Algorithm:** `Utils/complete_algorithm.py` writes `manifest.json` on the first trial of a cell (SB3 hyperparameters + metric definitions). Later runs validate config against that manifest and refuse to mix experiments in the same folder. See [`Docs/README_DEVELOPMENT.md`](Docs/README_DEVELOPMENT.md#evaluation-metrics-life-vs-end).

**Run log:** `run.log` is created only when a run has warnings or errors (compact report, not full console). No file means the run looked fine — send `run.log` to the maintainer only if it exists. `Results/orchestration.log` is written only when orchestration fails or a cell produced a `run.log`.

**T trials:** `experiment.nb_runs` independent RL trials share one dataset/split (`data_seed`); trial `t` uses `rl_seed = seeds.rl_base + t`. Sweep CSV upserts by `trial_id`; resume skips completed trials. Column `evaluation_split`: CLIRS + JCRec fair `test` (70/30 hold-out); JCRec author `all_learners`. End-of-run bootstrap summary → `reports/{method}_sweep_summary.json`.

**Manage outputs:**

```bash
poetry run python CLIRS/Scripts/manage_results.py list
poetry run python CLIRS/Scripts/manage_results.py backup --config Config/run.json
```

## Dependencies

Managed in `pyproject.toml` (Python ^3.10): stable-baselines3, gymnasium, scikit-learn, numpy, pandas, matplotlib, seaborn, PyYAML, tqdm.

Legacy list: `requirements.txt`.

