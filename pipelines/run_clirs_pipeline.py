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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLIRS_SCRIPTS = _REPO_ROOT / "CLIRS" / "Scripts"
for path in (_CLIRS_SCRIPTS, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Dataset import Dataset
from Reinforce import Reinforce
from load_config import load_config
from pipelines.sweep_eval import run_sweep_eval
from Utils.course_clusters import ensure_course_clusterer
from Utils.complete_algorithm import CompleteAlgorithmStage, ManifestValidationError
from Utils.experiment_log import ExperimentRunLog
from Utils.results_paths import (
    ensure_experiment_dirs,
    read_training_life_proxy,
    rl_seed_for_trial,
    trial_artifact_paths,
)
from Utils.trial_sweep import (
    apply_rl_seed,
    trial_plan_summary,
    trials_to_run,
    validate_trial_config,
)
from JCRecFair.split_sync import publish_clirs_split_artifact, require_clirs_split_file


def _manifest_exists(experiment_root: str) -> bool:
    return os.path.isfile(os.path.join(experiment_root, "manifest.json"))


def _freeze_manifest(
    config,
    dirs,
    trial_id: int,
    recommender,
    dataset,
    run_log: ExperimentRunLog,
) -> None:
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
        run_log.record_exception(exc, trial_id=trial_id, phase="manifest_freeze")
        raise


def _run_clirs_trial(
    config,
    dirs,
    trial_id: int,
    dataset,
    run_log: ExperimentRunLog,
    *,
    freeze_manifest: bool,
    clusterer=None,
) -> None:
    rl_seed = rl_seed_for_trial(config, trial_id)
    apply_rl_seed(rl_seed)
    config["current_rl_seed"] = rl_seed
    config["current_trial_id"] = trial_id
    print(
        f"\n--- Trial {trial_id + 1}/{config['nb_runs']} "
        f"(trial_id={trial_id}, rl_seed={rl_seed}) ---"
    )

    try:
        recommender = Reinforce(
            dataset,
            config["model"],
            config["k"],
            config["threshold"],
            trial_id,
            config["total_steps"],
            config["eval_freq"],
            clusterer=clusterer,
        )
    except Exception as exc:
        run_log.record_exception(exc, trial_id=trial_id, phase="model_init")
        raise

    if freeze_manifest:
        _freeze_manifest(config, dirs, trial_id, recommender, dataset, run_log)

    try:
        recommender.reinforce_recommendation()
    except Exception as exc:
        run_log.record_exception(exc, trial_id=trial_id, phase="train_and_eval")
        raise

    paths = trial_artifact_paths(config, trial_id)
    life = None
    end = None
    try:
        life = read_training_life_proxy(paths["training"])
        if config.get("save_raw", True) and os.path.isfile(paths["eval"]):
            with open(paths["eval"], encoding="utf-8") as f:
                end = json.load(f).get("end")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        run_log.record_exception(exc, trial_id=trial_id, phase="read_trial_artifacts")

    run_log.check_trial_artifacts(
        trial_id=trial_id,
        eval_path=paths["eval"],
        save_raw=bool(config.get("save_raw", True)),
    )
    run_log.record_trial(
        trial_id=trial_id,
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
            run_log.record_exception(exc, trial_id=trial_id, phase="clustering_summary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLIRS / baseline RL pipeline.")
    parser.add_argument(
        "--Config",
        default=r"Config/run.json",
        help="Path to run.json or run.yaml",
    )
    parser.add_argument("--from-trial", type=int, default=0)
    parser.add_argument("--to-trial", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run trials even if already present in sweep CSV",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Alias for --no-resume",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip sweep summary at end of run",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.Config)
    except Exception as exc:
        print(f"[ERROR] Failed to load config {args.Config}: {exc}")
        traceback.print_exc()
        sys.exit(1)

    config["pipeline"] = "clirs"
    resume = not (args.no_resume or args.force)

    try:
        validate_trial_config(config)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    try:
        dirs = ensure_experiment_dirs(config)
    except Exception as exc:
        print(f"[ERROR] Failed to create experiment directories: {exc}")
        traceback.print_exc()
        sys.exit(1)

    config["clustering_plots_dir"] = dirs["clustering_plots"]
    config["clustering_reports_dir"] = dirs["reports"]
    trial_ids = trials_to_run(
        config,
        from_trial=args.from_trial,
        to_trial=args.to_trial,
        resume=resume,
    )

    with ExperimentRunLog(
        config,
        dirs["root"],
        config_path=args.Config,
        pipeline="clirs",
        repo_root=str(_REPO_ROOT),
    ) as run_log:
        for warning in validate_trial_config(config):
            run_log.warn(warning)

        print(f"Experiment cell: {dirs['root']}")
        print(
            f"Method={config.get('model')}, clustering={config.get('use_clustering')}, "
            f"nb_runs={config['nb_runs']}"
        )
        print(trial_plan_summary(config, trial_ids, resume=resume))

        if not trial_ids:
            print("No trials scheduled — sweep already complete for this range.")
            try:
                require_clirs_split_file(config)
            except SystemExit as exc:
                run_log.warn(str(exc))
                print(f"[ERROR] {exc}")
                sys.exit(1)
        else:
            try:
                dataset = Dataset(config)
                split_path = publish_clirs_split_artifact(config, dataset)
                print(f"CLIRS split published: {split_path}")
                print(dataset)
            except Exception as exc:
                run_log.record_exception(exc, phase="dataset_load")
                raise

            shared_clusterer = None
            if config.get("use_clustering"):
                try:
                    shared_clusterer = ensure_course_clusterer(
                        config, dataset, dirs
                    )
                except ManifestValidationError as exc:
                    run_log.warn(str(exc))
                    print(
                        "Hint: use a new experiment cell or delete the existing "
                        "Results folder."
                    )
                    sys.exit(1)
                except Exception as exc:
                    run_log.record_exception(exc, phase="course_clustering")
                    raise

            need_manifest = not _manifest_exists(dirs["root"])
            for trial_id in trial_ids:
                freeze = need_manifest
                _run_clirs_trial(
                    config,
                    dirs,
                    trial_id,
                    dataset,
                    run_log,
                    freeze_manifest=freeze,
                    clusterer=shared_clusterer,
                )
                if freeze:
                    need_manifest = False

        if not args.skip_eval:
            run_sweep_eval(config, dirs["root"], run_log)

        print(f"\nDone. Results under: {dirs['root']}")


if __name__ == "__main__":
    main()
