"""
JCRec fair-split pipeline — same 70/30 hold-out as CLIRS (primary baseline).

Run from repo root::

    poetry run python pipelines/run_jcrec_fair_pipeline.py --Config Config/run.json

Outputs under ``Results/JCRecFair/steps_*/data_*/courses_*/k_*/``.
Requires CLIRS cell to be run first (same ``data_seed``) for split alignment.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLIRS_SCRIPTS = _REPO_ROOT / "CLIRS" / "Scripts"
for path in (_CLIRS_SCRIPTS, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Dataset import Dataset
from load_config import load_config
from pipelines.sweep_eval import run_sweep_eval
from Utils.complete_algorithm import CompleteAlgorithmStage, ManifestValidationError
from Utils.experiment_log import ExperimentRunLog
from JCRecFair import ClirsSplitNotFoundError, JcrecFairReinforce, ensure_clirs_split
from Utils.results_paths import ensure_experiment_dirs, trial_artifact_paths
from Utils.trial_sweep import apply_rl_seed, rl_seed_for_trial, trial_plan_summary, trials_to_run, validate_trial_config

RL_ALGORITHMS = frozenset({"dqn", "ppo"})


def _prepare_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    cfg["pipeline"] = "jcrec_fair"
    cfg["results_lineage"] = cfg.get("jcrec_fair_results_lineage", "JCRecFair")
    cfg["use_clustering"] = False
    return cfg


def _manifest_exists(experiment_root: str) -> bool:
    return os.path.isfile(os.path.join(experiment_root, "manifest.json"))


def _run_fair_trial(
    config: dict[str, Any],
    dirs: dict[str, str],
    trial_id: int,
    dataset,
    run_log: ExperimentRunLog,
    *,
    freeze_manifest: bool,
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
        recommender = JcrecFairReinforce(
            dataset,
            config["model"],
            int(config["k"]),
            float(config["threshold"]),
            trial_id,
            int(config["total_steps"]),
            int(config["eval_freq"]),
        )
    except Exception as exc:
        run_log.record_exception(exc, trial_id=trial_id, phase="model_init")
        raise

    if freeze_manifest:
        try:
            CompleteAlgorithmStage(config, dirs["root"]).ensure(
                recommender.model, dataset
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

    try:
        results = recommender.run_trial()
    except Exception as exc:
        run_log.record_exception(exc, trial_id=trial_id, phase="jcrec_fair_trial")
        raise

    paths = trial_artifact_paths(config, trial_id)
    run_log.check_trial_artifacts(
        trial_id=trial_id,
        eval_path=paths["eval"],
        save_raw=bool(config.get("save_raw", True)),
    )
    run_log.record_trial(
        trial_id=trial_id,
        algorithm=config.get("model", ""),
        life=results.get("life"),
        end=results.get("end"),
        training_path=paths["training"],
    )
    print(f"Trial {trial_id} logged (jcrec_fair end={results.get('end')})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JCRec with CLIRS train/test split (fair baseline)."
    )
    parser.add_argument("--Config", default=r"Config/run.json")
    parser.add_argument("--from-trial", type=int, default=0)
    parser.add_argument("--to-trial", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    try:
        config = _prepare_config(load_config(args.Config))
    except Exception as exc:
        print(f"[ERROR] Failed to load config {args.Config}: {exc}")
        traceback.print_exc()
        sys.exit(1)

    algorithm = str(config.get("model", "")).lower()
    if algorithm not in RL_ALGORITHMS:
        print(f"[ERROR] jcrec_fair only supports dqn|ppo, got {algorithm!r}")
        sys.exit(1)

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
        pipeline="jcrec_fair",
        repo_root=str(_REPO_ROOT),
    ) as run_log:
        for warning in validate_trial_config(config):
            run_log.warn(warning)

        print(f"Experiment cell: {dirs['root']}")
        print(f"JCRec fair model={algorithm}, nb_runs={config['nb_runs']}")
        print(trial_plan_summary(config, trial_ids, resume=resume))

        if not trial_ids:
            print("No trials scheduled — sweep already complete for this range.")
        else:
            try:
                dataset = Dataset(config)
            except Exception as exc:
                run_log.record_exception(exc, phase="dataset_load")
                raise

            try:
                split_path = ensure_clirs_split(dataset, config)
                print(f"Split pinned from CLIRS: {split_path}")
                print(dataset)
            except ClirsSplitNotFoundError as exc:
                run_log.warn(str(exc))
                print(f"[ERROR] {exc}")
                sys.exit(1)
            except SystemExit:
                raise
            except Exception as exc:
                run_log.record_exception(exc, phase="clirs_split_sync")
                raise

            need_manifest = not _manifest_exists(dirs["root"])
            for trial_id in trial_ids:
                freeze = need_manifest
                _run_fair_trial(
                    config, dirs, trial_id, dataset, run_log, freeze_manifest=freeze
                )
                if freeze:
                    need_manifest = False

        if not args.skip_eval:
            run_sweep_eval(config, dirs["root"], run_log)

        print(f"\nDone. Results under: {dirs['root']}")


if __name__ == "__main__":
    main()
