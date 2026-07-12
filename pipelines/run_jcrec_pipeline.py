"""
JCRec author pipeline — same Results layout + Complete Algorithm freeze as CLIRS.

Run from repo root::

    poetry run python pipelines/run_jcrec_pipeline.py --Config Config/run.json

Outputs under ``Results/JCRec/steps_*/data_*/courses_*/k_*/``. Method code in ``jcrec/`` only.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from time import process_time
from typing import Any, Mapping

from stable_baselines3.common.utils import set_random_seed

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JCREC_DIR = _REPO_ROOT / "jcrec"
_CLIRS_SCRIPTS = _REPO_ROOT / "CLIRS" / "Scripts"

if str(_JCREC_DIR) not in sys.path:
    sys.path.insert(0, str(_JCREC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_config_module():
    path = _CLIRS_SCRIPTS / "load_config.py"
    spec = importlib.util.spec_from_file_location("clirs_load_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load load_config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from Greedy import Greedy
from Optimal import Optimal
from Reinforce import Reinforce
from Dataset import Dataset as JcrecDataset

from Utils.complete_algorithm import CompleteAlgorithmStage, ManifestValidationError
from Utils.experiment_log import ExperimentRunLog
from Utils.results_paths import (
    append_trial_csv_row,
    ensure_experiment_dirs,
    method_slug,
    read_training_life_proxy,
    rl_seed_for_trial,
    trial_artifact_paths,
)

_HEURISTIC = {"greedy": Greedy, "optimal": Optimal}


def _prepare_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    cfg["pipeline"] = "jcrec"
    cfg["results_lineage"] = cfg.get("jcrec_results_lineage", "JCRec")
    cfg["use_clustering"] = False
    if cfg.get("model") in _HEURISTIC:
        cfg["total_steps"] = 0
    return cfg


def _trial_header(config: Mapping[str, Any], trial_id: int, n_learners: int) -> dict:
    return {
        "trial_id": trial_id,
        "data_seed": config.get("seed"),
        "rl_seed": rl_seed_for_trial(config, trial_id),
        "method": method_slug(config),
        "algorithm": config.get("model"),
        "pipeline": "jcrec",
        "total_steps": config.get("total_steps", 0),
        "nb_courses": config.get("nb_courses"),
        "k": config.get("k"),
        "threshold": config.get("threshold"),
        "clustering_reward_shaping": False,
        "evaluation_split": "all_learners",
        "learner_split": {"train_size": 0, "test_size": n_learners},
    }


def _write_trial_artifacts(
    config: Mapping[str, Any],
    *,
    trial_id: int,
    results: dict,
    n_learners: int,
    life: float | None,
    end: float,
    original_applicable_jobs: float,
    run_log: ExperimentRunLog | None = None,
) -> None:
    paths = trial_artifact_paths(config, trial_id)
    training_path = paths["training"] if config.get("save_raw", True) else None

    if config.get("save_raw", True):
        try:
            with open(paths["eval"], "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
        except (OSError, TypeError, ValueError) as exc:
            if run_log is not None:
                run_log.record_exception(exc, trial_id=trial_id, phase="write_eval_json")
            raise

    try:
        csv_path = append_trial_csv_row(
            config,
            {
                "trial_id": trial_id,
                "data_seed": config.get("seed"),
                "rl_seed": rl_seed_for_trial(config, trial_id),
                "method": method_slug(config),
                "algorithm": config.get("model"),
                "total_steps": config.get("total_steps", 0),
                "nb_courses": config.get("nb_courses"),
                "k": config.get("k"),
                "threshold": config.get("threshold"),
                "clustering_reward_shaping": False,
                "life": life,
                "end": end,
                "original_applicable_jobs": original_applicable_jobs,
                "train_size": 0,
                "test_size": n_learners,
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        if run_log is not None:
            run_log.record_exception(exc, trial_id=trial_id, phase="write_sweep_csv")
        raise

    print(f"Trial {trial_id} logged to sweep CSV: {csv_path}")
    if config.get("save_raw", True):
        print(f"Raw eval JSON: {paths['eval']}")
        if os.path.isfile(paths["training"]):
            print(f"Raw training log: {paths['training']}")

    if run_log is not None:
        run_log.check_trial_artifacts(
            trial_id=trial_id,
            eval_path=paths["eval"],
            save_raw=bool(config.get("save_raw", True)),
        )
        run_log.record_trial(
            trial_id=trial_id,
            algorithm=config.get("model", ""),
            life=life,
            end=end,
            training_path=training_path,
        )


def _run_heuristic_trial(solver, dataset, config, trial_id: int, run_log=None) -> None:
    k = int(config["k"])
    threshold = float(config["threshold"])
    algorithm = config["model"]
    n = len(dataset.learners)

    results = _trial_header(config, trial_id, n)
    results["original_attractiveness"] = dataset.get_avg_learner_attractiveness()
    results["original_applicable_jobs"] = dataset.get_avg_applicable_jobs(threshold)
    print(
        f"All learners (n={n}): baseline applicable jobs = "
        f"{results['original_applicable_jobs']:.4f}"
    )

    try:
        time_start = process_time()
        for i in range(n):
            if algorithm == "greedy":
                for _ in range(k):
                    solver.recommend_and_update(i)
            else:
                solver.recommend_and_update(i, k)
        elapsed = process_time() - time_start
    except Exception as exc:
        if run_log is not None:
            run_log.record_exception(exc, trial_id=trial_id, phase="heuristic_eval")
        raise

    results["avg_recommendation_time"] = elapsed / n if n else 0.0
    results["new_attractiveness"] = dataset.get_avg_learner_attractiveness()
    end = dataset.get_avg_applicable_jobs(threshold)
    results["end"] = end
    print(f"All learners: {algorithm} end = {end:.4f}")

    _write_trial_artifacts(
        config,
        trial_id=trial_id,
        results=results,
        n_learners=n,
        life=None,
        end=end,
        original_applicable_jobs=results["original_applicable_jobs"],
        run_log=run_log,
    )


def _init_rl_recommender(dataset, config, trial_id: int, run_log=None):
    try:
        recommender = Reinforce(
            dataset,
            config["model"],
            int(config["k"]),
            float(config["threshold"]),
            trial_id,
            config["total_steps"],
            config["eval_freq"],
        )
    except Exception as exc:
        if run_log is not None:
            run_log.record_exception(exc, trial_id=trial_id, phase="model_init")
        raise

    artifacts = trial_artifact_paths(config, trial_id)
    training_path = artifacts["training"]
    raw_dir = os.path.dirname(training_path)
    saved_results_path = dataset.config["results_path"]

    if config.get("save_raw", True):
        try:
            os.makedirs(raw_dir, exist_ok=True)
            dataset.config["results_path"] = raw_dir
            name = os.path.basename(training_path)
            recommender.all_results_filename = name
            recommender.eval_callback.all_results_filename = name
        except OSError as exc:
            if run_log is not None:
                run_log.record_exception(exc, trial_id=trial_id, phase="training_log_setup")
            raise

    return recommender, saved_results_path, training_path


def _run_rl_trial(
    recommender,
    dataset,
    config,
    trial_id: int,
    saved_results_path: str,
    training_path: str,
    run_log=None,
):
    threshold = float(config["threshold"])
    n = len(dataset.learners)

    results = _trial_header(config, trial_id, n)
    results["original_attractiveness"] = dataset.get_avg_learner_attractiveness()
    results["original_applicable_jobs"] = dataset.get_avg_applicable_jobs(threshold)

    life = None
    try:
        try:
            recommender.model.learn(
                total_timesteps=config["total_steps"],
                callback=recommender.eval_callback,
            )
            life = read_training_life_proxy(training_path)
        except Exception as exc:
            if run_log is not None:
                run_log.record_exception(exc, trial_id=trial_id, phase="rl_training")
            raise
        finally:
            dataset.config["results_path"] = saved_results_path

        time_start = process_time()
        try:
            for i, learner in enumerate(dataset.learners):
                recommender.eval_env.reset(learner=learner)
                done = False
                recommendation_sequence = []
                while not done:
                    obs = recommender.eval_env._get_obs()
                    action, _state = recommender.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = recommender.eval_env.step(action)
                    if reward != -1:
                        recommendation_sequence.append(int(action.item()))
                for course in recommendation_sequence:
                    dataset.learners[i] = recommender.update_learner_profile(
                        learner, dataset.courses[course]
                    )
            elapsed = process_time() - time_start
        except Exception as exc:
            if run_log is not None:
                run_log.record_exception(exc, trial_id=trial_id, phase="rl_evaluation")
            raise
    except Exception:
        dataset.config["results_path"] = saved_results_path
        raise
    results["avg_recommendation_time"] = elapsed / n if n else 0.0
    results["new_attractiveness"] = dataset.get_avg_learner_attractiveness()
    end = dataset.get_avg_applicable_jobs(threshold)
    if life is None:
        life = read_training_life_proxy(training_path)
    results["life"] = life
    results["end"] = end

    _write_trial_artifacts(
        config,
        trial_id=trial_id,
        results=results,
        n_learners=n,
        life=life,
        end=end,
        original_applicable_jobs=results["original_applicable_jobs"],
        run_log=run_log,
    )


def _freeze(config, dirs, run, model, dataset, run_log=None):
    if run != 0:
        return
    try:
        CompleteAlgorithmStage(config, dirs["root"]).ensure(model, dataset)
        print("Complete Algorithm manifest OK (frozen or validated).")
    except ManifestValidationError as exc:
        if run_log is not None:
            run_log.warn(str(exc))
        print(exc)
        print("Hint: use a new experiment cell or delete the existing Results folder.")
        sys.exit(1)
    except Exception as exc:
        if run_log is not None:
            run_log.record_exception(exc, trial_id=run, phase="manifest_freeze")
        raise


def _run_jcrec_trial(
    config,
    dirs,
    run: int,
    run_log: ExperimentRunLog,
) -> None:
    rl_seed = rl_seed_for_trial(config, run)
    set_random_seed(rl_seed)
    config["current_rl_seed"] = rl_seed
    config["current_trial_id"] = run
    print(f"\n--- Trial {run + 1}/{config['nb_runs']} (rl_seed={rl_seed}) ---")

    try:
        dataset = JcrecDataset(config)
        print(dataset)
    except Exception as exc:
        run_log.record_exception(exc, trial_id=run, phase="dataset_load")
        raise

    algorithm = config["model"]

    if algorithm in _HEURISTIC:
        if run == 0:
            _freeze(config, dirs, run, None, dataset, run_log)
        try:
            solver = _HEURISTIC[algorithm](dataset, config["threshold"])
        except Exception as exc:
            run_log.record_exception(exc, trial_id=run, phase="heuristic_init")
            raise
        _run_heuristic_trial(solver, dataset, config, run, run_log)
        return

    recommender, saved_path, training_path = _init_rl_recommender(
        dataset, config, run, run_log
    )
    if run == 0:
        _freeze(config, dirs, run, recommender.model, dataset, run_log)
    _run_rl_trial(
        recommender, dataset, config, run, saved_path, training_path, run_log
    )
def main() -> None:
    parser = argparse.ArgumentParser(description="Run JCRec with CLIRS Results layout.")
    parser.add_argument("--Config", default=r"Config/run.json")
    args = parser.parse_args()

    try:
        load_config = _load_config_module().load_config
        config = _prepare_config(load_config(args.Config))
    except Exception as exc:
        print(f"[ERROR] Failed to load config {args.Config}: {exc}")
        traceback.print_exc()
        sys.exit(1)

    try:
        dirs = ensure_experiment_dirs(config)
    except Exception as exc:
        print(f"[ERROR] Failed to create experiment directories: {exc}")
        traceback.print_exc()
        sys.exit(1)

    with ExperimentRunLog(
        config,
        dirs["root"],
        config_path=args.Config,
        pipeline="jcrec",
        repo_root=str(_REPO_ROOT),
    ) as run_log:
        print(f"Experiment cell: {dirs['root']}")
        print(f"Trials: {config['nb_runs']} (JCRec model={config.get('model')})")

        for run in range(config["nb_runs"]):
            _run_jcrec_trial(config, dirs, run, run_log)

        print(f"\nDone. Results under: {dirs['root']}")


if __name__ == "__main__":
    main()
