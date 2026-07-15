"""
JCRec RL with CLIRS train/test split — fair ``end`` on ``test_indices``.

Uses ``jcrec/CourseRecEnv`` + CLIRS ``Dataset``. Does not modify ``jcrec/`` sources.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JCREC_DIR = _REPO_ROOT / "jcrec"
if str(_JCREC_DIR) not in sys.path:
    sys.path.insert(0, str(_JCREC_DIR))

from CourseRecEnv import CourseRecEnv as JcrecCourseRecEnv  # noqa: E402

from Utils.results_paths import (
    method_slug,
    read_training_life_proxy,
    rl_seed_for_trial,
    trial_artifact_paths,
    upsert_trial_csv_row,
)


class JcrecTrainSplitEnv(JcrecCourseRecEnv):
    """Training episodes sample real CV learners from ``train_indices``."""

    def reset(self, seed=None, learner=None):
        super(JcrecCourseRecEnv, self).reset(seed=seed)
        if learner is not None:
            self._agent_skills = np.asarray(learner, dtype=np.int32).copy()
        else:
            train_indices = self.dataset.train_indices
            if train_indices is None or len(train_indices) == 0:
                raise ValueError("Dataset missing train_indices for jcrec_fair")
            idx = int(self.np_random.choice(train_indices))
            self._agent_skills = self.dataset.learners[idx].copy()
        self.nb_recommendations = 0
        return self._get_obs(), self._get_info()


class JcrecFairEvaluateCallback(BaseCallback):
    """``life`` proxy on train split (CLIRS-compatible ``*_training.txt`` format)."""

    def __init__(
        self,
        eval_env: JcrecCourseRecEnv,
        eval_freq: int,
        training_log_path: str,
        monitor_indices: np.ndarray,
        *,
        save_raw: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.training_log_path = training_log_path
        self.monitor_indices = np.asarray(monitor_indices, dtype=int)
        self.save_raw = save_raw
        self._mode = "w"

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0 or self.monitor_indices.size == 0:
            return True

        time_start = process_time()
        avg_jobs = 0.0
        for idx in self.monitor_indices:
            learner = self.eval_env.dataset.learners[int(idx)].copy()
            self.eval_env.reset(learner=learner)
            done = False
            tmp_avg_jobs = self.eval_env._get_info()["nb_applicable_jobs"]
            while not done:
                obs = self.eval_env._get_obs()
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                if reward != -1:
                    tmp_avg_jobs = reward
            avg_jobs += tmp_avg_jobs

        mean_jobs = avg_jobs / self.monitor_indices.size
        elapsed = process_time() - time_start
        print(
            f"Iteration {self.n_calls}. Train-split average jobs: "
            f"{mean_jobs:.4f} Time: {elapsed:.4f}"
        )
        if self.save_raw:
            os.makedirs(os.path.dirname(self.training_log_path), exist_ok=True)
            with open(self.training_log_path, self._mode, encoding="utf-8") as f:
                f.write(f"{self.n_calls} {mean_jobs} {elapsed}\n")
            if self._mode == "w":
                self._mode = "a"
        return True


class JcrecFairReinforce:
    """JCRec SB3 algorithms with CLIRS hold-out eval protocol."""

    RL_ALGORITHMS = frozenset({"dqn", "ppo"})

    def __init__(
        self,
        dataset: Any,
        model_name: str,
        k: int,
        threshold: float,
        trial_id: int,
        total_steps: int,
        eval_freq: int,
    ):
        if model_name not in self.RL_ALGORITHMS:
            raise ValueError(f"jcrec_fair supports dqn|ppo, got {model_name!r}")

        self.dataset = dataset
        self.model_name = model_name
        self.k = k
        self.threshold = threshold
        self.trial_id = trial_id
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.config = dataset.config
        self.save_raw = bool(self.config.get("save_raw", True))

        self.train_env = JcrecTrainSplitEnv(dataset, threshold=threshold, k=k)
        self.eval_env = JcrecCourseRecEnv(dataset, threshold=threshold, k=k)
        self._init_model()

        artifacts = trial_artifact_paths(self.config, trial_id)
        self.training_log_path = artifacts["training"]
        self.eval_json_path = artifacts["eval"]
        self.eval_callback = JcrecFairEvaluateCallback(
            self.eval_env,
            eval_freq=eval_freq,
            training_log_path=self.training_log_path,
            monitor_indices=dataset.train_indices,
            save_raw=self.save_raw,
        )

    def _init_model(self) -> None:
        if self.model_name == "dqn":
            self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")
        else:
            self.model = PPO(env=self.train_env, verbose=0, policy="MlpPolicy")

    @staticmethod
    def update_learner_profile(learner: np.ndarray, course) -> np.ndarray:
        return np.maximum(learner, course[1])

    def _apply_recommendation_sequence(
        self, learner: np.ndarray, recommendation_sequence: list[int]
    ) -> np.ndarray:
        profile = learner.copy()
        for course_idx in recommendation_sequence:
            profile = self.update_learner_profile(profile, self.dataset.courses[course_idx])
        return profile

    def evaluate_learner_indices(self, indices: np.ndarray) -> tuple[np.ndarray, float]:
        time_start = process_time()
        updated_profiles: list[np.ndarray] = []
        for matrix_idx in indices:
            learner = self.dataset.learners[int(matrix_idx)].copy()
            self.eval_env.reset(learner=learner)
            done = False
            recommendation_sequence: list[int] = []
            while not done:
                obs = self.eval_env._get_obs()
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                if reward != -1:
                    recommendation_sequence.append(int(action.item()))
            updated_profiles.append(
                self._apply_recommendation_sequence(learner, recommendation_sequence)
            )
        return np.array(updated_profiles), process_time() - time_start

    def run_trial(self) -> dict[str, Any]:
        wall_start = perf_counter()
        test_indices = self.dataset.test_indices
        train_indices = self.dataset.train_indices
        threshold = self.threshold

        results: dict[str, Any] = {
            "trial_id": self.trial_id,
            "data_seed": self.config.get("seed"),
            "rl_seed": rl_seed_for_trial(self.config, self.trial_id),
            "method": method_slug(self.config),
            "algorithm": self.model_name,
            "pipeline": "jcrec_fair",
            "evaluation_split": "test",
            "training_episodes_from": "train_split_cv",
            "learner_split": {
                "holdout": True,
                "train_size": int(len(train_indices)),
                "test_size": int(len(test_indices)),
            },
        }

        results["original_applicable_jobs"] = self.dataset.get_avg_applicable_jobs(
            threshold, test_indices
        )
        print(
            f"Test split (n={len(test_indices)}): baseline applicable jobs = "
            f"{results['original_applicable_jobs']:.4f}"
        )

        self.model.learn(
            total_timesteps=self.total_steps,
            callback=self.eval_callback,
        )
        life = read_training_life_proxy(self.training_log_path)

        test_profiles, eval_elapsed = self.evaluate_learner_indices(test_indices)
        n_test = len(test_indices)
        results["avg_recommendation_time"] = eval_elapsed / n_test if n_test else 0.0
        results["end"] = self.dataset.get_avg_applicable_jobs_for_profiles(
            test_profiles, threshold
        )
        results["life"] = life
        print(f"Test split: jcrec_fair end = {results['end']:.4f}")
        trial_wall_minutes = round((perf_counter() - wall_start) / 60.0, 3)
        results["trial_wall_minutes"] = trial_wall_minutes
        print(f"Trial {self.trial_id} wall time: {trial_wall_minutes:.3f} min")

        if self.save_raw:
            with open(self.eval_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

        upsert_trial_csv_row(
            self.config,
            {
                "trial_id": self.trial_id,
                "data_seed": self.config.get("seed"),
                "rl_seed": results["rl_seed"],
                "method": method_slug(self.config),
                "algorithm": self.model_name,
                "total_steps": self.total_steps,
                "nb_courses": self.config.get("nb_courses"),
                "k": self.k,
                "threshold": threshold,
                "clustering_reward_shaping": False,
                "evaluation_split": "test",
                "life": life,
                "end": results["end"],
                "original_applicable_jobs": results["original_applicable_jobs"],
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "trial_wall_minutes": trial_wall_minutes,
            },
        )
        return results
