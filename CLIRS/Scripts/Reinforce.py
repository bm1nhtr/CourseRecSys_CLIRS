import json
import sys
from pathlib import Path
from time import process_time

import numpy as np
from stable_baselines3 import DQN, A2C, PPO

from CourseRecEnv import CourseRecEnv, EvaluateCallback

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Utils.results_paths import (
    append_trial_csv_row,
    method_slug,
    read_training_life_proxy,
    rl_seed_for_trial,
    trial_artifact_paths,
)


class Reinforce:
    """Reinforcement Learning-based Course Recommendation System."""

    def __init__(
        self, dataset, model, k, threshold, run, total_steps=1000, eval_freq=100
    ):
        self.dataset = dataset
        self.model_name = model
        self.k = k
        self.threshold = threshold
        self.run = run
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.config = dataset.config
        self.save_raw = bool(self.config.get("save_raw", True))

        self.train_env = CourseRecEnv(
            dataset, threshold=self.threshold, k=self.k, is_training=True
        )
        self.eval_env = CourseRecEnv(
            dataset, threshold=self.threshold, k=self.k, is_training=False
        )
        self.get_model()

        artifacts = trial_artifact_paths(self.config, run)
        self.training_log_path = artifacts["training"]
        self.eval_json_path = artifacts["eval"]

        self.eval_callback = EvaluateCallback(
            self.eval_env,
            eval_freq=self.eval_freq,
            training_log_path=self.training_log_path,
            save_raw=self.save_raw,
        )

    def get_model(self):
        if self.model_name == "dqn":
            self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")
        elif self.model_name == "a2c":
            self.model = A2C(
                env=self.train_env, verbose=0, policy="MlpPolicy", device="cpu"
            )
        elif self.model_name == "ppo":
            self.model = PPO(env=self.train_env, verbose=0, policy="MlpPolicy")

    def update_learner_profile(self, learner, course):
        return np.maximum(learner, course[1])

    def apply_recommendation_sequence(self, learner, recommendation_sequence):
        updated_profile = learner.copy()
        for course_idx in recommendation_sequence:
            updated_profile = self.update_learner_profile(
                updated_profile, self.dataset.courses[course_idx]
            )
        return updated_profile

    def evaluate_learner_indices(self, indices):
        time_start = process_time()
        updated_profiles = []

        for matrix_idx in indices:
            learner = self.dataset.learners[matrix_idx].copy()
            self.eval_env.reset(learner=learner)
            done = False
            recommendation_sequence = []

            while not done:
                obs = self.eval_env._get_obs()
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                if reward != -1:
                    recommendation_sequence.append(action.item())

            updated_profiles.append(
                self.apply_recommendation_sequence(learner, recommendation_sequence)
            )

        elapsed = process_time() - time_start
        return np.array(updated_profiles), elapsed

    def reinforce_recommendation(self):
        results = {}
        test_indices = self.dataset.test_indices
        trial_id = self.run
        rl_seed = rl_seed_for_trial(self.config, trial_id)

        results["trial_id"] = trial_id
        results["data_seed"] = self.config.get("seed")
        results["rl_seed"] = rl_seed
        results["method"] = method_slug(self.config)
        results["algorithm"] = self.model_name
        results["total_steps"] = self.total_steps
        results["nb_courses"] = self.config.get("nb_courses")
        results["k"] = self.k
        results["threshold"] = self.threshold
        results["clustering_reward_shaping"] = self.train_env.use_clustering

        results["learner_split"] = {
            "train_size": int(len(self.dataset.train_indices)),
            "test_size": int(len(test_indices)),
            "train_ratio": self.dataset.config.get("train_ratio", 0.7),
            "test_ratio": self.dataset.config.get("test_ratio", 0.3),
        }
        results["evaluation_split"] = "test"
        results["training_episodes_from"] = "train_split_cv"

        avg_l_attrac_debut = self.dataset.get_avg_learner_attractiveness(test_indices)
        print(
            f"Test split (n={len(test_indices)}): "
            f"average attractiveness = {avg_l_attrac_debut:.2f}"
        )
        results["original_attractiveness"] = avg_l_attrac_debut

        avg_app_j_debut = self.dataset.get_avg_applicable_jobs(
            self.threshold, test_indices
        )
        print(
            f"Test split: average applicable jobs per learner = {avg_app_j_debut:.2f}"
        )
        results["original_applicable_jobs"] = avg_app_j_debut

        # Train on train_env (optional clustering shaping). Callback logs train-split
        # progress to *_training.txt — used later for metric ``life``.
        self.model.learn(total_timesteps=self.total_steps, callback=self.eval_callback)

        # Final eval on held-out test learners only → metric ``end`` (primary).
        test_profiles, eval_elapsed = self.evaluate_learner_indices(test_indices)

        n_test = len(test_indices)
        avg_recommendation_time = eval_elapsed / n_test if n_test else 0.0
        print(f"Average recommendation time (test): {avg_recommendation_time:.4f} seconds")
        results["avg_recommendation_time"] = avg_recommendation_time

        avg_l_attrac_fin = self.dataset.get_avg_learner_attractiveness_for_profiles(
            test_profiles
        )
        print(f"Test split: new average attractiveness = {avg_l_attrac_fin:.2f}")
        results["new_attractiveness"] = avg_l_attrac_fin

        avg_app_j_fin = self.dataset.get_avg_applicable_jobs_for_profiles(
            test_profiles, self.threshold
        )
        print(f"Test split: new average applicable jobs = {avg_app_j_fin:.2f}")
        life = read_training_life_proxy(self.training_log_path)
        results["life"] = life
        results["end"] = avg_app_j_fin

        if self.save_raw:
            with open(self.eval_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

        csv_path = append_trial_csv_row(
            self.config,
            {
                "trial_id": trial_id,
                "data_seed": self.config.get("seed"),
                "rl_seed": rl_seed,
                "method": method_slug(self.config),
                "algorithm": self.model_name,
                "total_steps": self.total_steps,
                "nb_courses": self.config.get("nb_courses"),
                "k": self.k,
                "threshold": self.threshold,
                "clustering_reward_shaping": self.train_env.use_clustering,
                "life": life,
                "end": avg_app_j_fin,
                "original_applicable_jobs": avg_app_j_debut,
                "train_size": len(self.dataset.train_indices),
                "test_size": len(test_indices),
            },
        )
        print(f"Trial {trial_id} logged to sweep CSV: {csv_path}")
        if self.save_raw:
            print(f"Raw eval JSON: {self.eval_json_path}")
