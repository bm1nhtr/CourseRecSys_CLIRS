import os
import json
import numpy as np
from time import process_time
from stable_baselines3 import DQN, A2C, PPO

from CourseRecEnv import CourseRecEnv, EvaluateCallback


class Reinforce:
    """Reinforcement Learning-based Course Recommendation System.
    
    This class implements a reinforcement learning approach for course recommendations
    using various RL algorithms from stable-baselines3 with mastery levels and clustering.
    
    The system trains an RL agent to recommend courses to learners with the goal of
    maximizing their job opportunities. The agent learns a policy that maps learner
    skill profiles to course recommendations, considering mastery levels and optional
    clustering-based reward adjustment.
    
    Features:
    - Support for multiple RL algorithms (DQN, A2C, PPO)
    - Mastery levels (1-3) for skills
    - Optional clustering-based reward adjustment
    - Comprehensive evaluation metrics
    
    Attributes:
        dataset: Dataset object containing learners, jobs, and courses data
        model_name (str): Name of the RL algorithm to use ('dqn', 'a2c', or 'ppo')
        k (int): Maximum number of course recommendations per learner
        threshold (float): Minimum matching score required for job applicability
        run (int): Run identifier for experiment tracking
        total_steps (int): Total number of training steps
        eval_freq (int): Frequency of model evaluation during training
    """
    
    def __init__(
        self, dataset, model, k, threshold, run, total_steps=1000, eval_freq=100
    ):  
        """Initialize the reinforcement learning recommendation system.
        
        Args:
            dataset: Dataset object containing the recommendation system data
            model (str): Name of the RL algorithm ('dqn', 'a2c', or 'ppo')
            k (int): Maximum number of course recommendations per learner
            threshold (float): Minimum matching score for job applicability
            run (int): Run identifier for experiment tracking
            total_steps (int, optional): Total training steps. Defaults to 1000.
            eval_freq (int, optional): Evaluation frequency. Defaults to 100.
        """
        self.dataset = dataset
        self.model_name = model
        self.k = k
        self.threshold = threshold
        self.run = run
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        
        # train_env: learn() samples episodes from dataset.train_indices (+ clustering reward).
        # eval_env: val/test CVs passed explicitly; raw job-count reward.
        self.train_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k, is_training=True)
        self.eval_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k, is_training=False)
        self.get_model()
        
        # Check if model uses clustering based on config
        if self.train_env.use_clustering:  # Only use clustering if explicitly enabled
            self.all_results_filename = (
                f"all_{self.model_name}_k_{self.k}_total_steps_{self.total_steps}_clusters_auto_run_{run}.txt"
            )
            self.final_results_filename = (
                f"final_{self.model_name}_k_{self.k}_total_steps_{self.total_steps}_clusters_auto_run_{run}.json"
            )
        else:  # model without clustering
            self.all_results_filename = (
                f"all_{self.model_name}_k_{self.k}_total_steps_{self.total_steps}_run_{run}.txt"
            )
            self.final_results_filename = (
                f"final_{self.model_name}_k_{self.k}_total_steps_{self.total_steps}_run_{run}.json"
            )

        self.eval_callback = EvaluateCallback(
            self.eval_env,
            eval_freq=self.eval_freq,
            all_results_filename=self.all_results_filename,
        )

    def get_model(self):
        """Initialize the reinforcement learning model.
        
        Sets up the specified RL algorithm (DQN, A2C, or PPO) with default parameters.
        The model is configured to use a Multi-Layer Perceptron (MLP) policy.
        
        Supported algorithms:
        - DQN: Deep Q-Network for discrete action spaces
        - A2C: Advantage Actor-Critic for continuous action spaces
        - PPO: Proximal Policy Optimization for both discrete and continuous spaces
        """
        # on training env
        if self.model_name == "dqn":
            self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")
        elif self.model_name == "a2c":
            self.model = A2C(env=self.train_env, verbose=0, policy="MlpPolicy", device="cpu")
        elif self.model_name == "ppo":
            self.model = PPO(env=self.train_env, verbose=0, policy="MlpPolicy")

    def update_learner_profile(self, learner, course):
        """Merge one course's provided skills into a learner profile.

        Uses element-wise maximum so mastery levels never decrease. For a single
        course this is sufficient. When applying multiple courses (k > 1), call
        :meth:`apply_recommendation_sequence` so each course merges into the
        profile left by the previous one.

        Args:
            learner (np.ndarray): Current learner skill vector (not necessarily
                the episode starting profile when used inside a sequence).
            course (np.ndarray): Course skills array ``[required, provided]``.

        Returns:
            np.ndarray: Updated learner skill vector.
        """
        return np.maximum(learner, course[1])

    def apply_recommendation_sequence(self, learner, recommendation_sequence):
        """Build the post-episode learner profile from an ordered course list.

        Each course merges into the profile produced by the previous one via
        ``np.maximum`` on provided skills. Use this whenever ``k > 1`` so the
        stored profile matches the skill state the env reaches after the same
        sequence of ``step`` calls.

        Args:
            learner (np.ndarray): Skill vector at episode reset (before any
                recommendation).
            recommendation_sequence (list): Course indices in the order they
                were recommended; invalid actions are omitted.

        Returns:
            np.ndarray: Updated skill vector for metrics (does not modify ``dataset.learners``).
        """
        updated_profile = learner.copy()
        for course_idx in recommendation_sequence:
            updated_profile = self.update_learner_profile(
                updated_profile, self.dataset.courses[course_idx]
            )
        return updated_profile

    def evaluate_learner_indices(self, indices):
        """Recommend courses for learners at the given row indices.

        Uses ``eval_env`` with deterministic policy. Profiles are returned in
        memory only so ``dataset.learners`` (source CV matrix) stays unchanged.

        Args:
            indices (array-like): Row indices into ``dataset.learners``.

        Returns:
            tuple: ``(recommendations, updated_profiles, elapsed_seconds)`` where
            ``recommendations`` maps CV id → course id list and ``updated_profiles``
            has shape ``(len(indices), n_skills)``.
        """
        time_start = process_time()
        recommendations = {}
        updated_profiles = []

        for matrix_idx in indices:
            learner = self.dataset.learners[matrix_idx].copy()
            self.eval_env.reset(learner=learner)
            done = False
            learner_id = self.dataset.learners_index[matrix_idx]
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
            recommendations[learner_id] = [
                self.dataset.courses_index[course_id]
                for course_id in recommendation_sequence
            ]

        elapsed = process_time() - time_start
        return recommendations, np.array(updated_profiles), elapsed

    def reinforce_recommendation(self):
        """Train and evaluate the RL model for course recommendations.
        
        This method:
        1. Calculates initial metrics:
           - Average learner attractiveness
           - Average number of applicable jobs
        2. Trains the RL model on train-split CV profiles via ``train_env`` (``train_indices``)
        3. Evaluates the model on the **test** learner split:
           - Generates course recommendations
           - Computes post-recommend profiles in memory (does not overwrite CV data)
           - Tracks recommendation time
        4. Calculates final metrics on the test split only
        5. Saves results:
           - Intermediate evaluation results to text file
           - Final metrics and recommendations to JSON file
        
        The results are saved in two files:
        - A text file with intermediate evaluation results during training
        - A JSON file with final metrics and recommendations for each learner
        """
        results = dict()
        test_indices = self.dataset.test_indices

        results["learner_split"] = {
            "train_size": int(len(self.dataset.train_indices)),
            "val_size": int(len(self.dataset.val_indices)),
            "test_size": int(len(test_indices)),
            "train_ratio": self.dataset.config.get("train_ratio", 0.7),
            "val_ratio": self.dataset.config.get("val_ratio", 0.15),
            "test_ratio": self.dataset.config.get("test_ratio", 0.15),
        }
        results["evaluation_split"] = "test"
        results["training_episodes_from"] = "train_split_cv"

        avg_l_attrac_debut = self.dataset.get_avg_learner_attractiveness(test_indices)
        print(
            f"Test split (n={len(test_indices)}): "
            f"average attractiveness = {avg_l_attrac_debut:.2f}"
        )
        results["original_attractiveness"] = avg_l_attrac_debut

        avg_app_j_debut = self.dataset.get_avg_applicable_jobs(self.threshold, test_indices)
        print(
            f"Test split: average applicable jobs per learner = {avg_app_j_debut:.2f}"
        )
        results["original_applicable_jobs"] = avg_app_j_debut

        # Train the model using train env
        self.model.learn(total_timesteps=self.total_steps, callback=self.eval_callback)

        # Final evaluation: held-out test learners only; CV matrix in dataset stays intact.
        recommendations, test_profiles, eval_elapsed = self.evaluate_learner_indices(
            test_indices
        )

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
        results["new_applicable_jobs"] = avg_app_j_fin

        results["recommendations"] = recommendations

        # Create branch directory if it doesn't exist
        branch_dir = os.path.join(self.dataset.config["results_path"], self.dataset.config["branch_name"])
        os.makedirs(branch_dir, exist_ok=True)
        
        # Create data directory for this branch
        data_dir = os.path.join(branch_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        json.dump(
            results,
            open(
                os.path.join(
                    data_dir,
                    self.final_results_filename,
                ),
                "w",
            ),
            indent=4,
        )
