"""
Clustering Module for Course Recommendation System

This module implements clustering functionality to improve RL performance by adjusting rewards
based on course cluster membership. The clustering helps identify similar courses based on their
required and provided skills, which is then used to modify the reward signal to encourage more stable learning.

The clustering is based on five key features for each course:
1. Coverage: Overall skill coverage ratio (average of required and provided skill coverage)
2. Required Entropy: Diversity of required skills (entropy-based measure)
3. Provided Entropy: Diversity of provided skills (entropy-based measure)
4. Avg Level Gap: Average difference between required and provided skill levels
5. Max Level Gap: Maximum difference between required and provided skill levels

Reward shaping (CLIRS): compare each course base reward against the persistent
adjusted reference R'_adjusted,ref (the last bonused adjusted reward in the sequence).
1. First recommendation C_1: fixed bonus (first_recommendation multiplier)
2. Later steps: if R_base > R'_adjusted,ref, apply progress_increase and update the ref
3. Otherwise: no_improvement multiplier; ref unchanged
   



"""

import json
import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt

FEATURE_SPEC = "coverage_entropy_active_levelgap_v2"
ARTIFACT_SCHEMA_VERSION = 1

class CourseClusterer:
    """Class for clustering courses and adjusting rewards based on cluster membership.
    
    This class implements a clustering-based reward adjustment mechanism that helps
    stabilize and improve the learning process in the RL environment. It clusters
    courses based on their provided skills and uses this information to modify
    rewards according to predefined rules.
    
    Attributes:
        n_clusters (int): Number of clusters to create
        course_clusters (np.ndarray): Array of cluster assignments for each course
        scaler (StandardScaler): Scaler for normalizing features before clustering
        features (np.ndarray): Original features used for clustering
        random_state (int): Random seed for reproducibility
        auto_clusters (bool): Whether to automatically determine optimal number of clusters
        max_clusters (int): Maximum number of clusters to try when using elbow method
        optimal_k (int): Optimal number of clusters determined by elbow method
        clustering_dir (str): Directory to save clustering Results
        reward_multipliers (dict): Dictionary of reward adjustment multipliers
        best_reward_so_far (float): R'_adjusted,ref for the current recommendation sequence
    """
    
    def __init__(
        self,
        random_state=42,
        auto_clusters=False,
        max_clusters=10,
        config=None,
        clustering_dir=None,
        reports_dir=None,
        selection_method="silhouette",
        min_cluster_size=5,
        max_level=3,
        fixed_n_clusters=5,
    ):
        """Initialize the clusterer.

        When ``auto_clusters`` is true (default CLIRS path), k is chosen from data.
        ``fixed_n_clusters`` applies only when ``auto_clusters`` is false.
        """
        self.n_clusters = fixed_n_clusters
        self.course_clusters = None
        self.scaler = StandardScaler()
        self.features = None
        self.random_state = random_state
        self.auto_clusters = auto_clusters
        self.max_clusters = max_clusters
        self.optimal_k = None
        self.selection_method = selection_method
        self.min_cluster_size = min_cluster_size
        self.max_level = max_level
        self.reports_dir = reports_dir
        self.selection_report = None
        self.quality_metrics = None
        self.best_reward_so_far = 0.0  # R'_adjusted,ref for the current k-step sequence

        cfg = config or {}
        # CLIRS keys from Config/run.json clustering.reward_multipliers
        self.reward_multipliers = {
            "first_recommendation": cfg.get("first_recommendation", 1.3),
            "progress_increase": cfg.get(
                "progress_increase", cfg.get("diff_cluster_increase", 1.3)
            ),
            "no_improvement": cfg.get("no_improvement", 1.0),
        }
        
        if clustering_dir:
            self.clustering_dir = clustering_dir
        else:
            self.clustering_dir = os.path.join("Results", "plots", "clustering")
        os.makedirs(self.clustering_dir, exist_ok=True)
        if self.reports_dir:
            os.makedirs(self.reports_dir, exist_ok=True)

    def _build_features(self, courses):
        """Extract course features for clustering (see FEATURE_SPEC)."""
        required_skills = courses[:, 0]
        provided_skills = courses[:, 1]
        n_skills = required_skills.shape[1]
        max_level = self.max_level

        required_coverage = np.sum(required_skills, axis=1) / (n_skills * max_level)
        provided_coverage = np.sum(provided_skills, axis=1) / (n_skills * max_level)
        coverage = (required_coverage + provided_coverage) / 2

        required_sums = np.sum(required_skills, axis=1, keepdims=True)
        provided_sums = np.sum(provided_skills, axis=1, keepdims=True)
        required_distribution = required_skills / (required_sums + 1e-10)
        provided_distribution = provided_skills / (provided_sums + 1e-10)
        required_entropy = np.where(
            required_sums.ravel() > 0,
            -np.sum(
                required_distribution * np.log2(required_distribution + 1e-10),
                axis=1,
            ),
            0.0,
        )
        provided_entropy = np.where(
            provided_sums.ravel() > 0,
            -np.sum(
                provided_distribution * np.log2(provided_distribution + 1e-10),
                axis=1,
            ),
            0.0,
        )

        level_gap = np.abs(provided_skills - required_skills)
        active = (required_skills > 0) | (provided_skills > 0)
        avg_level_gap = np.zeros(len(courses), dtype=float)
        max_level_gap = np.zeros(len(courses), dtype=float)
        for i in range(len(courses)):
            if active[i].any():
                gaps = level_gap[i, active[i]]
                avg_level_gap[i] = float(np.mean(gaps))
                max_level_gap[i] = float(np.max(gaps))

        return np.column_stack(
            [coverage, required_entropy, provided_entropy, avg_level_gap, max_level_gap]
        )

    def _evaluate_k(self, features_scaled, k):
        """Score one k for cluster selection."""
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(features_scaled)
        sizes = np.bincount(labels, minlength=k)
        min_size = int(sizes.min()) if len(sizes) else 0
        silhouette = (
            float(silhouette_score(features_scaled, labels))
            if k > 1 and len(features_scaled) > k
            else float("nan")
        )
        return {
            "k": int(k),
            "inertia": float(kmeans.inertia_),
            "silhouette": silhouette,
            "cluster_sizes": [int(s) for s in sizes],
            "min_cluster_size": min_size,
        }

    def _pick_k_from_elbow(self, records):
        """Elbow on inertia over valid k candidates (k >= 2)."""
        ks = [r["k"] for r in records]
        inertias = [r["inertia"] for r in records]
        if len(inertias) < 3:
            return records[-1]["k"]
        changes = np.diff(inertias)
        changes_r = np.diff(changes)
        idx = int(np.argmax(changes_r))
        return ks[idx + 2]

    def select_n_clusters(self, features_scaled):
        """Choose k (fixed or auto) using silhouette or elbow; k starts at 2."""
        n_samples = len(features_scaled)
        max_k = min(self.max_clusters, n_samples - 1)
        if max_k < 2:
            return 1, []

        if not self.auto_clusters:
            k = min(max(int(self.n_clusters), 1), n_samples)
            record = self._evaluate_k(features_scaled, k)
            self._plot_selection_curve([record], k)
            return k, [record]

        print("\nSelecting number of clusters...")
        records = [
            self._evaluate_k(features_scaled, k) for k in range(2, max_k + 1)
        ]
        valid = [
            r
            for r in records
            if r["min_cluster_size"] >= self.min_cluster_size
        ]
        if not valid:
            valid = records

        if self.selection_method == "elbow":
            chosen_k = self._pick_k_from_elbow(valid)
        else:
            chosen_k = max(
                valid,
                key=lambda r: (
                    r["silhouette"] if not np.isnan(r["silhouette"]) else -1.0
                ),
            )["k"]

        self._plot_selection_curve(records, chosen_k)
        print(
            f"Selected k={chosen_k} via {self.selection_method} "
            f"(min_cluster_size={self.min_cluster_size})"
        )
        return chosen_k, records

    def _plot_selection_curve(self, records, chosen_k):
        """Save inertia/silhouette vs k plot."""
        if not records:
            return
        ks = [r["k"] for r in records]
        inertias = [r["inertia"] for r in records]
        silhouettes = [r["silhouette"] for r in records]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(ks, inertias, "bx-", label="Inertia")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia")
        ax1.axvline(x=chosen_k, color="r", linestyle="--", label=f"Selected k = {chosen_k}")

        ax2 = ax1.twinx()
        ax2.plot(ks, silhouettes, "g+-", label="Silhouette")
        ax2.set_ylabel("Silhouette")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        ax1.set_title("Cluster Selection")
        fig.tight_layout()
        plot_path = os.path.join(self.clustering_dir, "cluster_selection_curve.png")
        plt.savefig(plot_path)
        plt.close()

    def _compute_quality_metrics(self, features_scaled):
        """Post-fit cluster quality metrics."""
        labels = self.course_clusters
        k = self.n_clusters
        sizes = np.bincount(labels, minlength=k)
        silhouette = (
            float(silhouette_score(features_scaled, labels))
            if k > 1 and len(features_scaled) > k
            else float("nan")
        )
        davies_bouldin = (
            float(davies_bouldin_score(features_scaled, labels))
            if k > 1
            else float("nan")
        )
        balance = float(sizes.min() / sizes.max()) if sizes.max() > 0 else 0.0
        feature_names = [
            "Coverage",
            "Required Entropy",
            "Provided Entropy",
            "Avg Level Gap",
            "Max Level Gap",
        ]
        per_cluster = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_feats = self.features[mask]
            per_cluster.append(
                {
                    "id": int(cluster_id),
                    "n": int(sizes[cluster_id]),
                    "mean_features": {
                        name: float(cluster_feats[:, j].mean())
                        for j, name in enumerate(feature_names)
                    },
                }
            )
        return {
            "n_clusters": int(k),
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "inertia": float(self.inertia_),
            "cluster_sizes": {str(i): int(s) for i, s in enumerate(sizes)},
            "size_balance_ratio": balance,
            "per_cluster": per_cluster,
        }

    def write_reports(self):
        """Write cluster_selection.json and cluster_quality.json under reports_dir."""
        if not self.reports_dir:
            return
        os.makedirs(self.reports_dir, exist_ok=True)
        if self.selection_report is not None:
            with open(
                os.path.join(self.reports_dir, "cluster_selection.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "selection_method": self.selection_method,
                        "min_cluster_size": self.min_cluster_size,
                        "chosen_k": self.n_clusters,
                        "candidates": self.selection_report,
                    },
                    f,
                    indent=2,
                )
        if self.quality_metrics is not None:
            with open(
                os.path.join(self.reports_dir, "cluster_quality.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.quality_metrics, f, indent=2)

    def to_artifact_payload(
        self,
        *,
        data_seed,
        nb_courses,
        courses_index=None,
    ):
        """Serialize fitted state for course_clusters.json."""
        labels = [int(x) for x in self.course_clusters.tolist()]
        labels_by_course_id = {}
        if courses_index:
            for idx, cluster_id in enumerate(labels):
                course_id = courses_index.get(idx)
                if course_id is not None:
                    labels_by_course_id[str(course_id)] = cluster_id
        sizes = np.bincount(self.course_clusters, minlength=self.n_clusters)
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "feature_spec": FEATURE_SPEC,
            "data_seed": data_seed,
            "nb_courses": int(nb_courses),
            "auto_clusters": bool(self.auto_clusters),
            "n_clusters_fitted": int(self.n_clusters),
            "selection_method": self.selection_method,
            "min_cluster_size": int(self.min_cluster_size),
            "max_clusters": int(self.max_clusters),
            "random_state": int(self.random_state),
            "labels_by_index": labels,
            "labels_by_course_id": labels_by_course_id,
            "cluster_sizes": {str(i): int(s) for i, s in enumerate(sizes)},
            "inertia": float(self.inertia_),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "selection_report": self.selection_report,
            "quality_metrics": self.quality_metrics,
        }

    @classmethod
    def from_artifact(cls, path, **runtime_kwargs):
        """Restore a fitted clusterer from course_clusters.json."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        clusterer = cls(
            random_state=payload.get("random_state", 42),
            auto_clusters=payload.get("auto_clusters", False),
            max_clusters=runtime_kwargs.get("max_clusters", 10),
            config=runtime_kwargs.get("config"),
            clustering_dir=runtime_kwargs.get("clustering_dir"),
            reports_dir=runtime_kwargs.get("reports_dir"),
            selection_method=payload.get("selection_method", "silhouette"),
            min_cluster_size=payload.get("min_cluster_size", 5),
            max_level=runtime_kwargs.get("max_level", 3),
            fixed_n_clusters=payload["n_clusters_fitted"],
        )
        clusterer.n_clusters = payload["n_clusters_fitted"]
        clusterer.max_clusters = payload.get(
            "max_clusters", runtime_kwargs.get("max_clusters", 10)
        )
        if payload.get("auto_clusters"):
            clusterer.optimal_k = payload["n_clusters_fitted"]
        clusterer.course_clusters = np.array(
            payload["labels_by_index"], dtype=int
        )
        clusterer.inertia_ = payload.get("inertia")
        clusterer.selection_report = payload.get("selection_report")
        clusterer.quality_metrics = payload.get("quality_metrics")
        clusterer.scaler.mean_ = np.array(payload["scaler_mean"], dtype=float)
        clusterer.scaler.scale_ = np.array(payload["scaler_scale"], dtype=float)
        clusterer.scaler.n_features_in_ = len(clusterer.scaler.mean_)
        return clusterer
        
    def visualize_feature_pairs(self, features_scaled):
        """Visualize relationships between features using correlation matrix.
        
        Args:
            features_scaled: Scaled features used for clustering
        """
        print("\nStarting visualize_feature_pairs...")
        # Create DataFrame for easier plotting
        feature_names = ['Coverage', 'Required Entropy', 'Provided Entropy', 
                        'Avg Level Gap', 'Max Level Gap']
        df = pd.DataFrame(features_scaled, columns=feature_names)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create figure with larger size
        plt.figure(figsize=(10, 8))
        
        # Plot correlation matrix
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values with bold text
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                        fontsize=12,
                        fontweight='bold')
        
        # Add colorbar with larger font
        cbar = plt.colorbar(label='Correlation Coefficient')
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel('Correlation Coefficient', fontsize=14, fontweight='bold')
        
        # Add labels with larger font
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(range(len(feature_names)), feature_names, fontsize=12, fontweight='bold')
        
        # Add title with larger font
        plt.title('Feature Correlation Matrix', pad=20, fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.clustering_dir, 'feature_correlation.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved correlation matrix to: {plot_path}")
        print("Calling visualize_cluster_correlations...")
        
        # Also visualize correlations for each cluster
        self.visualize_cluster_correlations(features_scaled, feature_names)

    def visualize_cluster_correlations(self, features_scaled, feature_names):
        """Visualize feature correlations for each cluster separately."""
        print("\nStarting visualize_cluster_correlations...")
        print(f"Number of clusters: {len(np.unique(self.course_clusters))}")
        print(f"Shape of features_scaled: {features_scaled.shape}")
        
        # Create DataFrame with features and cluster assignments
        df = pd.DataFrame(features_scaled, columns=feature_names)
        df['Cluster'] = self.course_clusters
        
        # Calculate number of rows and columns for subplot grid
        n_clusters = len(np.unique(self.course_clusters))
        n_cols = min(3, n_clusters)  # Maximum 3 columns
        n_rows = (n_clusters + n_cols - 1) // n_cols  # Ceiling division
        
        print(f"Creating subplot grid with {n_rows} rows and {n_cols} columns")
        
        # Create figure with subplots - increased figure size for better resolution
        fig = plt.figure(figsize=(8*n_cols, 7*n_rows), dpi=300)
        
        # Set global font properties
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        
        # Plot correlation matrix for each cluster
        for i in range(n_clusters):
            print(f"\nProcessing cluster {i}...")
            # Get data for current cluster
            cluster_data = df[df['Cluster'] == i][feature_names]
            print(f"Number of courses in cluster {i}: {len(cluster_data)}")
            
            # Calculate correlation matrix
            corr_matrix = cluster_data.corr()
            
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            # Plot correlation matrix with darker colors
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add correlation values with larger, bolder font
            for j in range(len(feature_names)):
                for k in range(len(feature_names)):
                    value = corr_matrix.iloc[j, k]
                    # Make text larger and bolder
                    ax.text(k, j, f'{value:.2f}',
                           ha='center', va='center',
                           color='white' if abs(value) > 0.5 else 'black',
                           fontsize=14,
                           fontweight='bold')
            
            # Add labels with larger, bolder font
            ax.set_xticks(range(len(feature_names)))
            ax.set_yticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
            ax.set_yticklabels(feature_names, fontsize=12, fontweight='bold')
            
            # Add title with larger, bolder font
            ax.set_title(f'Cluster {i} (n={len(cluster_data)})', pad=20, fontsize=16, fontweight='bold')
            
            # Add grid for better readability
            ax.grid(False)
            
            # Make the plot square
            ax.set_aspect('equal')
        
        # Add colorbar with larger font
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel('Correlation Coefficient', fontsize=14, fontweight='bold')
        
        # Add main title
        fig.suptitle('Feature Correlations by Cluster', fontsize=20, fontweight='bold', y=0.95)
        
        # Adjust layout with more padding
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save plot with high DPI for better quality
        plot_path = os.path.join(self.clustering_dir, 'cluster_correlations.png')
        print(f"\nSaving cluster correlation matrices to: {plot_path}")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        print(f"\nSaved cluster correlation matrices to: {plot_path}")
        
        # Print summary statistics for each cluster
        print("\nCluster Feature Statistics:")
        for i in range(n_clusters):
            cluster_data = df[df['Cluster'] == i][feature_names]
            print(f"\nCluster {i} (n={len(cluster_data)}):")
            print(cluster_data.describe().round(3))
        
    def fit_course_clusters(self, courses, courses_index=None, *, write_plots=True):
        """Fit clusters for courses based on their required and provided skills.
        
        Args:
            courses: Array of courses, each containing required and provided skills
            courses_index: Optional index→course_id map from Dataset
            write_plots: When False, skip visualization (loaded-from-artifact path)
        """
        print("\nStarting course clustering...")
        self.features = self._build_features(courses)
        features_scaled = self.scaler.fit_transform(self.features)

        chosen_k, self.selection_report = self.select_n_clusters(features_scaled)
        if self.auto_clusters:
            self.optimal_k = chosen_k
        self.n_clusters = chosen_k

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.course_clusters = kmeans.fit_predict(features_scaled)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        self.quality_metrics = self._compute_quality_metrics(features_scaled)

        print("\nCluster Information:")
        for i in range(self.n_clusters):
            n_formations = np.sum(self.course_clusters == i)
            print(f"Cluster {i}: {n_formations} formations")

        if write_plots:
            self.visualize_clusters(features_scaled)
            self.visualize_feature_pairs(features_scaled)

        self.write_reports()
        
    def visualize_clusters(self, features_scaled):
        """Visualize the clusters using PCA for dimensionality reduction.
        
        This method:
        1. Reduces the 5D feature space to 2D using PCA
        2. Creates a scatter plot of courses in the reduced space
        3. Shows cluster centers and explained variance
        4. Prints feature contributions to each principal component
        
        The PCA components (PC1, PC2) are linear combinations of the original features.
        The coefficients (feature contributions) show:
        - Positive values: Feature increases when the PC increases
        - Negative values: Feature decreases when the PC increases
        - Magnitude: How strongly the feature influences the PC
        
        For example, if PC1 has:
        - Coverage: 0.562 (positive)
        - Provided Entropy: -0.516 (negative)
        This means courses with high PC1 values tend to have:
        - High coverage
        - Low provided entropy
        """
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create figure with larger size and higher DPI
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Plot clusters
        for i in range(self.n_clusters):
            cluster_points = features_2d[self.course_clusters == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f'Cluster {i} ({len(cluster_points)} courses)',
                alpha=0.7,
                s=100
            )
        
        # Plot cluster centers
        centers_2d = pca.transform(self.cluster_centers_)
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c='black',
            marker='x',
            s=200,
            linewidths=3,
            label='Cluster Centers'
        )
        
        # Add labels and title with larger font size
        plt.xlabel('First Principal Component', fontsize=14, fontweight='bold')
        plt.ylabel('Second Principal Component', fontsize=14, fontweight='bold')
        plt.title('Course Clusters (PCA Visualization)', fontsize=16, fontweight='bold', pad=20)
        
        # Add explained variance ratio with larger font
        explained_variance = pca.explained_variance_ratio_
        plt.figtext(
            0.73, 0.02,
            f'Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5)
        )
        
        # Add legend outside the plot with larger font
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            prop={'size': 12, 'weight': 'bold'},
            frameon=True,
            title='Clusters',
            title_fontsize=14
        )
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot with high quality
        plot_path = os.path.join(self.clustering_dir, f'cluster_visualization_pca.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        # Print feature contributions to principal components
        print("\nFeature Contribution to Principal Components:")
        print("(Values show how much each feature contributes to the principal components)")
        print("(Positive values mean the feature increases with the PC, negative values mean it decreases)")
        print("(The magnitude shows how strongly the feature influences the PC)")
        for i, component in enumerate(pca.components_):
            print(f"\nPC{i+1} (Explained variance: {pca.explained_variance_ratio_[i]:.2%}):")
            for j, feature in enumerate(['Coverage', 'Required Entropy', 'Provided Entropy', 
                                      'Avg Level Gap', 'Max Level Gap']):
                print(f"{feature}: {component[j]:.3f}")
        
    def adjust_reward(self, course_idx, original_reward, prev_reward):
        """CLIRS persistent adjusted reference reward shaping.

        Compares each course base reward (R_base) against R'_adjusted,ref stored in
        ``best_reward_so_far``. The reference only advances when a bonus is applied.

        Args:
            course_idx (int): Index of the current course (unused; kept for API stability)
            original_reward (float): Base reward (applicable jobs) from the environment
            prev_reward (float): Adjusted reward from the previous step in the sequence

        Returns:
            float: Shaped reward passed to the RL agent
        """
        if self.course_clusters is None:
            return original_reward

        multipliers = self.reward_multipliers
        first_step = prev_reward is None or prev_reward == 0

        if first_step:
            adjusted_reward = original_reward * multipliers["first_recommendation"]
            self.best_reward_so_far = adjusted_reward
            return adjusted_reward

        if original_reward > self.best_reward_so_far:
            adjusted_reward = original_reward * multipliers["progress_increase"]
            self.best_reward_so_far = adjusted_reward
            return adjusted_reward

        return original_reward * multipliers["no_improvement"] 