"""
Learning curve plots from experiment raw logs.

Reads training logs under Results/{lineage}/steps_*/data_*/courses_*/raw/*_training.txt
(layout from Utils.results_paths). Plots are written to .../plots/.

Usage (repo root):
    poetry run python Utils/visualize_learning_curves.py
    poetry run python Utils/visualize_learning_curves.py --config Config/run.json
"""

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SCRIPTS = os.path.join(_REPO_ROOT, "CLIRS", "Scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from load_config import load_config
from Utils.results_paths import ensure_experiment_dirs


def load_experiment_layout(config_path=None):
    """Load config and return (config, dirs) for the active experiment cell."""
    if config_path is None:
        config_path = os.path.join(_REPO_ROOT, "Config", "run.json")
    config = load_config(config_path)
    dirs = ensure_experiment_dirs(config, write_manifest=False)
    return config, dirs


def get_plots_directory(layout_dirs):
    """Return plots/ under the experiment cell."""
    plots_dir = layout_dirs["plots"]
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _raw_training_files(raw_dir):
    return glob.glob(os.path.join(raw_dir, "*_training.txt"))


def _parse_training_filename(path):
    """Parse clirs_dqn_data42_rl43_k2_training.txt into parts."""
    name = os.path.basename(path).replace("_training.txt", "")
    match = re.match(
        r"^(?P<method>clirs|baseline)_(?P<algo>\w+)_data\d+_rl\d+_k(?P<k>\d+)$",
        name,
    )
    if not match:
        return None
    return match.groupdict()


def load_evaluation_data(file_path):
    """Load (steps, metrics, start_step) from a training log text file."""
    try:
        data = np.loadtxt(file_path)
        if len(data.shape) < 2 or data.shape[1] < 2:
            print(f"Warning: Invalid data format in {file_path}")
            return None, None, None

        steps = data[:, 0]
        metrics = data[:, 1]
        start_step = steps[0]
        steps = steps - start_step
        return steps, metrics, start_step
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None, None


def get_experiment_title(model_name, k, is_clustered):
    """Generate a descriptive title for the experiment."""
    model_map = {"dqn": "DQN", "ppo": "PPO"}
    model = model_map.get(model_name.lower(), model_name.upper())
    clustering_info = "with Clustering" if is_clustered else "without Clustering"
    return f"{model} - {clustering_info} (k={k})"


def plot_learning_curves(layout_dirs):
    """Plot learning curves from raw/*_training.txt files."""
    raw_dir = layout_dirs.get("raw", os.path.join(layout_dirs["root"], "raw"))
    result_files = _raw_training_files(raw_dir)

    if not result_files:
        print(f"\nWarning: No training logs found in {raw_dir}")
        return

    plots_dir = get_plots_directory(layout_dirs)

    plt.rcParams.update({
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })

    existing_plots = glob.glob(os.path.join(plots_dir, "*.png"))
    print(f"\nFound {len(existing_plots)} existing plots in {plots_dir}")
    print(f"\nFound {len(result_files)} result files to process...")

    new_plots = 0
    for file_path in result_files:
        filename = os.path.basename(file_path)
        parsed = _parse_training_filename(file_path)
        if not parsed:
            print(f"Skipping unrecognized filename: {filename}")
            continue

        model_name = parsed["algo"]
        k = parsed["k"]
        has_clustering = parsed["method"] == "clirs"

        clustering_suffix = "clustered" if has_clustering else "no_clustering"
        output_filename = f"{model_name}_mastery_levels_{clustering_suffix}_k{k}.png"
        output_path = os.path.join(plots_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping existing plot: {output_filename}")
            continue

        print(f"Creating new plot: {output_filename}...")
        new_plots += 1

        steps, metrics, _ = load_evaluation_data(file_path)
        if steps is None:
            continue

        plt.figure(figsize=(15, 8))
        plt.plot(steps, metrics, "b-", linewidth=2, label="Average Applicable Jobs")
        plt.xlabel("Training Steps", fontsize=14, fontweight="bold")
        plt.ylabel("Average Applicable Jobs", fontsize=14, fontweight="bold")
        plt.title(
            get_experiment_title(model_name, k, has_clustering),
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc="lower right", bbox_to_anchor=(1.0, 0.0))
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.ticklabel_format(style="plain", axis="y")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nSummary:")
    print(f"- Existing plots: {len(existing_plots)}")
    print(f"- New plots created: {new_plots}")
    print(f"- Total plots after update: {len(existing_plots) + new_plots}")


def compare_clustering_effect(layout_dirs, model_name, k):
    """Compare clirs vs baseline training logs for one algorithm and k."""
    plots_dir = get_plots_directory(layout_dirs)
    raw_dir = layout_dirs.get("raw", os.path.join(layout_dirs["root"], "raw"))

    clustered_files = glob.glob(
        os.path.join(raw_dir, f"clirs_{model_name}_data*_rl*_k{k}_training.txt")
    )
    no_cluster_files = glob.glob(
        os.path.join(raw_dir, f"baseline_{model_name}_data*_rl*_k{k}_training.txt")
    )

    if not clustered_files or not no_cluster_files:
        print(
            f"\nWarning: Need both clirs_* and baseline_* logs for {model_name} k={k} in {raw_dir}"
        )
        return

    clustered_steps, clustered_metrics, clustered_start = load_evaluation_data(
        clustered_files[0]
    )
    no_cluster_steps, no_cluster_metrics, no_cluster_start = load_evaluation_data(
        no_cluster_files[0]
    )
    if clustered_steps is None or no_cluster_steps is None:
        return

    plt.figure(figsize=(15, 8))
    plt.plot(
        clustered_steps,
        clustered_metrics,
        "b-",
        linewidth=2,
        label=f"CLIRS (from step {clustered_start:,})",
    )
    plt.plot(
        no_cluster_steps,
        no_cluster_metrics,
        "r--",
        linewidth=2,
        label=f"Baseline (from step {no_cluster_start:,})",
    )
    plt.xlabel("Training Steps (normalized)", fontsize=14, fontweight="bold")
    plt.ylabel("Average Applicable Jobs", fontsize=14, fontweight="bold")
    plt.title(
        f"{model_name.upper()} k={k} — clustering comparison",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="lower right")
    output_path = os.path.join(plots_dir, f"{model_name}_k{k}_clustering_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Created comparison plot: {os.path.basename(output_path)}")


def compare_models(layout_dirs):
    """Compare clirs vs baseline for each (algorithm, k) found in raw logs."""
    raw_dir = layout_dirs.get("raw", os.path.join(layout_dirs["root"], "raw"))
    seen = set()
    for path in _raw_training_files(raw_dir):
        parsed = _parse_training_filename(path)
        if not parsed:
            continue
        key = (parsed["algo"], parsed["k"])
        if key in seen:
            continue
        seen.add(key)
        compare_clustering_effect(layout_dirs, parsed["algo"], parsed["k"])


def compare_clustering_versions(layout_dirs):
    """Alias: compare clirs vs baseline using the new raw log layout."""
    compare_models(layout_dirs)


def show_menu():
    """Display the main menu and get user choice."""
    print("\nLearning Curves Visualization Menu:")
    print("1. Generate all plots")
    print("2. Generate model comparison plots")
    print("3. Generate clustering versions comparison")
    print("4. Exit")

    while True:
        try:
            choice = int(input("\nEnter your choice (1-4): "))
            if choice in [1, 2, 3, 4]:
                return choice
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from Results/")
    parser.add_argument(
        "--config",
        default=os.path.join(_REPO_ROOT, "Config", "run.json"),
        help="Config file that resolves the experiment cell (default: Config/run.json)",
    )
    args = parser.parse_args()

    config, dirs = load_experiment_layout(args.config)
    print(f"Experiment cell: {dirs['root']}")
    print(f"Plots directory: {dirs['plots']}")
    raw_dir = dirs.get("raw", os.path.join(dirs["root"], "raw"))
    print(f"Raw logs: {raw_dir}")

    if not os.path.isdir(dirs["root"]):
        print(f"Error: experiment directory not found: {dirs['root']}")
        print("Run the training pipeline first:")
        print("  poetry run python CLIRS/Scripts/pipeline.py --Config Config/run.json")
        sys.exit(1)

    while True:
        choice = show_menu()

        if choice == 1:
            plot_learning_curves(dirs)
            compare_models(dirs)
        elif choice == 2:
            compare_models(dirs)
        elif choice == 3:
            compare_clustering_versions(dirs)
        else:
            break

    print("\nLearning curves update completed.")


if __name__ == "__main__":
    main()
