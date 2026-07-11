"""
Results management for CLIRS experiment outputs.

Uses Utils.results_paths layout:
  Results/{lineage}/steps_{steps}/data_{data_seed}/courses_{nb_courses}/
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Utils.results_paths import courses_dir_slug, experiment_root, repo_root


def _results_base() -> str:
    return os.path.join(str(repo_root()), "Results")


def _cell_stats(cell: str) -> tuple[int, int]:
    sweeps = os.path.join(cell, "sweeps")
    raw = os.path.join(cell, "raw")
    n_csv = len(os.listdir(sweeps)) if os.path.isdir(sweeps) else 0
    n_raw = len(os.listdir(raw)) if os.path.isdir(raw) else 0
    return n_csv, n_raw


def list_experiments():
    """List experiment cells under Results/."""
    base = _results_base()
    if not os.path.isdir(base):
        print("No Results/ directory found.")
        return

    found = False
    for lineage in sorted(os.listdir(base)):
        lineage_path = os.path.join(base, lineage)
        if not os.path.isdir(lineage_path):
            continue
        for steps_dir in sorted(os.listdir(lineage_path)):
            steps_path = os.path.join(lineage_path, steps_dir)
            if not os.path.isdir(steps_path):
                continue
            for data_dir in sorted(os.listdir(steps_path)):
                data_path = os.path.join(steps_path, data_dir)
                if not os.path.isdir(data_path):
                    continue

                course_dirs = [
                    name
                    for name in sorted(os.listdir(data_path))
                    if os.path.isdir(os.path.join(data_path, name))
                    and name.startswith("courses_")
                ]

                if course_dirs:
                    for courses_dir in course_dirs:
                        cell = os.path.join(data_path, courses_dir)
                        found = True
                        n_csv, n_raw = _cell_stats(cell)
                        print(f"- {lineage}/{steps_dir}/{data_dir}/{courses_dir}")
                        print(f"    sweeps: {n_csv} csv | raw: {n_raw} files")
                elif os.path.isfile(os.path.join(data_path, "manifest.json")):
                    # Legacy layout without courses_* subfolder
                    found = True
                    n_csv, n_raw = _cell_stats(data_path)
                    print(f"- {lineage}/{steps_dir}/{data_dir} (legacy)")
                    print(f"    sweeps: {n_csv} csv | raw: {n_raw} files")

    if not found:
        print("No experiment cells found under Results/.")


def backup_experiment(config_path: str):
    """Backup one experiment cell resolved from config."""
    sys.path.insert(0, str(_REPO_ROOT / "CLIRS" / "Scripts"))
    from load_config import load_config

    config = load_config(config_path)
    source = experiment_root(config)
    if not os.path.isdir(source):
        print(f"Experiment directory not found: {source}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lineage = config.get("results_lineage", "CLIRS")
    data_seed = config.get("seed", 42)
    backup_name = f"{lineage}_data{data_seed}_{courses_dir_slug(config)}_{timestamp}"
    dest = os.path.join(str(repo_root()), "backups", backup_name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(source, dest)
    print(f"Backed up to {dest}")


def main():
    parser = argparse.ArgumentParser(description="Manage CLIRS Results/")
    parser.add_argument(
        "command",
        choices=["list", "backup"],
        help="list experiment cells or backup one config's experiment",
    )
    parser.add_argument(
        "--config",
        default=str(repo_root() / "Config" / "run.json"),
        help="Config path for backup (default: Config/run.json)",
    )
    args = parser.parse_args()

    if args.command == "list":
        list_experiments()
    else:
        backup_experiment(args.config)


if __name__ == "__main__":
    main()
