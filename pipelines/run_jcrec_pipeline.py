"""
JCRec author pipeline (greedy / optimal / RL) — unchanged method code from ``jcrec/``.

Run from repo root::

    poetry run python pipelines/run_jcrec_pipeline.py --Config Config/run.json
"""

import argparse
import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JCREC_DIR = _REPO_ROOT / "jcrec"
_CLIRS_SCRIPTS = _REPO_ROOT / "CLIRS" / "Scripts"
for path in (_JCREC_DIR, _CLIRS_SCRIPTS, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Dataset import Dataset
from Greedy import Greedy
from Optimal import Optimal
from Reinforce import Reinforce
from load_config import load_config


def create_and_print_dataset(config):
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JCRec author recommender pipeline.")
    parser.add_argument(
        "--Config",
        default=r"Config/run.json",
        help="Path to run.json or run.yaml",
    )
    args = parser.parse_args()
    config = copy.deepcopy(load_config(args.Config))
    config["results_lineage"] = config.get("jcrec_results_lineage", "JCRec")

    model_classes = {
        "greedy": Greedy,
        "optimal": Optimal,
    }

    print(
        f"JCRec pipeline: model={config.get('model')}, "
        f"results_lineage={config['results_lineage']}"
    )

    for run in range(config["nb_runs"]):
        print(f"\n--- Trial {run + 1}/{config['nb_runs']} ---")
        dataset = create_and_print_dataset(config)

        if config["model"] in model_classes:
            recommender = model_classes[config["model"]](dataset, config["threshold"])
            recommendation_method = getattr(
                recommender, f"{config['model']}_recommendation"
            )
            recommendation_method(config["k"], run)
        else:
            recommender = Reinforce(
                dataset,
                config["model"],
                config["k"],
                config["threshold"],
                run,
                config["total_steps"],
                config["eval_freq"],
            )
            recommender.reinforce_recommendation()

    print(f"\nDone. Results under: {config['results_path']}")


if __name__ == "__main__":
    main()
