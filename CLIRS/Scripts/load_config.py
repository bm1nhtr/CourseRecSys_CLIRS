"""Load experiment configuration from run.json (primary) or flat YAML."""

import json
import os

import yaml

_PATH_KEYS = (
    "taxonomy_path",
    "mastery_levels_path",
    "cv_path",
    "job_path",
    "course_path",
    "results_path",
)


def _project_root(config_path):
    """Repo root: parent of CLIRS/ when config lives under CLIRS/config/."""
    config_dir = os.path.dirname(os.path.abspath(config_path))
    class_dir = os.path.dirname(config_dir)
    if os.path.basename(class_dir) == "CLIRS":
        return os.path.dirname(class_dir)
    return os.getcwd()


def _resolve_paths(config, config_path):
    root = _project_root(config_path)
    for key in _PATH_KEYS:
        value = config.get(key)
        if value and not os.path.isabs(value):
            config[key] = os.path.normpath(os.path.join(root, value))
    return config


def flatten_run_json(raw):
    """Map nested run.json sections to the flat dict used by Dataset and Reinforce."""
    experiment = raw.get("experiment", {})
    seeds = raw.get("seeds", {})
    data = raw.get("data", {})
    split = raw.get("split", {})
    model = raw.get("model", {})
    environment = raw.get("environment", {})
    clustering = raw.get("clustering", {})
    results = raw.get("results", {})

    reward_multipliers = clustering.get("reward_multipliers", {})
    clustering_cfg = {
        key: value
        for key, value in reward_multipliers.items()
        if not key.startswith("_") and value is not None
    }

    config = {}
    config.update(experiment)
    config["seed"] = seeds.get("data", 42)
    config.update(data)
    config.update(split)
    config["model"] = model.get("algorithm", model.get("model", "dqn"))
    config["total_steps"] = model.get("total_steps", 500000)
    config["eval_freq"] = model.get("eval_freq", 1000)
    config.update(environment)
    config["use_clustering"] = clustering.get("use_clustering", False)
    config["n_clusters"] = clustering.get("n_clusters", 5)
    config["auto_clusters"] = clustering.get("auto_clusters", False)
    config["max_clusters"] = clustering.get("max_clusters", 10)
    config["clustering"] = clustering_cfg
    config.update(results)
    if "results_path" not in config and config.get("results_dir"):
        config["results_path"] = config["results_dir"]
    return config


def load_config(config_path):
    """Load config from JSON (nested) or YAML (flat). Returns a flat runtime dict."""
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.lower().endswith(".json"):
            raw = json.load(f)
            config = flatten_run_json(raw)
        else:
            config = yaml.load(f, Loader=yaml.FullLoader) or {}
    if "results_path" not in config and config.get("results_dir"):
        config["results_path"] = config["results_dir"]
    return _resolve_paths(config, config_path)
