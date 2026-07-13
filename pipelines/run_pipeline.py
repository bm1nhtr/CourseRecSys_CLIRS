"""
Orchestrate CLIRS, JCRec fair-split, and JCRec author pipelines.

Run from repo root::

    poetry run python pipelines/run_pipeline.py --Config Config/run.json

Config (``run.json``)::

    "orchestration": { "pipelines": ["clirs", "jcrec_fair", "jcrec"] }
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLIRS_SCRIPTS = _REPO_ROOT / "CLIRS" / "Scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from JCRecFair.split_sync import ClirsSplitNotFoundError, require_clirs_split_file
from Utils.experiment_log import append_orchestration_note
from Utils.results_paths import experiment_log_path
from pipelines.cross_lineage_eval import run_cross_lineage_compares

_PIPELINE_SCRIPTS = {
    "clirs": _REPO_ROOT / "pipelines" / "run_clirs_pipeline.py",
    "jcrec_fair": _REPO_ROOT / "pipelines" / "run_jcrec_fair_pipeline.py",
    "jcrec": _REPO_ROOT / "pipelines" / "run_jcrec_pipeline.py",
}


def _load_flat_config(config_path: str) -> dict:
    path = _CLIRS_SCRIPTS / "load_config.py"
    spec = importlib.util.spec_from_file_location("clirs_load_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load load_config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config(config_path)


def _pipeline_run_log_path(config_path: str, pipeline_name: str) -> str:
    config = _load_flat_config(config_path)
    if pipeline_name == "clirs":
        config["pipeline"] = "clirs"
    elif pipeline_name == "jcrec_fair":
        config["pipeline"] = "jcrec_fair"
        config["results_lineage"] = config.get("jcrec_fair_results_lineage", "JCRecFair")
        config["use_clustering"] = False
    elif pipeline_name == "jcrec":
        config["pipeline"] = "jcrec"
        config["results_lineage"] = config.get("jcrec_results_lineage", "JCRec")
        config["use_clustering"] = False
    return experiment_log_path(config)


def validate_orchestration_order(names: list[str]) -> None:
    """jcrec_fair must run after clirs when both are in the same orchestration."""
    if "jcrec_fair" not in names or "clirs" not in names:
        return
    if names.index("clirs") >= names.index("jcrec_fair"):
        raise SystemExit(
            "Invalid orchestration order: jcrec_fair must run after clirs "
            "(CLIRS publishes split_indices.json for JCRecFair).\n"
            f"  Got: {names}\n"
            '  Expected clirs before jcrec_fair, e.g. ["clirs", "jcrec_fair", "jcrec"]'
        )


def load_orchestration_pipelines(config_path: str) -> list[str]:
    """Read ``orchestration.pipelines`` from run.json — only used by this script."""
    path = Path(config_path).resolve()
    with open(path, encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            raw = json.load(f)
        else:
            raw = yaml.load(f, Loader=yaml.FullLoader) or {}
    pipelines = raw.get("orchestration", {}).get("pipelines")
    return pipelines if pipelines else ["clirs"]


def run_pipelines(config_path: str, names: list[str]) -> list[str]:
    cell_logs: list[str] = []
    for index, name in enumerate(names, start=1):
        script = _PIPELINE_SCRIPTS.get(name)
        if script is None:
            known = ", ".join(sorted(_PIPELINE_SCRIPTS))
            raise SystemExit(f"Unknown pipeline {name!r}. Known: {known}")
        print(f"\n========== Pipeline {index}/{len(names)}: {name} ==========")
        try:
            subprocess.run(
                [sys.executable, str(script), "--Config", config_path],
                check=True,
                cwd=str(_REPO_ROOT),
            )
        except subprocess.CalledProcessError as exc:
            cell_logs.append(_pipeline_run_log_path(config_path, name))
            print(
                f"[ERROR] Pipeline {name!r} failed with exit code {exc.returncode}"
            )
            raise
        cell_logs.append(_pipeline_run_log_path(config_path, name))

        if name == "clirs" and "jcrec_fair" in names[names.index("clirs") + 1 :]:
            config = _load_flat_config(config_path)
            try:
                split_path = require_clirs_split_file(config)
                print(f"CLIRS split verified for jcrec_fair: {split_path}")
            except ClirsSplitNotFoundError as exc:
                raise SystemExit(
                    f"CLIRS finished but split_indices.json is missing — "
                    f"jcrec_fair cannot run.\n{exc}"
                ) from exc
    return cell_logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CLIRS / JCRec fair / JCRec author pipelines from Config/run.json."
    )
    parser.add_argument("--Config", default=r"Config/run.json")
    args = parser.parse_args()

    try:
        names = load_orchestration_pipelines(args.Config)
    except Exception as exc:
        print(f"[ERROR] Failed to read orchestration from {args.Config}: {exc}")
        sys.exit(1)

    try:
        validate_orchestration_order(names)
    except SystemExit as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"Orchestration: {names}")
    cell_logs: list[str] = []
    try:
        cell_logs = run_pipelines(args.Config, names)
    except subprocess.CalledProcessError:
        append_orchestration_note(
            str(_REPO_ROOT / "Results"),
            config_path=args.Config,
            pipelines=names,
            cell_logs=cell_logs,
            status="FAILED",
        )
        raise
    orch_log = append_orchestration_note(
        str(_REPO_ROOT / "Results"),
        config_path=args.Config,
        pipelines=names,
        cell_logs=cell_logs,
    )

    if "clirs" in names:
        try:
            config = _load_flat_config(args.Config)
            run_cross_lineage_compares(config)
        except Exception as exc:
            print(f"[WARN] Cross-lineage compare failed: {exc}")

    print("\nOrchestration complete.")
    if orch_log is not None:
        print(f"Issues logged — see: {orch_log}")


if __name__ == "__main__":
    main()

