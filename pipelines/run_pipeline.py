"""
Orchestrate CLIRS and JCRec pipelines.

Run from repo root::

    poetry run python pipelines/run_pipeline.py --Config Config/run.json

Config (``run.json``)::

    "orchestration": { "pipelines": ["clirs", "jcrec"] }
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLIRS_SCRIPTS = _REPO_ROOT / "CLIRS" / "Scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PIPELINE_SCRIPTS = {
    "clirs": _REPO_ROOT / "pipelines" / "run_clirs_pipeline.py",
    "jcrec": _REPO_ROOT / "pipelines" / "run_jcrec_pipeline.py",
}


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


def run_pipelines(config_path: str, names: list[str]) -> None:
    for index, name in enumerate(names, start=1):
        script = _PIPELINE_SCRIPTS.get(name)
        if script is None:
            known = ", ".join(sorted(_PIPELINE_SCRIPTS))
            raise SystemExit(f"Unknown pipeline {name!r}. Known: {known}")
        print(f"\n========== Pipeline {index}/{len(names)}: {name} ==========")
        subprocess.run(
            [sys.executable, str(script), "--Config", config_path],
            check=True,
            cwd=str(_REPO_ROOT),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CLIRS and/or JCRec pipelines from Config/run.json."
    )
    parser.add_argument("--Config", default=r"Config/run.json")
    args = parser.parse_args()

    names = load_orchestration_pipelines(args.Config)
    print(f"Orchestration: {names}")
    run_pipelines(args.Config, names)
    print("\nOrchestration complete.")


if __name__ == "__main__":
    main()
