"""Per-experiment ``run.log`` — written only when warnings or failures occur."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

_RL_ALGORITHMS = frozenset({"dqn", "ppo"})


def experiment_log_path(experiment_root: str) -> str:
    """Canonical per-experiment log: ``{experiment_root}/run.log``."""
    return os.path.join(experiment_root, "run.log")


def _git_revision(repo_root: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "n/a"


class ExperimentRunLog:
    """Collect issues in memory; flush a compact ``run.log`` only if needed."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_root: str,
        *,
        config_path: str | None = None,
        pipeline: str | None = None,
        repo_root: str | None = None,
    ) -> None:
        self.config = config
        self.experiment_root = experiment_root
        self.config_path = config_path
        self.pipeline = pipeline
        self.repo_root = repo_root or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.path = experiment_log_path(experiment_root)
        self._warnings: list[str] = []

    def __enter__(self) -> ExperimentRunLog:
        os.makedirs(self.experiment_root, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        failed = exc_type is not None
        if failed or self._warnings:
            self._write_report(failed=failed, exc_val=exc_val)
            print(
                f"\n*** Issues recorded — send run.log to the maintainer:\n    {self.path}"
            )
            for message in self._warnings:
                print(f"  [WARN] {message}")
            if failed and exc_val is not None:
                print(f"  [ERROR] {type(exc_val).__name__}: {exc_val}")
        elif os.path.isfile(self.path):
            os.remove(self.path)
        return False

    def warn(self, message: str) -> None:
        self._warnings.append(message)
        print(f"[WARN] {message}")

    def record_exception(
        self,
        exc: BaseException,
        *,
        trial_id: int | None = None,
        phase: str = "unknown",
    ) -> None:
        """Log a caught exception with trial/phase context before re-raising."""
        where = f"Trial {trial_id}" if trial_id is not None else "Setup"
        self.warn(f"{where} [{phase}]: {type(exc).__name__}: {exc}")

    def record_trial(
        self,
        *,
        trial_id: int,
        algorithm: str,
        life: float | None,
        end: float | None,
        training_path: str | None = None,
    ) -> None:
        """Record trial metrics and emit warnings for common failure modes."""
        expects_life = algorithm in _RL_ALGORITHMS
        if expects_life:
            if not training_path or not os.path.isfile(training_path):
                self.warn(
                    f"Trial {trial_id}: missing training log at {training_path!r} "
                    f"(metric life will be null)"
                )
            elif life is None:
                self.warn(
                    f"Trial {trial_id}: life is null — training log exists but could "
                    f"not be parsed ({training_path})"
                )
        if end is None:
            self.warn(f"Trial {trial_id}: end metric is null")

    def check_trial_artifacts(
        self,
        *,
        trial_id: int,
        eval_path: str | None = None,
        save_raw: bool = True,
    ) -> None:
        """Warn when expected per-trial eval JSON was not written."""
        if save_raw and (not eval_path or not os.path.isfile(eval_path)):
            self.warn(f"Trial {trial_id}: missing eval JSON at {eval_path!r}")

    def _write_report(
        self, *, failed: bool, exc_val: BaseException | None
    ) -> None:
        started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        cfg = self.config
        lines = [
            f"RUN ISSUE REPORT {started}",
            f"pipeline: {self.pipeline or cfg.get('pipeline', 'unknown')}",
            f"config: {self.config_path or 'n/a'}",
            f"experiment: {self.experiment_root}",
            (
                f"model: {cfg.get('model')} | steps: {cfg.get('total_steps')} | "
                f"data_seed: {cfg.get('seed')} | trials: {cfg.get('nb_runs')}"
            ),
            f"python: {platform.python_version()} | platform: {sys.platform}",
            f"git: {_git_revision(self.repo_root)}",
            f"status: {'FAILED' if failed else 'OK_WITH_WARNINGS'}",
        ]
        if failed and exc_val is not None:
            lines.append(f"error: {type(exc_val).__name__}: {exc_val}")
        if self._warnings:
            lines.append("warnings:")
            lines.extend(f"  - {message}" for message in self._warnings)
        lines.append("")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))


def append_orchestration_note(
    results_root: str,
    *,
    config_path: str,
    pipelines: list[str],
    cell_logs: list[str],
    status: str = "OK",
) -> str | None:
    """Append to ``Results/orchestration.log`` only on failure or non-empty cell logs."""
    issue_logs = [
        path
        for path in cell_logs
        if os.path.isfile(path) and os.path.getsize(path) > 0
    ]
    if status == "OK" and not issue_logs:
        return None

    os.makedirs(results_root, exist_ok=True)
    path = os.path.join(results_root, "orchestration.log")
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"ORCHESTRATION {started} status={status}\n")
        f.write(f"config: {config_path}\n")
        f.write(f"pipelines: {', '.join(pipelines)}\n")
        if issue_logs:
            f.write("issue run logs:\n")
            for log_path in issue_logs:
                f.write(f"  - {log_path}\n")
        f.write("\n")
    return path
