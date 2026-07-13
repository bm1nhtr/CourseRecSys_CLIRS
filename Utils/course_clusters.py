"""Fit-once / freeze course cluster labels for a CLIRS experiment cell."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from Utils.complete_algorithm import ManifestValidationError

COURSE_CLUSTERS_FILENAME = "course_clusters.json"


def course_clusters_path(experiment_root: str) -> str:
    return os.path.join(experiment_root, COURSE_CLUSTERS_FILENAME)


def _clusterer_runtime_kwargs(config: Mapping[str, Any], dirs: Mapping[str, str]):
    return {
        "random_state": config.get("seed", 42),
        "auto_clusters": config.get("auto_clusters", False),
        "max_clusters": config.get("max_clusters", 10),
        "config": config.get("clustering", {}),
        "clustering_dir": config.get("clustering_plots_dir", dirs.get("clustering_plots")),
        "reports_dir": dirs.get("reports"),
        "selection_method": config.get("cluster_selection", "silhouette"),
        "min_cluster_size": config.get("min_cluster_size", 5),
        "max_level": config.get("max_level", 3),
    }


def validate_course_clusters_artifact(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    dataset: Any,
) -> list[str]:
    """Check frozen labels match current config and dataset."""
    from clustering import FEATURE_SPEC

    errors: list[str] = []
    if payload.get("feature_spec") != FEATURE_SPEC:
        errors.append(
            f"course_clusters feature_spec: saved={payload.get('feature_spec')!r} "
            f"!= current {FEATURE_SPEC!r}"
        )
    if payload.get("data_seed") != config.get("seed"):
        errors.append(
            f"course_clusters data_seed: saved={payload.get('data_seed')!r} "
            f"!= config={config.get('seed')!r}"
        )
    n_courses = len(dataset.courses)
    if payload.get("nb_courses") != n_courses:
        errors.append(
            f"course_clusters nb_courses: saved={payload.get('nb_courses')!r} "
            f"!= current={n_courses!r}"
        )
    labels = payload.get("labels_by_index", [])
    if len(labels) != n_courses:
        errors.append(
            f"course_clusters labels_by_index: len={len(labels)} != n_courses={n_courses}"
        )

    checks = (
        ("auto_clusters", "auto_clusters"),
        ("max_clusters", None),
        ("cluster_selection", "selection_method"),
        ("min_cluster_size", "min_cluster_size"),
    )
    for config_key, artifact_key in checks:
        artifact_key = artifact_key or config_key
        expected = payload.get(artifact_key)
        actual = config.get(config_key)
        if expected is not None and actual is not None and expected != actual:
            errors.append(
                f"course_clusters {artifact_key}: saved={expected!r} config={actual!r}"
            )

    courses_index = getattr(dataset, "courses_index", None)
    if courses_index and payload.get("labels_by_course_id"):
        for idx in range(n_courses):
            course_id = courses_index.get(idx)
            if course_id is None:
                continue
            saved = payload["labels_by_course_id"].get(str(course_id))
            if saved is None:
                saved = payload["labels_by_course_id"].get(course_id)
            if saved is not None and int(saved) != int(labels[idx]):
                errors.append(
                    f"course_clusters labels_by_course_id mismatch at index {idx}"
                )
                break

    return errors


def clustering_manifest_summary(payload: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compact clustering block for manifest.json."""
    summary = {
        "artifact": COURSE_CLUSTERS_FILENAME,
        "feature_spec": payload.get("feature_spec"),
        "selection_method": payload.get("selection_method"),
        "auto_clusters": payload.get("auto_clusters"),
        "n_clusters_fitted": payload.get("n_clusters_fitted"),
        "min_cluster_size": payload.get("min_cluster_size"),
        "inertia": payload.get("inertia"),
    }
    if config is not None:
        summary["max_clusters"] = config.get("max_clusters")
    elif payload.get("max_clusters") is not None:
        summary["max_clusters"] = payload.get("max_clusters")
    return summary


def ensure_course_clusterer(config, dataset, dirs):
    """
    Fit course clusters once per experiment cell, or load frozen labels.

    Returns a fitted ``CourseClusterer`` when ``use_clustering`` is true, else None.
    """
    if not config.get("use_clustering"):
        return None

    from clustering import CourseClusterer

    path = course_clusters_path(dirs["root"])
    runtime_kwargs = _clusterer_runtime_kwargs(config, dirs)

    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        errors = validate_course_clusters_artifact(payload, config, dataset)
        if errors:
            raise ManifestValidationError(errors)
        print(f"Loaded frozen course clusters from {path}")
        config["_clustering_manifest"] = clustering_manifest_summary(payload, config)
        return CourseClusterer.from_artifact(path, **runtime_kwargs)

    print("Fitting course clusters once for this experiment cell...")
    clusterer = CourseClusterer(**runtime_kwargs)
    clusterer.fit_course_clusters(
        dataset.courses,
        courses_index=getattr(dataset, "courses_index", None),
    )
    payload = clusterer.to_artifact_payload(
        data_seed=config.get("seed"),
        nb_courses=len(dataset.courses),
        courses_index=getattr(dataset, "courses_index", None),
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote frozen course clusters to {path}")
    config["_clustering_manifest"] = clustering_manifest_summary(payload, config)
    return clusterer
