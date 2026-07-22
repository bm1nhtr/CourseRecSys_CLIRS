"""Parity tests: vectorized learner–job matching vs scalar reference loop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_CLIRS_SCRIPTS = _REPO / "CLIRS" / "Scripts"
_JCREC = _REPO / "jcrec"


def _load_matchings(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    try:
        if "matchings" in sys.modules:
            del sys.modules["matchings"]
        import matchings as m  # noqa: WPS433
        return m
    finally:
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))


class MatchingVectorizedParityTests(unittest.TestCase):
    def _assert_module_parity(self, m):
        rng = np.random.default_rng(0)
        n_skills, n_jobs = 46, 80

        for _ in range(40):
            learner = rng.integers(0, 4, size=n_skills, dtype=np.int32)
            # Sparse-ish learner
            mask = rng.random(n_skills) > 0.7
            learner = learner * mask.astype(np.int32)

            jobs = rng.integers(0, 4, size=(n_jobs, n_skills), dtype=np.int32)
            jobs *= (rng.random((n_jobs, n_skills)) > 0.75).astype(np.int32)

            # Edge rows: empty job, empty-ish
            jobs[0] = 0
            jobs[1, 0] = 2

            batch = m.learner_jobs_matching(learner, jobs)
            for i in range(n_jobs):
                ref = float(m.learner_job_matching(learner, jobs[i]))
                self.assertEqual(
                    batch[i],
                    ref,
                    msg=f"job {i}: batch={batch[i]!r} ref={ref!r}",
                )

        # Empty learner → all zeros
        empty = np.zeros(n_skills, dtype=np.int32)
        jobs = rng.integers(0, 4, size=(10, n_skills), dtype=np.int32)
        self.assertTrue(np.all(m.learner_jobs_matching(empty, jobs) == 0.0))

        # Single-job reshape path
        job = jobs[3]
        self.assertEqual(m.learner_jobs_matching(learner, job).shape, (1,))
        self.assertEqual(
            m.learner_jobs_matching(learner, job)[0],
            float(m.learner_job_matching(learner, job)),
        )

    def test_clirs_matchings_parity(self):
        self._assert_module_parity(_load_matchings(_CLIRS_SCRIPTS))

    def test_jcrec_matchings_parity(self):
        self._assert_module_parity(_load_matchings(_JCREC))

    def test_applicable_count_matches_loop(self):
        """Dataset candidate set + threshold count equals scalar loop."""
        m = _load_matchings(_CLIRS_SCRIPTS)
        rng = np.random.default_rng(1)
        n_skills, n_jobs = 30, 50
        learner = rng.integers(0, 4, size=n_skills, dtype=np.int32)
        learner[rng.random(n_skills) > 0.6] = 0
        jobs = rng.integers(0, 4, size=(n_jobs, n_skills), dtype=np.int32)
        jobs *= (rng.random((n_jobs, n_skills)) > 0.7).astype(np.int32)

        # Build inverted index like Dataset
        inverted = {s: set() for s in range(n_skills)}
        for j in range(n_jobs):
            for s in range(n_skills):
                if jobs[j, s] > 0:
                    inverted[s].add(j)

        subset = set()
        for s in np.nonzero(learner)[0]:
            subset.update(inverted[s])

        for threshold in (0.0, 0.5, 0.8, 1.0):
            expected = 0
            for jid in subset:
                if m.learner_job_matching(learner, jobs[jid]) >= threshold:
                    expected += 1
            if not subset:
                got = 0
            else:
                idx = np.fromiter(subset, dtype=np.intp, count=len(subset))
                scores = m.learner_jobs_matching(learner, jobs[idx])
                got = int(np.count_nonzero(scores >= threshold))
            self.assertEqual(got, expected, msg=f"threshold={threshold}")


if __name__ == "__main__":
    unittest.main()
