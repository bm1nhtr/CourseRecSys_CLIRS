import numpy as np


def matching(level1, level2):

    # get the minimum of the two arrays
    minimum_skill = np.minimum(level1, level2)

    # get the indices of the non zero elements of the job skill levels
    nonzero_indices = np.nonzero(level2)[0]

    # divide the minimum by the job skill levels on the non zero indices
    matching = minimum_skill[nonzero_indices] / level2[nonzero_indices]

    # sum the result and divide by the number of non zero job skill levels
    matching = np.sum(matching) / np.count_nonzero(level2)

    return matching


def learner_job_matching(learner, job):

    # check if one of the arrays is empty
    if not (np.any(job) and np.any(learner)):
        return 0

    return matching(learner, job)


def learner_jobs_matching(learner, jobs):
    """Batch learner–job matching; same semantics as row-wise ``learner_job_matching``.

    ``learner`` shape ``(S,)``, ``jobs`` shape ``(N, S)`` or ``(S,)``.
    Returns scores shape ``(N,)``. Empty learner or empty job rows yield ``0.0``.

    Uses the same per-job nonzero term order and ``np.sum`` accumulation as
    ``matching`` (segment-wise ``np.sum``). Avoid ``bincount`` / ``reduceat`` /
    global ``cumsum`` — they can flip ``>= threshold`` on float edges.
    """
    learner = np.asarray(learner)
    jobs = np.asarray(jobs)
    if jobs.ndim == 1:
        jobs = jobs.reshape(1, -1)

    n_jobs = jobs.shape[0]
    scores = np.zeros(n_jobs, dtype=np.float64)
    if n_jobs == 0 or not np.any(learner):
        return scores

    job_nnz = np.count_nonzero(jobs, axis=1)
    if not np.any(job_nnz):
        return scores

    rows, cols = np.nonzero(jobs)
    vals = np.minimum(learner[cols], jobs[rows, cols]) / jobs[rows, cols]
    uniq, starts = np.unique(rows, return_index=True)
    ends = np.empty(len(starts), dtype=np.intp)
    ends[:-1] = starts[1:]
    ends[-1] = len(vals)
    group_sums = np.fromiter(
        (np.sum(vals[s:e]) for s, e in zip(starts, ends)),
        dtype=np.float64,
        count=len(starts),
    )
    scores[uniq] = group_sums / job_nnz[uniq]
    return scores


def learner_course_required_matching(learner, course):

    required_course = course[0]

    # check if the course has no required skills and return 1
    if not np.any(required_course):
        return 1.0

    return matching(learner, required_course)


def learner_course_provided_matching(learner, course):

    provided_course = course[1]

    return matching(learner, provided_course)


def learner_course_matching(learner, course):

    # get the required and provided matchings
    required_matching = learner_course_required_matching(learner, course)
    provided_matching = learner_course_provided_matching(learner, course)

    return required_matching * (1 - provided_matching)
