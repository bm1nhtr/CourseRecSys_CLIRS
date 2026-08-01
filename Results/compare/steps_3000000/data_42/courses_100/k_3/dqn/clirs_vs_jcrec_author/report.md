# CLIRS vs JCRec author reproduction

## Cell

- Algorithm: `dqn`
- Steps: `3000000`
- Data seed: `42`
- Courses: `100`
- k: `3`
- Compare dir: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author`

## Lineage roots

- CLIRS: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/CLIRS/steps_3000000/data_42/courses_100/k_3`
- Baseline: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/JCRec/steps_3000000/data_42/courses_100/k_3`

## Protocol

| Method | Sweep `evaluation_split` | `end` learner population |
|--------|--------------------------|---------------------------|
| CLIRS | `test` | 70/30 hold-out test split |
| JCRec | `all_learners` | Full learner pool (author protocol) |

Different `end` populations — author replication context only.


Pairwise CSV records the same populations as `method_a_end_population` / `method_b_end_population` (`test` = hold-out; `all_learners` = full pool).

## Artifacts

- Trial metrics: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author/compare_trial_metrics.csv`
- Pairwise: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author/pairwise_comparison.csv`
- Bootstrap `end_mean`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author/bootstrap_end_mean.csv`
- Bootstrap `end_median`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author/bootstrap_end_median.csv`
- Bootstrap `life_mean`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author/bootstrap_life_mean.csv`
- Bootstrap `life_median`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author/bootstrap_life_median.csv`
- Plot `end`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author/plots/ecdf_end_clirs_dqn_vs_jcrec_dqn.png`
- Plot `life`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_author/plots/ecdf_life_clirs_dqn_vs_jcrec_dqn.png`

## Pairwise summary

### `end`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.0909090909090909 | 0.0 | 0.2727272727272727 | 11 | 1.0 | 0.0 | 10.0 |
| pbpt | 0.0330578512396694 | 0.0022447155205522 | 0.0638709869587866 | 11 | nan | nan | nan |

### `life`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 1.0 | 1.0 | 1.0 | 11 | 11.0 | 0.0 | 0.0 |
| pbpt | 1.0 | 1.0 | 1.0 | 11 | nan | nan | nan |
