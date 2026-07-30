# CLIRS vs JCRec fair (test split)

## Cell

- Algorithm: `dqn`
- Steps: `3000000`
- Data seed: `42`
- Courses: `100`
- k: `3`
- Compare dir: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair`

## Lineage roots

- CLIRS: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/CLIRS/steps_3000000/data_42/courses_100/k_3`
- Baseline: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/JCRecFair/steps_3000000/data_42/courses_100/k_3`

## Protocol

| Method | Sweep `evaluation_split` | `end` learner population |
|--------|--------------------------|---------------------------|
| CLIRS | `test` | 70/30 hold-out test split |
| JCRecFair | `test` | Same hold-out test split (JCRec algo + CLIRS split) |

Both methods report `end` on the **same test_indices** (paired trials, same `data_seed` / `rl_seed` policy). Primary thesis comparison.


Pairwise CSV records the same populations as `method_a_end_population` / `method_b_end_population` (`test` = hold-out; `all_learners` = full pool).

## Artifacts

- Trial metrics: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair/compare_trial_metrics.csv`
- Pairwise: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair/pairwise_comparison.csv`
- Bootstrap `end_mean`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair/bootstrap_end_mean.csv`
- Bootstrap `end_median`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair/bootstrap_end_median.csv`
- Bootstrap `life_mean`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair/bootstrap_life_mean.csv`
- Bootstrap `life_median`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair/bootstrap_life_median.csv`
- Plot `end`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair/plots/ecdf_end_clirs_dqn_vs_jcrec_fair_dqn.png`
- Plot `life`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_3/dqn/clirs_vs_jcrec_fair/plots/ecdf_life_clirs_dqn_vs_jcrec_fair_dqn.png`

## Pairwise summary

### `end`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.5416666666666666 | 0.2916666666666667 | 0.7916666666666666 | 12 | 6.0 | 1.0 | 5.0 |
| pbpt | 0.6597222222222222 | 0.4510132545809469 | 0.8684311898634975 | 12 | nan | nan | nan |

### `life`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.4583333333333333 | 0.2083333333333333 | 0.75 | 12 | 5.0 | 1.0 | 6.0 |
| pbpt | 0.4444444444444444 | 0.2578149826117105 | 0.6310739062771783 | 12 | nan | nan | nan |
