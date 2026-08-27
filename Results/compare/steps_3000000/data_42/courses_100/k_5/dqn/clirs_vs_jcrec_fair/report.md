# CLIRS vs JCRec fair (test split)

## Cell

- Algorithm: `dqn`
- Steps: `3000000`
- Data seed: `42`
- Courses: `100`
- k: `5`
- Compare dir: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair`

## Lineage roots

- CLIRS: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\CLIRS\steps_3000000\data_42\courses_100\k_5`
- Baseline: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\JCRecFair\steps_3000000\data_42\courses_100\k_5`

## Protocol

| Method | Sweep `evaluation_split` | `end` learner population |
|--------|--------------------------|---------------------------|
| CLIRS | `test` | 70/30 hold-out test split |
| JCRecFair | `test` | Same hold-out test split (JCRec algo + CLIRS split) |

Both methods report `end` on the **same test_indices** (paired trials, same `data_seed` / `rl_seed` policy). Primary thesis comparison.


Pairwise CSV records the same populations as `method_a_end_population` / `method_b_end_population` (`test` = hold-out; `all_learners` = full pool).

## Artifacts

- Trial metrics: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair\compare_trial_metrics.csv`
- Pairwise: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair\pairwise_comparison.csv`
- Bootstrap `end_mean`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair\bootstrap_end_mean.csv`
- Bootstrap `end_median`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair\bootstrap_end_median.csv`
- Bootstrap `life_mean`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair\bootstrap_life_mean.csv`
- Bootstrap `life_median`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair\bootstrap_life_median.csv`
- Plot `end`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair\plots\ecdf_end_clirs_dqn_vs_jcrec_fair_dqn.png`
- Plot `life`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_5\dqn\clirs_vs_jcrec_fair\plots\ecdf_life_clirs_dqn_vs_jcrec_fair_dqn.png`

## Pairwise summary

### `end`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.4166666666666667 | 0.1666666666666666 | 0.6666666666666666 | 12 | 5.0 | 0.0 | 7.0 |
| pbpt | 0.4375 | 0.212721121145511 | 0.662278878854489 | 12 | nan | nan | nan |

### `life`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.3333333333333333 | 0.0833333333333333 | 0.5833333333333334 | 12 | 4.0 | 0.0 | 8.0 |
| pbpt | 0.375 | 0.2367454177052219 | 0.513254582294778 | 12 | nan | nan | nan |
