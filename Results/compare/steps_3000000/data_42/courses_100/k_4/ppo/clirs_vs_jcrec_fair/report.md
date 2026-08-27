# CLIRS vs JCRec fair (test split)

## Cell

- Algorithm: `ppo`
- Steps: `3000000`
- Data seed: `42`
- Courses: `100`
- k: `4`
- Compare dir: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair`

## Lineage roots

- CLIRS: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\CLIRS\steps_3000000\data_42\courses_100\k_4`
- Baseline: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\JCRecFair\steps_3000000\data_42\courses_100\k_4`

## Protocol

| Method | Sweep `evaluation_split` | `end` learner population |
|--------|--------------------------|---------------------------|
| CLIRS | `test` | 70/30 hold-out test split |
| JCRecFair | `test` | Same hold-out test split (JCRec algo + CLIRS split) |

Both methods report `end` on the **same test_indices** (paired trials, same `data_seed` / `rl_seed` policy). Primary thesis comparison.


Pairwise CSV records the same populations as `method_a_end_population` / `method_b_end_population` (`test` = hold-out; `all_learners` = full pool).

## Artifacts

- Trial metrics: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair\compare_trial_metrics.csv`
- Pairwise: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair\pairwise_comparison.csv`
- Bootstrap `end_mean`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair\bootstrap_end_mean.csv`
- Bootstrap `end_median`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair\bootstrap_end_median.csv`
- Bootstrap `life_mean`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair\bootstrap_life_mean.csv`
- Bootstrap `life_median`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair\bootstrap_life_median.csv`
- Plot `end`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair\plots\ecdf_end_clirs_ppo_vs_jcrec_fair_ppo.png`
- Plot `life`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_4\ppo\clirs_vs_jcrec_fair\plots\ecdf_life_clirs_ppo_vs_jcrec_fair_ppo.png`

## Pairwise summary

### `end`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.6666666666666666 | 0.4166666666666667 | 0.9166666666666666 | 12 | 8.0 | 0.0 | 4.0 |
| pbpt | 0.5625 | 0.3642279535819898 | 0.7607720464180102 | 12 | nan | nan | nan |

### `life`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.2083333333333333 | 0.0 | 0.4583333333333333 | 12 | 2.0 | 1.0 | 9.0 |
| pbpt | 0.1458333333333333 | -0.0485442837647917 | 0.3402109504314584 | 12 | nan | nan | nan |
