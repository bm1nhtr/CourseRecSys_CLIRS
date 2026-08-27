# CLIRS vs JCRec author reproduction

## Cell

- Algorithm: `ppo`
- Steps: `3000000`
- Data seed: `42`
- Courses: `100`
- k: `3`
- Compare dir: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author`

## Lineage roots

- CLIRS: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\CLIRS\steps_3000000\data_42\courses_100\k_3`
- Baseline: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\JCRec\steps_3000000\data_42\courses_100\k_3`

## Protocol

| Method | Sweep `evaluation_split` | `end` learner population |
|--------|--------------------------|---------------------------|
| CLIRS | `test` | 70/30 hold-out test split |
| JCRec | `all_learners` | Full learner pool (author protocol) |

Different `end` populations — author replication context only.


Pairwise CSV records the same populations as `method_a_end_population` / `method_b_end_population` (`test` = hold-out; `all_learners` = full pool).

## Artifacts

- Trial metrics: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author\compare_trial_metrics.csv`
- Pairwise: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author\pairwise_comparison.csv`
- Bootstrap `end_mean`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author\bootstrap_end_mean.csv`
- Bootstrap `end_median`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author\bootstrap_end_median.csv`
- Bootstrap `life_mean`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author\bootstrap_life_mean.csv`
- Bootstrap `life_median`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author\bootstrap_life_median.csv`
- Plot `end`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author\plots\ecdf_end_clirs_ppo_vs_jcrec_ppo.png`
- Plot `life`: `C:\Users\Prof\Desktop\TRAN\CourseRecSys_CLIRS\Results\compare\steps_3000000\data_42\courses_100\k_3\ppo\clirs_vs_jcrec_author\plots\ecdf_life_clirs_ppo_vs_jcrec_ppo.png`

## Pairwise summary

### `end`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.0 | 0.0 | 0.0 | 12 | 0.0 | 0.0 | 12.0 |
| pbpt | 0.0069444444444444 | -0.0083401747228586 | 0.0222290636117474 | 12 | nan | nan | nan |

### `life`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 1.0 | 1.0 | 1.0 | 12 | 12.0 | 0.0 | 0.0 |
| pbpt | 1.0 | 1.0 | 1.0 | 12 | nan | nan | nan |
