# CLIRS vs JCRec fair (test split)

## Cell

- Algorithm: `ppo`
- Steps: `3000000`
- Data seed: `42`
- Courses: `100`
- k: `2`
- Compare dir: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair`

## Lineage roots

- CLIRS: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/CLIRS/steps_3000000/data_42/courses_100/k_2`
- Baseline: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/JCRecFair/steps_3000000/data_42/courses_100/k_2`

## Protocol

| Method | Sweep `evaluation_split` | `end` learner population |
|--------|--------------------------|---------------------------|
| CLIRS | `test` | 70/30 hold-out test split |
| JCRecFair | `test` | Same hold-out test split (JCRec algo + CLIRS split) |

Both methods report `end` on the **same test_indices** (paired trials, same `data_seed` / `rl_seed` policy). Primary thesis comparison.


Pairwise CSV records the same populations as `method_a_end_population` / `method_b_end_population` (`test` = hold-out; `all_learners` = full pool).

## Artifacts

- Trial metrics: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair/compare_trial_metrics.csv`
- Pairwise: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair/pairwise_comparison.csv`
- Bootstrap `end_mean`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair/bootstrap_end_mean.csv`
- Bootstrap `end_median`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair/bootstrap_end_median.csv`
- Bootstrap `life_mean`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair/bootstrap_life_mean.csv`
- Bootstrap `life_median`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair/bootstrap_life_median.csv`
- Plot `end`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair/plots/ecdf_end_clirs_ppo_vs_jcrec_fair_ppo.png`
- Plot `life`: `/home/alesage/arno_data/CourseRecSys_CLIRS/Results/compare/steps_3000000/data_42/courses_100/k_2/ppo/clirs_vs_jcrec_fair/plots/ecdf_life_clirs_ppo_vs_jcrec_fair_ppo.png`

## Pairwise summary

### `end`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.5454545454545454 | 0.2727272727272727 | 0.8181818181818182 | 11 | 5.0 | 2.0 | 4.0 |
| pbpt | 0.5206611570247934 | 0.3372555041498092 | 0.7040668098997777 | 11 | nan | nan | nan |

### `life`

| statistic | value | ci_low | ci_high | n_pairs | wins | ties | losses |
|-----------|-------|--------|---------|---------|------|------|--------|
| win_rate | 0.3181818181818182 | 0.0909090909090909 | 0.5909090909090909 | 11 | 3.0 | 1.0 | 7.0 |
| pbpt | 0.2066115702479339 | 0.1392039057300398 | 0.2740192347658279 | 11 | nan | nan | nan |
