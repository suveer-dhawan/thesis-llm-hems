# Phase 3 run -- LIVE

- households (24): [8201999, 8208263, 8211127, 8395524, 8456189, 8463253, 8481835, 8568499, 8596070, 8619039, 8632083, 8638181, 9012382, 9911140, 9911204, 9911250, 9911254, 10042536, 10239118, 10278838, 10444433, 10556929, 10649333, 11105177]
- steps: ['elasticity', 'overlay', 'seeds']
- band mode: `fixed` (fixed +/-2.0 C for every household); comfort orientation = persona's
- WTP scaling: global_scale 1.0 x archetype factor {'A': 1.3, 'B': 1.0, 'C': 0.7}, identical for ['rule_based', 'llm']; calibration enabled: False
- seeds: primary 42; LLM [42] (T=0); noisy-MILP [42, 43, 44]; MILP + rule-based deterministic
- thermal heterogeneity: {'enabled': True, 'r_range': [0.8, 1.2], 'c_range': [0.8, 1.2], 't0_range': [-1.0, 1.0], 'seed': 42}
- rule-based k (fixed band fit): {'comfort_prioritising': 3.791, 'balanced': 0.62, 'comfort_flexible': 0.66}, w_max 0.888
- provider/model: gemini / gemini-3.5-flash-lite
- provenance: `provenance.json`; full config: `config_resolved.yaml`

## Summary by agent (population aggregate)

Peak / PAR on the summed load of all households; post = first post-CPP half-hour vs the CPP-window mean; degree-hours = sum over households of max(0, indoor - ceiling) x 0.5 h; elasticities pooled (`base`; midpoint alongside), total = whole household; Conway [-0.47, -0.026].

### Baseline day

| agent | seed | base_peak_kw | base_time_of_peak | base_par | base_cpp_window_mean_kw | base_post_20:00-20:30_kw | base_post_over_cpp_ratio | base_degree_hours_above_ceiling | base_max_exceedance_c | base_flexible_kwh | base_washer_completed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| milp | 42 | 109.548 | 14:30 | 2.024 | 89.709 | 69.072 | 0.770 | 0.000 | 0.000 | 221.000 | 24 |
| noisy_milp | 42 | 109.548 | 14:30 | 2.015 | 88.709 | 73.072 | 0.824 | 0.000 | 0.000 | 227.000 | 24 |
| noisy_milp | 43 | 105.548 | 14:30 | 1.937 | 88.376 | 73.072 | 0.827 | 0.000 | 0.000 | 230.000 | 24 |
| noisy_milp | 44 | 108.332 | 17:00 | 1.981 | 91.043 | 69.072 | 0.759 | 0.000 | 0.000 | 235.000 | 24 |
| rule_based | 42 | 101.532 | 21:00 | 1.945 | 86.043 | 77.072 | 0.896 | 113.632 | 1.808 | 175.000 | 24 |
| llm | 42 | 102.332 | 17:00 | 1.953 | 86.709 | 73.072 | 0.843 | 107.409 | 2.949 | 180.000 | 24 |

### CPP day

| agent | seed | cpp_peak_kw | cpp_time_of_peak | cpp_par | cpp_cpp_window_mean_kw | cpp_post_20:00-20:30_kw | cpp_post_over_cpp_ratio | cpp_degree_hours_above_ceiling | cpp_max_exceedance_c | cpp_flexible_kwh | cpp_washer_completed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| milp | 42 | 125.188 | 16:30 | 2.253 | 72.709 | 113.072 | 1.555 | 0.000 | 0.000 | 256.000 | 24 |
| noisy_milp | 42 | 125.188 | 16:30 | 2.256 | 73.376 | 107.072 | 1.459 | 0.000 | 0.000 | 254.000 | 24 |
| noisy_milp | 43 | 125.188 | 16:30 | 2.254 | 74.709 | 97.072 | 1.299 | 0.000 | 0.000 | 255.000 | 24 |
| noisy_milp | 44 | 125.188 | 16:30 | 2.234 | 73.709 | 105.072 | 1.425 | 0.000 | 0.000 | 267.000 | 24 |
| rule_based | 42 | 107.072 | 20:00 | 2.073 | 78.043 | 107.072 | 1.372 | 164.377 | 2.774 | 162.000 | 24 |
| llm | 42 | 101.532 | 21:00 | 1.948 | 80.709 | 85.072 | 1.054 | 143.795 | 3.170 | 173.000 | 24 |

### Elasticity and daily flexible change

| agent | seed | elasticity_flexible | elasticity_flexible_midpoint | elasticity_total | elasticity_total_midpoint | total_in_conway | daily_flexible_kwh_change | daily_flexible_pct_change |
|---|---|---|---|---|---|---|---|---|
| milp | 42 | -0.685 | -1.831 | -0.160 | -0.282 | True | 35.000 | 15.840 |
| noisy_milp | 42 | -0.649 | -1.674 | -0.146 | -0.255 | True | 27.000 | 11.890 |
| noisy_milp | 43 | -0.588 | -1.434 | -0.131 | -0.226 | True | 25.000 | 10.870 |
| noisy_milp | 44 | -0.657 | -1.707 | -0.161 | -0.283 | True | 32.000 | 13.620 |
| rule_based | 42 | -0.391 | -0.808 | -0.079 | -0.131 | True | -13.000 | -7.430 |
| llm | 42 | -0.282 | -0.538 | -0.059 | -0.096 | True | -7.000 | -3.890 |

## Elasticity detail (seed 42)

| agent | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total | in_conway_total |
|---|---|---|---|---|---|---|
| llm | 24 | -0.282 | nan | -0.282 | -0.059 | True |
| milp | 24 | -0.685 | nan | -0.685 | -0.160 | True |
| noisy_milp | 24 | -0.649 | nan | -0.649 | -0.146 | True |
| rule_based | 24 | -0.391 | nan | -0.391 | -0.079 | True |

### By archetype (prediction: C more elastic than B than A)

| agent | archetype | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total |
|---|---|---|---|---|---|---|
| llm | A | 8 | -0.548 | nan | -0.548 | -0.213 |
| llm | B | 8 | 0.000 | nan | 0.000 | 0.000 |
| llm | C | 8 | -0.282 | nan | -0.282 | -0.058 |
| milp | A | 8 | -0.762 | nan | -0.762 | -0.327 |
| milp | B | 8 | -0.635 | nan | -0.635 | -0.108 |
| milp | C | 8 | -0.662 | nan | -0.662 | -0.145 |
| noisy_milp | A | 8 | -0.713 | nan | -0.713 | -0.297 |
| noisy_milp | B | 8 | -0.592 | nan | -0.592 | -0.101 |
| noisy_milp | C | 8 | -0.645 | nan | -0.645 | -0.132 |
| rule_based | A | 8 | -0.376 | nan | -0.376 | -0.152 |
| rule_based | B | 8 | -0.264 | nan | -0.264 | -0.037 |
| rule_based | C | 8 | -0.517 | nan | -0.517 | -0.093 |

Archetype ordering: {"llm": {"values": {"A": -0.21326396085813182, "B": 0.0, "C": -0.057578272801369894}, "C_more_elastic_than_B": true, "B_more_elastic_than_A": false}, "rule_based": {"values": {"A": -0.15162688758244708, "B": -0.0372380976883943, "C": -0.09319808058167924}, "C_more_elastic_than_B": true, "B_more_elastic_than_A": false}}

## Load-curve metrics (population, primary seed)

| scope | series | day | par | peak_kw | time_of_peak | rebound_peak_kw | rebound_delta_kw |
|---|---|---|---|---|---|---|---|
| population | SGSC base | base | 1.789 | 80.332 | 17:00 | 77.532 | 0.000 |
| population | MILP | base | 2.024 | 109.548 | 14:30 | 77.532 | 35.540 |
| population | noisy-MILP | base | 2.015 | 109.548 | 14:30 | 79.532 | 27.540 |
| population | rule-based | base | 1.945 | 101.532 | 21:00 | 101.532 | 5.540 |
| population | LLM | base | 1.953 | 102.332 | 17:00 | 93.532 | 8.000 |
| population | SGSC base | cpp | 1.789 | 80.332 | 17:00 | 77.532 | 0.000 |
| population | MILP | cpp | 2.253 | 125.188 | 16:30 | 113.072 | 35.540 |
| population | noisy-MILP | cpp | 2.256 | 125.188 | 16:30 | 107.072 | 27.540 |
| population | rule-based | cpp | 2.073 | 107.072 | 20:00 | 107.072 | 5.540 |
| population | LLM | cpp | 1.948 | 101.532 | 21:00 | 101.532 | 8.000 |

Figures: `overlay_population.png`, `overlay_<household>.png`.

## Seed spread

| agent | seed | elasticity_total | elasticity_flexible | cpp_peak_kw | daily_flexible_kwh_change |
|---|---|---|---|---|---|
| noisy_milp | 42 | -0.146 | -0.649 | 125.188 | 27.000 |
| noisy_milp | 43 | -0.131 | -0.588 | 125.188 | 25.000 |
| noisy_milp | 44 | -0.161 | -0.657 | 125.188 | 32.000 |
| llm | 42 | -0.059 | -0.282 | 101.532 | -7.000 |

| agent | elasticity_total_min | elasticity_total_max | elasticity_total_std | elasticity_flexible_min | elasticity_flexible_max | elasticity_flexible_std |
|---|---|---|---|---|---|---|
| llm | -0.059 | -0.059 | nan | -0.282 | -0.282 | nan |
| noisy_milp | -0.161 | -0.131 | 0.015 | -0.657 | -0.588 | 0.038 |

## LLM call accounting

```
{
  "llm_prompted_decisions": 2304,
  "llm_live_calls": 787,
  "llm_model_versions": {
    "42": {
      "gemini-3.5-flash-lite": 2304
    }
  }
}
```
