# Phase 3 run -- LIVE

- households (3): [8211127, 10042536, 10444433]
- steps: ['elasticity']
- band mode: `fixed` (fixed +/-2.0 C for every household); comfort orientation = persona's
- WTP scaling: global_scale 1.0 x archetype factor {'A': 1.3, 'B': 1.0, 'C': 0.7}, identical for ['rule_based', 'llm']; calibration enabled: False
- seeds: primary 42; LLM [42] (T=0); noisy-MILP [42, 43, 44]; MILP + rule-based deterministic
- thermal heterogeneity: {'enabled': True, 'r_range': [0.8, 1.2], 'c_range': [0.8, 1.2], 't0_range': [-1.0, 1.0], 'seed': 42}
- rule-based k (fixed band fit): {'comfort_prioritising': 3.791, 'balanced': 0.62, 'comfort_flexible': 0.66}, w_max 0.888
- provider/model: gemini / gemini-3.8-flash
- provenance: `provenance.json`; full config: `config_resolved.yaml`

## Summary by agent (population aggregate)

Peak / PAR on the summed load of all households; post = first post-CPP half-hour vs the CPP-window mean; degree-hours = sum over households of max(0, indoor - ceiling) x 0.5 h; elasticities pooled (`base`; midpoint alongside), total = whole household; Conway [-0.47, -0.026].

### Baseline day

| agent | seed | base_peak_kw | base_time_of_peak | base_par | base_cpp_window_mean_kw | base_post_20:00-20:30_kw | base_post_over_cpp_ratio | base_degree_hours_above_ceiling | base_max_exceedance_c | base_flexible_kwh | base_washer_completed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| milp | 42 | 21.844 | 21:00 | 3.186 | 9.850 | 7.748 | 0.787 | 0.000 | 0.000 | 26.000 | 3 |
| noisy_milp | 42 | 21.844 | 21:00 | 3.186 | 9.850 | 7.748 | 0.787 | 0.000 | 0.000 | 26.000 | 3 |
| rule_based | 42 | 21.844 | 21:00 | 3.306 | 9.183 | 11.748 | 1.279 | 14.634 | 1.789 | 20.000 | 3 |
| llm | 42 | 23.844 | 21:00 | 3.586 | 8.850 | 9.748 | 1.101 | 22.502 | 2.694 | 21.000 | 3 |

### CPP day

| agent | seed | cpp_peak_kw | cpp_time_of_peak | cpp_par | cpp_cpp_window_mean_kw | cpp_post_20:00-20:30_kw | cpp_post_over_cpp_ratio | cpp_degree_hours_above_ceiling | cpp_max_exceedance_c | cpp_flexible_kwh | cpp_washer_completed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| milp | 42 | 21.844 | 21:00 | 3.148 | 8.183 | 13.748 | 1.680 | 0.000 | 0.000 | 28.000 | 3 |
| noisy_milp | 42 | 21.844 | 21:00 | 3.110 | 7.850 | 13.748 | 1.751 | 0.000 | 0.000 | 30.000 | 3 |
| rule_based | 42 | 25.844 | 21:00 | 3.937 | 8.517 | 11.748 | 1.379 | 20.395 | 2.406 | 19.000 | 3 |
| llm | 42 | 25.844 | 21:00 | 3.912 | 8.517 | 7.748 | 0.910 | 25.084 | 2.694 | 20.000 | 3 |

### Elasticity and daily flexible change

| agent | seed | elasticity_flexible | elasticity_flexible_midpoint | elasticity_total | elasticity_total_midpoint | total_in_conway | daily_flexible_kwh_change | daily_flexible_pct_change |
|---|---|---|---|---|---|---|---|---|
| milp | 42 | -0.604 | -1.496 | -0.143 | -0.249 | True | 2.000 | 7.690 |
| noisy_milp | 42 | -0.725 | -2.019 | -0.172 | -0.304 | True | 4.000 | 15.380 |
| rule_based | 42 | -0.338 | -0.673 | -0.061 | -0.101 | True | -1.000 | -5.000 |
| llm | 42 | -0.212 | -0.385 | -0.032 | -0.052 | True | -1.000 | -4.760 |

## Elasticity detail (seed 42)

| agent | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total | in_conway_total |
|---|---|---|---|---|---|---|
| llm | 3 | -0.212 | nan | -0.212 | -0.032 | True |
| milp | 3 | -0.604 | nan | -0.604 | -0.143 | True |
| noisy_milp | 3 | -0.725 | nan | -0.725 | -0.172 | True |
| rule_based | 3 | -0.338 | nan | -0.338 | -0.061 | True |

### By archetype (prediction: C more elastic than B than A)

| agent | archetype | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total |
|---|---|---|---|---|---|---|
| llm | B | 2 | 0.000 | nan | 0.000 | 0.000 |
| llm | C | 1 | -0.423 | nan | -0.423 | -0.043 |
| milp | B | 2 | -0.635 | nan | -0.635 | -0.282 |
| milp | C | 1 | -0.564 | nan | -0.564 | -0.082 |
| noisy_milp | B | 2 | -0.846 | nan | -0.846 | -0.376 |
| noisy_milp | C | 1 | -0.564 | nan | -0.564 | -0.082 |
| rule_based | B | 2 | 0.000 | nan | 0.000 | 0.000 |
| rule_based | C | 1 | -0.846 | nan | -0.846 | -0.087 |

Archetype ordering: {"llm": {"values": {"B": 0.0, "C": -0.043308109640385194}, "C_more_elastic_than_B": true, "B_more_elastic_than_A": false}, "rule_based": {"values": {"B": 0.0, "C": -0.08661621928077039}, "C_more_elastic_than_B": true, "B_more_elastic_than_A": false}}

## LLM call accounting

```
{
  "llm_prompted_decisions": 288,
  "llm_live_calls": 167,
  "llm_model_versions": {
    "42": {
      "gemini-3.8-flash": 288
    }
  }
}
```
