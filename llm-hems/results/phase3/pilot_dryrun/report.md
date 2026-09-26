# Phase 3 run -- DRY RUN (stub LLM, numbers are pipeline checks only)

- households (3): [8211127, 10042536, 10444433]
- steps: ['elasticity', 'overlay', 'scenario', 'seeds']
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
| milp | 42 | 24.244 | 21:00 | 3.436 | 9.850 | 10.148 | 1.030 | 0.000 | 0.000 | 30.800 | 3 |
| noisy_milp | 42 | 24.244 | 21:00 | 3.436 | 9.983 | 9.348 | 0.936 | 0.000 | 0.000 | 30.800 | 3 |
| noisy_milp | 43 | 24.244 | 21:00 | 3.415 | 9.650 | 8.548 | 0.886 | 0.000 | 0.000 | 31.800 | 3 |
| noisy_milp | 44 | 26.244 | 21:00 | 3.654 | 10.317 | 10.148 | 0.984 | 0.000 | 0.000 | 33.800 | 3 |
| rule_based | 42 | 21.844 | 21:00 | 3.209 | 10.783 | 11.748 | 1.089 | 14.634 | 1.789 | 24.800 | 3 |
| llm | 42 | 27.844 | 21:00 | 4.017 | 10.717 | 9.348 | 0.872 | 6.510 | 1.088 | 27.800 | 3 |

### CPP day

| agent | seed | cpp_peak_kw | cpp_time_of_peak | cpp_par | cpp_cpp_window_mean_kw | cpp_post_20:00-20:30_kw | cpp_post_over_cpp_ratio | cpp_degree_hours_above_ceiling | cpp_max_exceedance_c | cpp_flexible_kwh | cpp_washer_completed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| milp | 42 | 24.244 | 21:00 | 3.396 | 8.183 | 16.148 | 1.973 | 0.000 | 0.000 | 32.800 | 3 |
| noisy_milp | 42 | 24.244 | 21:00 | 3.356 | 7.983 | 15.348 | 1.923 | 0.000 | 0.000 | 34.800 | 3 |
| noisy_milp | 43 | 24.244 | 21:00 | 3.376 | 8.317 | 10.548 | 1.268 | 0.000 | 0.000 | 33.800 | 3 |
| noisy_milp | 44 | 26.244 | 21:00 | 3.571 | 8.317 | 14.148 | 1.701 | 0.000 | 0.000 | 37.800 | 3 |
| rule_based | 42 | 26.644 | 21:00 | 3.939 | 9.450 | 14.148 | 1.497 | 20.395 | 2.406 | 23.800 | 3 |
| llm | 42 | 25.444 | 21:00 | 3.671 | 8.917 | 16.148 | 1.811 | 11.726 | 2.087 | 27.800 | 3 |

### Elasticity and daily flexible change

| agent | seed | elasticity_flexible | elasticity_flexible_midpoint | elasticity_total | elasticity_total_midpoint | total_in_conway | daily_flexible_kwh_change | daily_flexible_pct_change |
|---|---|---|---|---|---|---|---|---|
| milp | 42 | -0.604 | -1.496 | -0.143 | -0.249 | True | 2.000 | 6.490 |
| noisy_milp | 42 | -0.686 | -1.836 | -0.170 | -0.300 | True | 4.000 | 12.990 |
| noisy_milp | 43 | -0.529 | -1.224 | -0.117 | -0.200 | True | 2.000 | 6.290 |
| noisy_milp | 44 | -0.604 | -1.496 | -0.164 | -0.289 | True | 4.000 | 11.830 |
| rule_based | 42 | -0.345 | -0.690 | -0.105 | -0.177 | True | -1.000 | -4.030 |
| llm | 42 | -0.476 | -1.054 | -0.142 | -0.247 | True | 0.000 | 0.000 |

## Elasticity detail (seed 42)

| agent | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total | in_conway_total |
|---|---|---|---|---|---|---|
| llm | 3 | -0.423 | -0.564 | -0.476 | -0.142 | True |
| milp | 3 | -0.604 | nan | -0.604 | -0.143 | True |
| noisy_milp | 3 | -0.725 | 0.000 | -0.686 | -0.170 | True |
| rule_based | 3 | -0.338 | -0.353 | -0.345 | -0.105 | True |

### By archetype (prediction: C more elastic than B than A)

| agent | archetype | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total |
|---|---|---|---|---|---|---|
| llm | B | 2 | -0.423 | -0.423 | -0.423 | -0.237 |
| llm | C | 1 | -0.423 | -0.846 | -0.582 | -0.090 |
| milp | B | 2 | -0.635 | nan | -0.635 | -0.282 |
| milp | C | 1 | -0.564 | nan | -0.564 | -0.082 |
| noisy_milp | B | 2 | -0.846 | 0.000 | -0.769 | -0.360 |
| noisy_milp | C | 1 | -0.564 | nan | -0.564 | -0.082 |
| rule_based | B | 2 | 0.000 | -0.212 | -0.109 | -0.060 |
| rule_based | C | 1 | -0.846 | -0.635 | -0.752 | -0.128 |

Archetype ordering: {"llm": {"values": {"B": -0.23726711423872296, "C": -0.0897646090046514}, "C_more_elastic_than_B": false, "B_more_elastic_than_A": false}, "rule_based": {"values": {"B": -0.060374873075550964, "C": -0.12809595551576825}, "C_more_elastic_than_B": true, "B_more_elastic_than_A": false}}

## Load-curve metrics (population, primary seed)

| scope | series | day | par | peak_kw | time_of_peak | rebound_peak_kw | rebound_delta_kw |
|---|---|---|---|---|---|---|---|
| population | SGSC base | base | 3.784 | 21.844 | 21:00 | 21.844 | 0.000 |
| population | MILP | base | 3.436 | 24.244 | 21:00 | 24.244 | 0.000 |
| population | noisy-MILP | base | 3.436 | 24.244 | 21:00 | 24.244 | 0.000 |
| population | rule-based | base | 3.209 | 21.844 | 21:00 | 21.844 | 4.800 |
| population | LLM | base | 4.017 | 27.844 | 21:00 | 27.844 | -2.400 |
| population | SGSC base | cpp | 3.784 | 21.844 | 21:00 | 21.844 | 0.000 |
| population | MILP | cpp | 3.396 | 24.244 | 21:00 | 24.244 | 0.000 |
| population | noisy-MILP | cpp | 3.356 | 24.244 | 21:00 | 24.244 | 0.000 |
| population | rule-based | cpp | 3.939 | 26.644 | 21:00 | 26.644 | 4.800 |
| population | LLM | cpp | 3.671 | 25.444 | 21:00 | 25.444 | -2.400 |

Figures: `overlay_population.png`, `overlay_<household>.png`.

## Seed spread

| agent | seed | elasticity_total | elasticity_flexible | cpp_peak_kw | daily_flexible_kwh_change |
|---|---|---|---|---|---|
| noisy_milp | 42 | -0.170 | -0.686 | 24.244 | 4.000 |
| noisy_milp | 43 | -0.117 | -0.529 | 24.244 | 2.000 |
| noisy_milp | 44 | -0.164 | -0.604 | 26.244 | 4.000 |
| llm | 42 | -0.142 | -0.476 | 25.444 | 0.000 |

| agent | elasticity_total_min | elasticity_total_max | elasticity_total_std | elasticity_flexible_min | elasticity_flexible_max | elasticity_flexible_std |
|---|---|---|---|---|---|---|
| llm | -0.142 | -0.142 | nan | -0.476 | -0.476 | nan |
| noisy_milp | -0.170 | -0.117 | 0.029 | -0.686 | -0.529 | 0.079 |

## Discriminating scenario (single CPP spike + near washer deadline)

Spike 18:00-19:00 (severe). Coherence judged against the scenario's true deadline.

| agent | variant | coherent_share | count |
|---|---|---|---|
| llm | deadline_forces | 1.000 | 3 |
| llm | slack_remains | 0.667 | 3 |
| milp | deadline_forces | 1.000 | 3 |
| milp | slack_remains | 1.000 | 3 |
| noisy_milp | deadline_forces | 0.667 | 9 |
| noisy_milp | slack_remains | 0.667 | 9 |
| rule_based | deadline_forces | 0.000 | 3 |
| rule_based | slack_remains | 1.000 | 3 |

## Projected live LLM calls (from this dry run)

```
{
  "prompted_decisions": 630,
  "prompted_by_appliance": {
    "hvac": 576,
    "washer": 54
  },
  "distinct_cache_keys": 197,
  "already_in_cache": 0,
  "cache_paths_checked": [
    "results/phase3/llm_cache_final.jsonl"
  ],
  "projected_live_calls": 197,
  "upper_bound_prompted_decisions": 651,
  "rate_limit_rpm": 1000.0,
  "projected_hours_at_rpm": 0.0,
  "upper_bound_hours_at_rpm": 0.01,
  "caveat": "counted on the dry-run stub's trajectories. The HVAC decision count is exact (prompted every step), but which prompts repeat -- across a household's baseline, CPP and scenario days until they diverge, and across households that reach the same rendered state -- depends on the real LLM's decisions, so the distinct-key count is an estimate. The upper bound (every HVAC step + every washer step from release to latest start, no cache reuse) is the hard ceiling. Retries not counted."
}
```

## LLM call accounting

```
{
  "llm_prompted_decisions": 630,
  "llm_live_calls": 0,
  "llm_model_versions": {}
}
```
