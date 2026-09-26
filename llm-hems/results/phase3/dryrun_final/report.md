# Phase 3 run -- DRY RUN (stub LLM, numbers are pipeline checks only)

- households (24): [8201999, 8208263, 8211127, 8395524, 8456189, 8463253, 8481835, 8568499, 8596070, 8619039, 8632083, 8638181, 9012382, 9911140, 9911204, 9911250, 9911254, 10042536, 10239118, 10278838, 10444433, 10556929, 10649333, 11105177]
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
| milp | 42 | 109.548 | 14:30 | 1.966 | 89.709 | 85.072 | 0.948 | 0.000 | 0.000 | 259.400 | 24 |
| noisy_milp | 42 | 109.548 | 14:30 | 1.957 | 89.909 | 84.272 | 0.937 | 0.000 | 0.000 | 265.400 | 24 |
| noisy_milp | 43 | 105.548 | 14:30 | 1.882 | 89.176 | 82.672 | 0.927 | 0.000 | 0.000 | 268.400 | 24 |
| noisy_milp | 44 | 108.332 | 17:00 | 1.924 | 92.243 | 82.672 | 0.896 | 0.000 | 0.000 | 273.400 | 24 |
| rule_based | 42 | 111.532 | 17:00 | 2.073 | 96.709 | 77.072 | 0.797 | 113.632 | 1.808 | 213.400 | 24 |
| llm | 42 | 109.188 | 16:30 | 1.997 | 93.443 | 98.672 | 1.056 | 53.419 | 1.449 | 234.400 | 24 |

### CPP day

| agent | seed | cpp_peak_kw | cpp_time_of_peak | cpp_par | cpp_cpp_window_mean_kw | cpp_post_20:00-20:30_kw | cpp_post_over_cpp_ratio | cpp_degree_hours_above_ceiling | cpp_max_exceedance_c | cpp_flexible_kwh | cpp_washer_completed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| milp | 42 | 129.072 | 20:00 | 2.258 | 72.709 | 129.072 | 1.775 | 0.000 | 0.000 | 294.400 | 24 |
| noisy_milp | 42 | 125.188 | 16:30 | 2.193 | 74.576 | 118.272 | 1.586 | 0.000 | 0.000 | 292.400 | 24 |
| noisy_milp | 43 | 125.188 | 16:30 | 2.191 | 75.509 | 106.672 | 1.413 | 0.000 | 0.000 | 293.400 | 24 |
| noisy_milp | 44 | 125.188 | 16:30 | 2.172 | 74.909 | 118.672 | 1.584 | 0.000 | 0.000 | 305.400 | 24 |
| rule_based | 42 | 116.672 | 20:00 | 2.191 | 85.509 | 116.672 | 1.364 | 164.377 | 2.774 | 200.400 | 24 |
| llm | 42 | 127.872 | 20:00 | 2.357 | 81.176 | 127.872 | 1.575 | 101.372 | 2.547 | 224.400 | 24 |

### Elasticity and daily flexible change

| agent | seed | elasticity_flexible | elasticity_flexible_midpoint | elasticity_total | elasticity_total_midpoint | total_in_conway | daily_flexible_kwh_change | daily_flexible_pct_change |
|---|---|---|---|---|---|---|---|---|
| milp | 42 | -0.685 | -1.831 | -0.160 | -0.282 | True | 35.000 | 13.490 |
| noisy_milp | 42 | -0.612 | -1.525 | -0.144 | -0.251 | True | 27.000 | 10.170 |
| noisy_milp | 43 | -0.565 | -1.349 | -0.130 | -0.223 | True | 25.000 | 9.310 |
| noisy_milp | 44 | -0.623 | -1.570 | -0.159 | -0.279 | True | 32.000 | 11.700 |
| rule_based | 42 | -0.338 | -0.673 | -0.098 | -0.165 | True | -13.000 | -6.090 |
| llm | 42 | -0.420 | -0.888 | -0.111 | -0.189 | True | -10.000 | -4.270 |

## Elasticity detail (seed 42)

| agent | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total | in_conway_total |
|---|---|---|---|---|---|---|
| llm | 24 | -0.380 | -0.497 | -0.420 | -0.111 | True |
| milp | 24 | -0.685 | nan | -0.685 | -0.160 | True |
| noisy_milp | 24 | -0.649 | 0.000 | -0.612 | -0.144 | True |
| rule_based | 24 | -0.391 | -0.254 | -0.338 | -0.098 | True |

### By archetype (prediction: C more elastic than B than A)

| agent | archetype | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total |
|---|---|---|---|---|---|---|
| llm | A | 8 | -0.423 | -0.542 | -0.469 | -0.231 |
| llm | B | 8 | -0.299 | -0.296 | -0.298 | -0.061 |
| llm | C | 8 | -0.423 | -0.658 | -0.496 | -0.110 |
| milp | A | 8 | -0.762 | nan | -0.762 | -0.327 |
| milp | B | 8 | -0.635 | nan | -0.635 | -0.108 |
| milp | C | 8 | -0.662 | nan | -0.662 | -0.145 |
| noisy_milp | A | 8 | -0.713 | 0.000 | -0.670 | -0.289 |
| noisy_milp | B | 8 | -0.592 | 0.000 | -0.559 | -0.100 |
| noisy_milp | C | 8 | -0.645 | 0.000 | -0.610 | -0.130 |
| rule_based | A | 8 | -0.376 | 0.000 | -0.220 | -0.118 |
| rule_based | B | 8 | -0.264 | -0.212 | -0.245 | -0.051 |
| rule_based | C | 8 | -0.517 | -0.635 | -0.558 | -0.141 |

Archetype ordering: {"llm": {"values": {"A": -0.23145306938331778, "B": -0.06072841474997823, "C": -0.10952405356136197}, "C_more_elastic_than_B": true, "B_more_elastic_than_A": false}, "rule_based": {"values": {"A": -0.11784051892679423, "B": -0.050818401005879726, "C": -0.14067781127249473}, "C_more_elastic_than_B": true, "B_more_elastic_than_A": false}}

## Load-curve metrics (population, primary seed)

| scope | series | day | par | peak_kw | time_of_peak | rebound_peak_kw | rebound_delta_kw |
|---|---|---|---|---|---|---|---|
| population | SGSC base | base | 1.789 | 80.332 | 17:00 | 77.532 | 0.000 |
| population | MILP | base | 1.966 | 109.548 | 14:30 | 93.532 | 35.540 |
| population | noisy-MILP | base | 1.957 | 109.548 | 14:30 | 96.332 | 21.940 |
| population | rule-based | base | 2.073 | 111.532 | 17:00 | 103.132 | 13.540 |
| population | LLM | base | 1.997 | 109.188 | 16:30 | 98.672 | 29.200 |
| population | SGSC base | cpp | 1.789 | 80.332 | 17:00 | 77.532 | 0.000 |
| population | MILP | cpp | 2.258 | 129.072 | 20:00 | 129.072 | 35.540 |
| population | noisy-MILP | cpp | 2.193 | 125.188 | 16:30 | 118.272 | 21.940 |
| population | rule-based | cpp | 2.191 | 116.672 | 20:00 | 116.672 | 13.540 |
| population | LLM | cpp | 2.357 | 127.872 | 20:00 | 127.872 | 29.200 |

Figures: `overlay_population.png`, `overlay_<household>.png`.

## Seed spread

| agent | seed | elasticity_total | elasticity_flexible | cpp_peak_kw | daily_flexible_kwh_change |
|---|---|---|---|---|---|
| noisy_milp | 42 | -0.144 | -0.612 | 125.188 | 27.000 |
| noisy_milp | 43 | -0.130 | -0.565 | 125.188 | 25.000 |
| noisy_milp | 44 | -0.159 | -0.623 | 125.188 | 32.000 |
| llm | 42 | -0.111 | -0.420 | 127.872 | -10.000 |

| agent | elasticity_total_min | elasticity_total_max | elasticity_total_std | elasticity_flexible_min | elasticity_flexible_max | elasticity_flexible_std |
|---|---|---|---|---|---|---|
| llm | -0.111 | -0.111 | nan | -0.420 | -0.420 | nan |
| noisy_milp | -0.159 | -0.130 | 0.015 | -0.623 | -0.565 | 0.031 |

## Discriminating scenario (single CPP spike + near washer deadline)

Spike 18:00-19:00 (severe). Coherence judged against the scenario's true deadline.

| agent | variant | coherent_share | count |
|---|---|---|---|
| llm | deadline_forces | 1.000 | 24 |
| llm | slack_remains | 0.833 | 24 |
| milp | deadline_forces | 1.000 | 24 |
| milp | slack_remains | 1.000 | 24 |
| noisy_milp | deadline_forces | 0.722 | 72 |
| noisy_milp | slack_remains | 0.722 | 72 |
| rule_based | deadline_forces | 0.333 | 24 |
| rule_based | slack_remains | 1.000 | 24 |

## Projected live LLM calls (from this dry run)

```
{
  "prompted_decisions": 4999,
  "prompted_by_appliance": {
    "hvac": 4608,
    "washer": 391
  },
  "distinct_cache_keys": 1081,
  "already_in_cache": 0,
  "cache_paths_checked": [
    "results/phase3/llm_cache_final.jsonl"
  ],
  "projected_live_calls": 1081,
  "upper_bound_prompted_decisions": 5172,
  "rate_limit_rpm": 5.0,
  "projected_hours_at_rpm": 3.6,
  "upper_bound_hours_at_rpm": 17.24,
  "caveat": "counted on the dry-run stub's trajectories. The HVAC decision count is exact (prompted every step), but which prompts repeat -- across a household's baseline, CPP and scenario days until they diverge, and across households that reach the same rendered state -- depends on the real LLM's decisions, so the distinct-key count is an estimate. The upper bound (every HVAC step + every washer step from release to latest start, no cache reuse) is the hard ceiling. Retries not counted."
}
```

## LLM call accounting

```
{
  "llm_prompted_decisions": 4999,
  "llm_live_calls": 0,
  "llm_model_versions": {}
}
```
