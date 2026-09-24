# Phase 3 run -- LIVE

- households: [8596070, 9911140]
- steps: ['elasticity', 'overlay']
- provider/model: gemini / gemini-3.5-flash-lite

## Elasticity (seed 42, wtp_scale 1.0, method `base`)

Window ['17:00', '20:00']; price 0.550 -> 1.200 $/kWh. Conway range (-0.47, -0.026). Per-appliance columns are pooled over households; `total` is whole-household (base + flexible) and is what the criterion is judged on.

### Aggregate by agent (discriminant check)

| agent | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total | in_conway_total |
|---|---|---|---|---|---|---|
| milp | 2 | -0.846 | nan | -0.846 | -0.258 | True |
| noisy_milp | 2 | -0.846 | 0.000 | -0.769 | -0.187 | True |
| llm | 2 | 0.000 | 0.000 | 0.000 | 0.000 | False |

Midpoint-formula robustness:

| agent | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total | in_conway_total |
|---|---|---|---|---|---|---|
| milp | 2 | -2.692 | nan | -2.692 | -0.484 | False |
| noisy_milp | 2 | -2.692 | 0.000 | -2.244 | -0.335 | True |
| llm | 2 | 0.000 | 0.000 | 0.000 | 0.000 | False |

### By archetype (prediction: C more elastic than B than A)

| agent | archetype | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total |
|---|---|---|---|---|---|---|
| llm | A | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| llm | C | 1 | 0.000 | nan | 0.000 | 0.000 |
| milp | A | 1 | -0.846 | nan | -0.846 | -0.716 |
| milp | C | 1 | -0.846 | nan | -0.846 | -0.157 |
| noisy_milp | A | 1 | -0.846 | 0.000 | -0.705 | -0.575 |
| noisy_milp | C | 1 | -0.846 | nan | -0.846 | -0.112 |

LLM ordering: {"values": {"A": 0.0, "C": 0.0}, "C_more_elastic_than_B": false, "B_more_elastic_than_A": false, "C_more_elastic_than_A": false}

### Per household

| household_id | agent | archetype | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total |
|---|---|---|---|---|---|---|
| 8596070 | llm | C | 0.000 | nan | 0.000 | 0.000 |
| 9911140 | llm | A | 0.000 | 0.000 | 0.000 | 0.000 |
| 8596070 | milp | C | -0.846 | nan | -0.846 | -0.157 |
| 9911140 | milp | A | -0.846 | nan | -0.846 | -0.716 |
| 8596070 | noisy_milp | C | -0.846 | nan | -0.846 | -0.112 |
| 9911140 | noisy_milp | A | -0.846 | 0.000 | -0.705 | -0.575 |

## Load-curve metrics (four-way)

Rebound = peak kW in the 4 steps after the CPP window ends; delta = CPP day - baseline day.

| scope | series | day | par | peak_kw | time_of_peak | rebound_peak_kw | rebound_delta_kw |
|---|---|---|---|---|---|---|---|
| population | SGSC base | base | 1.840 | 4.916 | 11:00 | 4.268 | 0.000 |
| population | MILP | base | 2.434 | 8.856 | 17:00 | 5.868 | 3.936 |
| population | noisy-MILP | base | 2.456 | 8.732 | 14:30 | 5.868 | 3.136 |
| population | LLM | base | 2.546 | 8.732 | 14:30 | 6.996 | 0.000 |
| population | SGSC base | cpp | 1.840 | 4.916 | 11:00 | 4.268 | 0.000 |
| population | MILP | cpp | 2.634 | 9.804 | 20:00 | 9.804 | 3.936 |
| population | noisy-MILP | cpp | 2.392 | 9.004 | 20:00 | 9.004 | 3.136 |
| population | LLM | cpp | 2.546 | 8.732 | 14:30 | 6.996 | 0.000 |

Figures: `overlay_population.png`, `overlay_<household>.png`.

## LLM call accounting

```
{
  "prompted_decisions": 218,
  "live_calls_this_process": 109,
  "client_stats_by_seed": {
    "42": {
      "live_calls": 109,
      "cache_hits": 109
    }
  },
  "dry_run": false,
  "wall_seconds": 1272.2
}
```
