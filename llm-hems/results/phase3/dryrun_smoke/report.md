# Phase 3 run -- DRY RUN (stub LLM, numbers are pipeline checks only)

- households: [8596070, 9911140, 8632083]
- steps: ['calibrate', 'elasticity', 'overlay', 'scenario', 'seeds']
- provider/model: gemini / gemini-3.5-flash-lite

## Calibration (pre-registered)

- target: -0.25 inside Conway [-0.47, -0.026]; split fraction 0.5 per archetype, split_seed 42, fit_seed 42
- calibration: [8632083, 9911140]
- held-out: [8596070]
- **frozen wtp_scale = 0.5** (calibration total elasticity -0.105, inside range: True)

| wtp_scale | elasticity_total | elasticity_flexible | in_range | distance_to_target |
|---|---|---|---|---|
| 0.250 | -0.073 | -0.846 | True | 0.177 |
| 0.500 | -0.105 | -0.618 | True | 0.145 |
| 1.000 | -0.083 | -0.366 | True | 0.167 |
| 2.000 | -0.071 | -0.289 | True | 0.179 |
| 4.000 | -0.025 | -0.092 | False | 0.225 |

## Elasticity (seed 42, wtp_scale 0.5, method `base`)

Window ['17:00', '20:00']; price 0.550 -> 1.200 $/kWh. Conway range (-0.47, -0.026). Per-appliance columns are pooled over households; `total` is whole-household (base + flexible) and is what the criterion is judged on.

### Aggregate by agent (discriminant check)

| agent | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total | in_conway_total |
|---|---|---|---|---|---|---|
| milp | 1 | -0.846 | nan | -0.846 | -0.157 | True |
| noisy_milp | 1 | -0.846 | nan | -0.846 | -0.112 | True |
| llm | 1 | -0.846 | nan | -0.846 | -0.060 | True |

Midpoint-formula robustness:

| agent | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total | in_conway_total |
|---|---|---|---|---|---|---|
| milp | 1 | -2.692 | nan | -2.692 | -0.276 | True |
| noisy_milp | 1 | -2.692 | nan | -2.692 | -0.191 | True |
| llm | 1 | -2.692 | nan | -2.692 | -0.099 | True |

### By archetype (prediction: C more elastic than B than A)

| agent | archetype | n | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total |
|---|---|---|---|---|---|---|
| llm | C | 1 | -0.846 | nan | -0.846 | -0.060 |
| milp | C | 1 | -0.846 | nan | -0.846 | -0.157 |
| noisy_milp | C | 1 | -0.846 | nan | -0.846 | -0.112 |

LLM ordering: {"values": {"C": -0.05988349937394524}, "C_more_elastic_than_B": false, "B_more_elastic_than_A": false, "C_more_elastic_than_A": false}

### Per household

| household_id | agent | archetype | elasticity_hvac | elasticity_washer | elasticity_flexible | elasticity_total |
|---|---|---|---|---|---|---|
| 8596070 | llm | C | -0.846 | nan | -0.846 | -0.060 |
| 8596070 | milp | C | -0.846 | nan | -0.846 | -0.157 |
| 8596070 | noisy_milp | C | -0.846 | nan | -0.846 | -0.112 |

## Load-curve metrics (four-way)

Rebound = peak kW in the 4 steps after the CPP window ends; delta = CPP day - baseline day.

| scope | series | day | par | peak_kw | time_of_peak | rebound_peak_kw | rebound_delta_kw |
|---|---|---|---|---|---|---|---|
| population | SGSC base | base | 1.957 | 4.716 | 15:00 | 4.024 | 0.000 |
| population | MILP | base | 2.306 | 6.672 | 17:00 | 4.824 | 2.000 |
| population | noisy-MILP | base | 2.330 | 6.644 | 18:00 | 4.748 | 1.276 |
| population | LLM | base | 2.475 | 6.748 | 21:00 | 6.748 | 0.076 |
| population | SGSC base | cpp | 1.957 | 4.716 | 15:00 | 4.024 | 0.000 |
| population | MILP | cpp | 2.325 | 6.824 | 20:00 | 6.824 | 2.000 |
| population | noisy-MILP | cpp | 2.217 | 6.600 | 16:00 | 6.024 | 1.276 |
| population | LLM | cpp | 2.541 | 6.824 | 20:00 | 6.824 | 0.076 |

Figures: `overlay_population.png`, `overlay_<household>.png`.

## Discriminating scenario (single CPP spike + near washer deadline)

Spike 18:00-19:00 (severe); variants slack_remains: release 17:30, deadline 21:30; deadline_forces: release 17:30, deadline 20:00. Coherence is judged against the scenario's true deadline.

| agent | variant | coherent_share | count |
|---|---|---|---|
| llm | deadline_forces | 1.000 | 9 |
| llm | slack_remains | 1.000 | 9 |
| milp | deadline_forces | 1.000 | 3 |
| milp | slack_remains | 1.000 | 3 |
| noisy_milp | deadline_forces | 0.667 | 9 |
| noisy_milp | slack_remains | 0.667 | 9 |

| household_id | agent | seed | variant | washer_start | ran_in_spike | finished_on_time | expected | coherent |
|---|---|---|---|---|---|---|---|---|
| 8596070 | milp | 42 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 8596070 | noisy_milp | 42 | slack_remains | 20:00 | False | False | defer past the spike, finish on time | False |
| 8596070 | noisy_milp | 43 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 8596070 | noisy_milp | 44 | slack_remains | 19:00 | False | True | defer past the spike, finish on time | True |
| 8596070 | llm | 42 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 8596070 | llm | 43 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 8596070 | llm | 44 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 9911140 | milp | 42 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 9911140 | noisy_milp | 42 | slack_remains | 19:00 | False | True | defer past the spike, finish on time | True |
| 9911140 | noisy_milp | 43 | slack_remains | 20:00 | False | False | defer past the spike, finish on time | False |
| 9911140 | noisy_milp | 44 | slack_remains | 19:00 | False | True | defer past the spike, finish on time | True |
| 9911140 | llm | 42 | slack_remains | 19:00 | False | True | defer past the spike, finish on time | True |
| 9911140 | llm | 43 | slack_remains | 19:00 | False | True | defer past the spike, finish on time | True |
| 9911140 | llm | 44 | slack_remains | 19:00 | False | True | defer past the spike, finish on time | True |
| 8632083 | milp | 42 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 8632083 | noisy_milp | 42 | slack_remains | 19:00 | False | True | defer past the spike, finish on time | True |
| 8632083 | noisy_milp | 43 | slack_remains | 20:00 | False | False | defer past the spike, finish on time | False |
| 8632083 | noisy_milp | 44 | slack_remains | 19:00 | False | True | defer past the spike, finish on time | True |
| 8632083 | llm | 42 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 8632083 | llm | 43 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 8632083 | llm | 44 | slack_remains | 19:30 | False | True | defer past the spike, finish on time | True |
| 8596070 | milp | 42 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |
| 8596070 | noisy_milp | 42 | deadline_forces | 18:30 | True | False | run through the spike, finish on time | False |
| 8596070 | noisy_milp | 43 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 8596070 | noisy_milp | 44 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 8596070 | llm | 42 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |
| 8596070 | llm | 43 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |
| 8596070 | llm | 44 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |
| 9911140 | milp | 42 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |
| 9911140 | noisy_milp | 42 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 9911140 | noisy_milp | 43 | deadline_forces | 18:30 | True | False | run through the spike, finish on time | False |
| 9911140 | noisy_milp | 44 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 9911140 | llm | 42 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 9911140 | llm | 43 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 9911140 | llm | 44 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 8632083 | milp | 42 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |
| 8632083 | noisy_milp | 42 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 8632083 | noisy_milp | 43 | deadline_forces | 18:30 | True | False | run through the spike, finish on time | False |
| 8632083 | noisy_milp | 44 | deadline_forces | 17:30 | True | True | run through the spike, finish on time | True |
| 8632083 | llm | 42 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |
| 8632083 | llm | 43 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |
| 8632083 | llm | 44 | deadline_forces | 18:00 | True | True | run through the spike, finish on time | True |

## Multi-seed spread (seeds [42, 43, 44], households [8596070, 9911140, 8632083])

| agent | seed | elasticity_total | elasticity_flexible | elasticity_hvac | elasticity_washer |
|---|---|---|---|---|---|
| llm | 42 | -0.091 | -0.655 | -0.677 | -0.564 |
| llm | 43 | -0.091 | -0.655 | -0.677 | -0.564 |
| llm | 44 | -0.091 | -0.655 | -0.677 | -0.564 |
| noisy_milp | 42 | -0.110 | -0.651 | -0.725 | 0.000 |
| noisy_milp | 43 | -0.112 | -0.725 | -0.725 | nan |
| noisy_milp | 44 | -0.104 | -0.498 | -0.564 | 0.000 |

| agent | elasticity_total_min | elasticity_total_max | elasticity_total_std | elasticity_flexible_min | elasticity_flexible_max | elasticity_flexible_std |
|---|---|---|---|---|---|---|
| llm | -0.091 | -0.091 | 0.000 | -0.655 | -0.655 | 0.000 |
| noisy_milp | -0.112 | -0.104 | 0.004 | -0.725 | -0.498 | 0.116 |

## LLM call accounting

```
{
  "prompted_decisions": 2806,
  "live_calls_this_process": 0,
  "client_stats_by_seed": {},
  "dry_run": true,
  "wall_seconds": 8.5
}
```
