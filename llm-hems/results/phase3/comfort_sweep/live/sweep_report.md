### Comfort-orientation sweep -- LIVE

HVAC only, seed 42, 2013-01-18, CPP 17:00-20:00 (0.55 -> 1.20 $/kWh), LLM window 14:00-21:00 (thermostat outside), method `base`. Households: 8211127 (comfort_prioritising, B), 8568499 (comfort_prioritising, C), 9911140 (balanced, A), 8596070 (balanced, C), 10278838 (comfort_flexible, A), 10444433 (comfort_flexible, B)

Graded archetype factors {'A': 1.3, 'B': 1.0, 'C': 0.7}. Elasticity is pooled per group (sum kWh, then ratio); more negative = more elastic. Ordering flags need strict separation end to end; NaN (no baseline HVAC in the window) -> False.

#### (a1) flexible-layer (HVAC) elasticity vs global_scale

| global_scale | pooled | prio | bal | flex | flex>bal>prio | arch A | arch B | arch C | C>B>A | HVAC kWh base->cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.300 | -0.592 | -0.423 | -0.846 | -0.846 | True | -0.846 | -0.423 | -0.564 | False | 10->3 |
| 0.500 | -0.494 | -0.169 | -0.564 | -0.846 | True | -0.677 | -0.282 | -0.423 | False | 12->5 |
| 0.700 | -0.484 | 0.000 | -0.423 | -0.846 | True | -0.677 | -0.508 | -0.212 | False | 14->6 |
| 1.000 | -0.130 | 0.000 | -0.212 | -0.169 | False | -0.212 | 0.000 | -0.212 | False | 13->11 |

#### (a2) whole-household elasticity vs global_scale

| global_scale | pooled | prio | bal | flex | flex>bal>prio | arch A | arch B | arch C | C>B>A |
|---|---|---|---|---|---|---|---|---|---|
| 0.300 | -0.178 | -0.209 | -0.108 | -0.309 | False | -0.572 | -0.188 | -0.085 | False |
| 0.500 | -0.168 | -0.076 | -0.101 | -0.453 | True | -0.526 | -0.106 | -0.081 | False |
| 0.700 | -0.182 | 0.000 | -0.096 | -0.536 | True | -0.526 | -0.254 | -0.041 | False |
| 1.000 | -0.047 | 0.000 | -0.048 | -0.100 | True | -0.156 | 0.000 | -0.041 | False |

#### (b) decomposition at global_scale 0.7: uniform (1.0/1.0/1.0) vs graded factors

| condition | comfort_prioritising e_flex | balanced e_flex | comfort_flexible e_flex | comfort spread (max-min) | arch A e_flex | arch B e_flex | arch C e_flex | C>B>A elastic | e_total (pooled) |
|---|---|---|---|---|---|---|---|---|---|
| uniform | 0.000 | -0.423 | -0.846 | 0.846 | -0.677 | -0.508 | -0.212 | False | -0.182 |
| graded | 0.000 | -0.423 | -0.846 | 0.846 | -0.677 | -0.508 | -0.212 | False | -0.182 |

#### (c) whole-household elasticity vs global_scale (Conway (-0.47, -0.026), target -0.25)

| global_scale | e_total (pooled) | e_flexible (pooled) | in_conway |
|---|---|---|---|
| 0.300 | -0.178 | -0.592 | True |
| 0.500 | -0.168 | -0.494 | True |
| 0.700 | -0.182 | -0.484 | True |
| 1.000 | -0.047 | -0.130 | True |

chosen scale: 0.7 (in range)

#### raw LLM WTP on the trajectories (distinct prompts), by C over own ceiling

| over_ceiling_bin | comfort_prioritising mean_wtp | comfort_prioritising n | balanced mean_wtp | balanced n | comfort_flexible mean_wtp | comfort_flexible n |
|---|---|---|---|---|---|---|
| -1.0..-0.5 | 0.088 | 4 | 0.050 | 2 | nan | 0 |
| -0.5..0 | 0.090 | 10 | 0.050 | 7 | 0.050 | 9 |
| 0..0.5 | 1.942 | 13 | 0.362 | 12 | 0.286 | 7 |
| 0.5..1.0 | 2.214 | 7 | 0.761 | 14 | 0.659 | 11 |
| > 1.0 | 4.618 | 17 | 1.342 | 46 | 0.921 | 36 |
| < -1.0 | nan | 0 | nan | 0 | nan | 0 |

#### (d) call accounting

```
{
  "prompted_decisions": 840,
  "distinct_prompts": 195,
  "live_calls": 195,
  "cache_hits": 645,
  "cache_hit_rate": 0.768,
  "misread_calls": 0,
  "wall_seconds": 2332.2,
  "dry_run": false
}
```
