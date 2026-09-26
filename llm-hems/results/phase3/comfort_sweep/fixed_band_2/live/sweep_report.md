### Comfort-orientation sweep -- LIVE

HVAC only, seed 42, 2013-01-18, CPP 17:00-20:00 (0.55 -> 1.20 $/kWh), LLM window 14:00-21:00 (thermostat outside), method `base`. Households: 8211127 (comfort_prioritising, B), 8568499 (comfort_prioritising, C), 10278838 (comfort_flexible, A), 10444433 (comfort_flexible, B)

Bands: fixed at +/-2.0C for every household (orientation sentence unchanged).

Graded archetype factors {'A': 1.3, 'B': 1.0, 'C': 0.7}. Elasticity is pooled per group (sum kWh, then ratio); more negative = more elastic. Ordering flags need strict separation end to end; NaN (no baseline HVAC in the window) -> False.

#### (a1) flexible-layer (HVAC) elasticity vs global_scale

| global_scale | pooled | prio | bal | flex | flex>bal>prio | arch A | arch B | arch C | C>B>A | HVAC kWh base->cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.000 | -0.212 | 0.000 | nan | -0.423 | False | -0.564 | 0.000 | 0.000 | False | 8->6 |

#### (a2) whole-household elasticity vs global_scale

| global_scale | pooled | prio | bal | flex | flex>bal>prio | arch A | arch B | arch C | C>B>A |
|---|---|---|---|---|---|---|---|---|---|
| 1.000 | -0.096 | 0.000 | nan | -0.226 | False | -0.435 | 0.000 | 0.000 | False |

#### (c) whole-household elasticity vs global_scale (Conway (-0.47, -0.026), target -0.25)

| global_scale | e_total (pooled) | e_flexible (pooled) | in_conway |
|---|---|---|---|
| 1.000 | -0.096 | -0.212 | True |

chosen scale: 1.0 (in range)

#### raw LLM WTP on the trajectories (distinct prompts), by C over own ceiling

| over_ceiling_bin | comfort_prioritising mean_wtp | comfort_prioritising n | comfort_flexible mean_wtp | comfort_flexible n |
|---|---|---|---|---|
| -1.0..-0.5 | 0.100 | 2 | 0.050 | 1 |
| -0.5..0 | 0.150 | 7 | 0.050 | 3 |
| 0..0.5 | 1.106 | 8 | 0.231 | 8 |
| 0.5..1.0 | 2.167 | 3 | 0.570 | 5 |
| > 1.0 | nan | 0 | 0.977 | 13 |
| < -1.0 | nan | 0 | nan | 0 |

#### (d) call accounting

```
{
  "prompted_decisions": 112,
  "distinct_prompts": 50,
  "live_calls": 50,
  "cache_hits": 62,
  "cache_hit_rate": 0.554,
  "misread_calls": 0,
  "wall_seconds": 550.1,
  "dry_run": false
}
```
