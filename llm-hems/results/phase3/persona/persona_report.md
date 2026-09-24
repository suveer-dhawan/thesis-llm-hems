# Persona assignment -- energy-burden archetypes

- average tariff rate: **$0.3024/kWh** (time-weighted over 2013, CPP excluded; method in use: `time_weighted`)
- income midpoints ($/wk): LOW=600, MED=1000, HI=1400 (SGSC data dictionary bands; open-ended bands closed at the adjacent $400 width)
- bucketing in use: **terciles**; fixed thresholds for comparison: [0.03, 0.06]

## Per-household burden

| customer_ID | cluster | income | annual kWh | est. cost $ | income $ | burden % | archetype (burden) | archetype (fixed) | archetype (income tier) | moved | band C | schedule | Group | elec-use |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9911140 | 5 | MED | 2,324 | 703 | 52,000 | 1.35 | A | A | B | yes | ±2.0 | evening | 2 | LOW |
| 9911204 | 4 | MED | 2,640 | 798 | 52,000 | 1.54 | A | A | B | yes | ±2.5 | evening | 3 | LOW |
| 9911250 | 6 | MED | 3,624 | 1,096 | 52,000 | 2.11 | A | A | B | yes | ±2.5 | evening | 3 | LOW |
| 8456189 | 3 | MED | 3,732 | 1,129 | 52,000 | 2.17 | A | A | B | yes | ±2.5 | evening | 2 | LOW |
| 8481835 | 7 | MED | 4,515 | 1,366 | 52,000 | 2.63 | A | A | B | yes | ±2.5 | evening | 2 | LOW |
| 9911254 | 10 | MED | 5,197 | 1,572 | 52,000 | 3.02 | A | B | B | yes | ±2.5 | evening | 2 | LOW |
| 11105177 | 8 | HI | 7,307 | 2,210 | 72,800 | 3.04 | A | B | A |  | ±2.5 | evening | 3 | LOW |
| 10278838 | 1 | MED | 5,468 | 1,654 | 52,000 | 3.18 | A | B | B | yes | ±2.5 | evening | 3 | LOW |
| 10556929 | 2 | HI | 7,833 | 2,369 | 72,800 | 3.25 | B | B | A | yes | ±2.5 | morning | 3 | MED |
| 8211127 | 3 | HI | 11,837 | 3,580 | 72,800 | 4.92 | B | B | A | yes | ±1.5 | evening | 3 | HI |
| 10444433 | 6 | HI | 12,459 | 3,768 | 72,800 | 5.18 | B | B | A | yes | ±2.5 | evening | 3 | HI |
| 10239118 | 9 | LOW | 5,885 | 1,780 | 31,200 | 5.70 | B | B | C | yes | ±2.5 | night | 3 | LOW |
| 10649333 | 4 | HI | 16,645 | 5,034 | 72,800 | 6.91 | B | C | A | yes | ±1.5 | evening | 3 | MED |
| 8395524 | 8 | MED | 12,615 | 3,815 | 52,000 | 7.34 | B | C | B |  | ±2.0 | evening | 2 | MED |
| 8463253 | 5 | LOW | 7,710 | 2,332 | 31,200 | 7.47 | B | C | C | yes | ±1.5 | evening | 3 | MED |
| 8201999 | 5 | HI | 17,993 | 5,442 | 72,800 | 7.47 | B | C | A | yes | ±2.0 | evening | 2 | HI |
| 9012382 | 9 | HI | 18,165 | 5,494 | 72,800 | 7.55 | C | C | A | yes | ±2.0 | night | 3 | LOW |
| 8596070 | 3 | LOW | 7,842 | 2,372 | 31,200 | 7.60 | C | C | C |  | ±2.0 | evening | 3 | LOW |
| 8208263 | 2 | LOW | 7,879 | 2,383 | 31,200 | 7.64 | C | C | C |  | ±2.5 | morning | 2 | LOW |
| 8638181 | 7 | HI | 18,642 | 5,638 | 72,800 | 7.74 | C | C | A | yes | ±1.5 | evening | 3 | MED |
| 8568499 | 1 | HI | 21,592 | 6,530 | 72,800 | 8.97 | C | C | A | yes | ±1.5 | evening | 3 | HI |
| 8619039 | 10 | LOW | 9,456 | 2,860 | 31,200 | 9.17 | C | C | C |  | ±2.0 | evening | 3 | MED |
| 8632083 | 7 | LOW | 11,322 | 3,424 | 31,200 | 10.98 | C | C | C |  | ±2.0 | evening | 3 | MED |
| 10042536 | 4 | LOW | 12,568 | 3,801 | 31,200 | 12.18 | C | C | C |  | ±2.0 | evening | 3 | LOW |

## Distribution

| archetype | burden (in use) | fixed thresholds | income tier |
|---|---|---|---|
| A | 8 | 5 | 9 |
| B | 8 | 7 | 8 |
| C | 8 | 12 | 7 |

Tercile burden ranges: A: 1.35%–3.18%; B: 3.25%–7.47%; C: 7.55%–12.18%

Lopsided (> 50% in one bucket)? burden-in-use: **False**; fixed thresholds: **False**

## Burden archetype vs income-tier archetype (17/24 move)

| archetype_income_tier \ archetype | A | B | C | total |
|---|---|---|---|---|
| A | 1 | 5 | 3 | 9 |
| B | 7 | 1 | 0 | 8 |
| C | 0 | 2 | 5 | 7 |
| total | 8 | 8 | 8 | 24 |

## Cross-check: web-portal engagement group (Group 3)

| archetype \ web_portal | Group3_portal | no_portal | total |
|---|---|---|---|
| A | 4 | 4 | 8 |
| B | 6 | 2 | 8 |
| C | 7 | 1 | 8 |
| total | 17 | 7 | 24 |

## Cross-check: assumed electricity-use tier

| archetype \ elec_use_tier | HI | LOW | MED | total |
|---|---|---|---|---|
| A | 0 | 8 | 0 | 8 |
| B | 3 | 1 | 4 | 8 |
| C | 1 | 4 | 3 | 8 |
| total | 4 | 13 | 7 | 24 |
