# SGSC Data Profile (Phase 0)

Summarises the raw Ausgrid Smart Grid, Smart City (SGSC) dataset in
`llm-hems/data/raw/` (gitignored, ~946MB), the working sample extracted from
it, and the resulting load-shape observations. Produced by
[`llm-hems/experiments/ingest_sgsc.py`](llm-hems/experiments/ingest_sgsc.py)
and [`llm-hems/experiments/profile_sgsc.py`](llm-hems/experiments/profile_sgsc.py).
No simulator, appliance, or MILP logic is implemented here — this is data
characterisation only.

## 1. Raw files

| File | Size | Format | Shape |
|---|---|---|---|
| `interval load data/data_3_groups.csv` | 990 MB | wide CSV | 17,520 rows × 6,032 cols |
| `index files/demographic_lookup_3_groups.csv` | 1.5 MB | CSV | 6,031 rows × 68 cols |
| `index files/cluster_index.csv` | 192 KB | CSV | 6,034 rows × 7 cols |
| `index files/Column Mappings.csv` | 164 KB | CSV | 659 rows × 13 cols — data dictionary only |

**`data_3_groups.csv`** — the interval load data. One `timestamp` column plus
one column per household, named by its `customer_ID` (6,031 households).
Rows run `2013-01-01 00:00:00` → `2013-12-31 23:30:00` at a constant **30-min
step** (17,520 = 365 days × 48 steps) — **confirmed: 30-min resolution, full
calendar year 2013, no other years present.** Cell values are **kWh consumed
in that half-hour** (not instantaneous kW); multiply by 2 for average kW over
the interval. No column headers or units are given in-file — units were
inferred from magnitude (median household ≈ 8,267 kWh/yr ≈ 22.6 kWh/day,
consistent with real AU household electricity use) and cross-checked against
`eDaily` (a precomputed average-daily-kWh field already present in the
demographic file, see below).

**`demographic_lookup_3_groups.csv`** — one row per household (`customer_ID`),
68 columns covering trial/control group, tariff product, location, dwelling
type, occupancy, appliance ownership (aircon, solar, pool pump, gas, etc.),
income band, and the study's own cluster labels (`cluster_all`,
`cluster_house`, `cluster_unit`) plus `eDaily` (avg daily kWh) and `peakTime`
(hour of daily peak) — both apparently precomputed by the original SGSC
study.

**`cluster_index.csv`** — an alternate/parallel view of cluster membership,
columns `customer_ID, 3a, 3h, 3u, ca, ch, cu`. `3a` is numerically identical
to `demographic_lookup`'s `cluster_all` (correlation 1.0, verified) — it's
the same 10-cluster, dwelling-agnostic labelling. `3h`/`3u` are the same
clustering restricted to house/unit subsets (`cluster_house`/`cluster_unit`
in the other file). `ca/ch/cu` are a second, sparser clustering scheme
(only 526–1,786 of 6,031 households labelled, weak correlation with `3a`/`ca`
≈ 0.12) whose provenance isn't documented in `Column Mappings.csv`; **not
used** for sampling below — `cluster_all` (10 clusters, fully populated) is
the one we stratify on.

**`Column Mappings.csv`** — Ausgrid's own data dictionary for the
demographic table (subject area, description, source schema/column, datatype)
— useful reference, not consumed programmatically.

All three data files key on the same `customer_ID` / `10-digit numeric
household id`, and the 6,031 IDs are **identical across all three files** (no
orphans either direction) — confirmed by exact set intersection.

## 2. Structure

**Household identification:** a 10-digit `customer_ID`, consistent across
the load-data column headers, `demographic_lookup_3_groups.csv`, and
`cluster_index.csv`.

**Clustering:** `cluster_all` (aka `3a`) assigns every one of the 6,031
households to one of **10 clusters** (labelled 1–10, sizes 323–951
households) via a dwelling-agnostic load-shape clustering — this is a
*label per household*, not a set of separate representative-profile rows.
`cluster_house`/`cluster_unit` are the same idea run separately within each
dwelling type (a household gets a nonzero label in exactly one of the two,
0 in the other). No literature/methodology reference for the algorithm is
included in the raw files.

**Dwelling type:** the interview-stated `DWELLING_TYPE_CD` field is mostly
missing (4,650/6,031 null; only 1,381 populated: SeparateHouse / Unit /
SemiDetached). The **assumed/imputed** `ASSRTD_DWELLING_TYPE_CD` field is
**fully populated** (`NotUnit`: 3,950, `Unit`: 2,081) and matches the
`cluster_house`/`cluster_unit` split exactly — this is the dwelling-type
field to rely on.

**Missing / filled values:**
- **Interval load data: zero missing values** across the entire 17,520 × 6,031
  matrix (verified by a full-file scan) — the dataset has already been
  gap-filled upstream by Ausgrid/UNSW; we cannot distinguish originally
  metered intervals from imputed ones from this file alone. All values are
  ≥ 0 (no negative/net-export readings; min 0.0, max 26.3 kWh/half-hour).
  51,489 / 105.7M cells (0.05%) are exactly `0.0`; one household
  (0.02% of the sample) has an unusually high zero-fraction (88%,
  likely a metering/logging gap) and was excluded from sampling (see below).
- **Demographic table:** most of the *optional interview* fields
  (`NUM_OCCUPANTS`, `HHOLD_INCOME_GROUP_CD`, etc.) are sparse — only ~1,380
  of 6,031 households (23%) have them populated, since these come from an
  opt-in lifestyle-audit interview, not the smart-meter trial itself.
  Structural fields (`ASSRTD_DWELLING_TYPE_CD`, `HAS_SOLAR`, `HAS_AIRCON`,
  `HAS_POOLPUMP`, `TRIAL_CUSTOMER_TYPE`, `cluster_all`) are fully populated.
- **Trial structure:** 1,786 households are `Control` (no trial tariff),
  4,245 are `Network` (trial participants, further split by `Group` into two
  network sub-cohorts). This matters if real historical tariffs are ever
  cross-referenced against this data — our simulator's tariff is synthetic
  (`config/tariff.yaml`, all `CONFIRM`), not the households' actual trial
  tariff, so this split doesn't currently constrain sampling.

## 3. Working sample

**Superseded (pre-Phase-3 fix):** the original working sample was 20
households picked by `ingest_sgsc.py` (fully deterministic, no RNG): for
each of the 20 `(cluster_all × ASSRTD_DWELLING_TYPE_CD)` strata (10 clusters
× 2 dwelling types), household is excluded if it has >5% zero-valued
intervals or annual consumption outside the population's 1st–99th
percentile, then the household **closest to that stratum's median annual
kWh** is picked. That stratified sample has been **replaced** by a
24-household sample drawn from a separate interview-completeness audit (one
`customer_ID` per stratum among households with a complete opt-in
lifestyle-audit interview, giving richer per-household fields — income
tier, climate zone, occupant count — than the demographic table's
sparsely-populated interview columns alone would allow; see §2 "Sparse
interview fields"). Demo customer `8596070` (cluster 3, `NotUnit`) is
common to both samples.

**Manifest of record:** [`llm-hems/config/household_sample.csv`](llm-hems/config/household_sample.csv)
(24 rows: `customer_ID`, cluster, income tier, dwelling type, climate zone,
aircon type, occupant count, has-children flag, annual kWh, peak hour).
[`llm-hems/experiments/build_household_sample.py`](llm-hems/experiments/build_household_sample.py)
freezes it against `data/raw/` into the two processed-parquet artifacts the
rest of the project depends on:
- [`llm-hems/data/processed/households.parquet`](llm-hems/data/processed/households.parquet)
  (24 rows: id, cluster, dwelling type, annual kWh, peak kW, manifest fields,
  and a few demographic flags)
- [`llm-hems/data/processed/load_curves.parquet`](llm-hems/data/processed/load_curves.parquet)
  (420,480 rows = 24 households × 17,520 steps, long format `customer_ID,
  timestamp, kwh`)

| customer_ID | cluster | dwelling | annual kWh | peak kW | trial group | solar | aircon | pool pump |
|---|---|---|---|---|---|---|---|---|
| 8568499 | 1 | NotUnit | 21,592 | 12.2 | Network | N | Y | N |
| 10278838 | 1 | Unit | 5,468 | 7.2 | Network | N | N | N |
| 8208263 | 2 | NotUnit | 7,879 | 9.4 | Network | N | N | N |
| 10556929 | 2 | Unit | 7,833 | 6.8 | Network | N | N | N |
| 8211127 | 3 | NotUnit | 11,837 | 17.4 | Network | N | Y | N |
| 8596070 | 3 | NotUnit | 7,842 | 7.8 | Network | N | Y | N |
| 8456189 | 3 | Unit | 3,732 | 5.5 | Network | N | N | N |
| 10649333 | 4 | NotUnit | 16,645 | 17.5 | Network | N | Y | N |
| 10042536 | 4 | NotUnit | 12,568 | 12.6 | Network | N | Y | N |
| 9911204 | 4 | Unit | 2,640 | 5.4 | Network | N | N | N |
| 8201999 | 5 | NotUnit | 17,993 | 14.2 | Network | N | Y | Y |
| 8463253 | 5 | NotUnit | 7,710 | 11.5 | Network | N | Y | N |
| 9911140 | 5 | Unit | 2,324 | 4.8 | Network | N | Y | N |
| 10444433 | 6 | NotUnit | 12,459 | 10.7 | Network | N | N | N |
| 9911250 | 6 | Unit | 3,624 | 6.5 | Network | N | N | N |
| 8638181 | 7 | NotUnit | 18,642 | 16.4 | Network | N | Y | N |
| 8632083 | 7 | NotUnit | 11,322 | 11.2 | Network | N | Y | N |
| 8481835 | 7 | Unit | 4,515 | 10.8 | Network | N | N | N |
| 8395524 | 8 | NotUnit | 12,615 | 11.2 | Network | N | Y | N |
| 11105177 | 8 | Unit | 7,307 | 9.8 | Network | N | N | N |
| 9012382 | 9 | NotUnit | 18,165 | 13.3 | Network | N | Y | Y |
| 10239118 | 9 | Unit | 5,885 | 9.2 | Network | N | N | N |
| 8619039 | 10 | NotUnit | 9,456 | 10.0 | Network | N | Y | N |
| 9911254 | 10 | Unit | 5,197 | 7.8 | Network | N | N | N |

Every one of the 24 households is a `Network` trial participant with
`HAS_SOLAR=N` (an artifact of which interview-complete households existed
per stratum, not a deliberate filter); clusters 3, 4, 5, and 7 contribute
three households each (HI/MED/LOW income tier) rather than one house + one
unit, since the audit's selection axis is income tier crossed with dwelling
type, not a strict one-house-one-unit-per-cluster rule. Houses (`NotUnit`)
still consistently show higher annual consumption than units within the
same cluster.

The full raw dataset (`data/raw/`, ~946MB) stays gitignored per the project's
`.gitignore`; only the processed sample is committed. Note: the
`.gitignore` in `llm-hems/` previously excluded *all* of `data/` and
`results/`, which would have also hidden this sample and the plots below —
it's been updated to `data/*` / `!data/processed/**` and `results/*` /
`!results/data_profile/**` so raw data stays out but the processed sample and
profile figures are trackable.

## 4. Load-shape observations

![Representative daily load curves](llm-hems/results/data_profile/representative_daily_curves.png)

Six sample households (clusters 1, 5, 9; one house + one unit each) on a
summer weekday (16 Jan) and a winter weekday (17 Jul). Individual-household
traces are spiky and dominated by a small number of appliance-driven peaks
(likely HVAC/hot-water/pool-pump cycling) rather than smooth curves — exactly
the kind of discrete on/off structure the appliance models in `§7` of the
build spec are meant to represent. Units (dashed) generally show lower,
flatter baseload than houses (solid) in the same cluster, consistent with
smaller dwellings / fewer major appliances.

*(This plot predates the §3 sample re-draw and shows six households from
the original 20-household stratified sample, not the current 24; the
qualitative shape observations still hold, but the specific `customer_ID`s
plotted are no longer in `config/household_sample.csv`. Not regenerated
here — droppable if a later phase wants a refreshed version against the
24-household sample.)*

![Average weekday load shape by cluster](llm-hems/results/data_profile/avg_weekday_shape_by_cluster.png)

Mean weekday (Mon–Fri) profile per cluster, averaged over **all** households
in that cluster (not just the 24-household sample) across the full year.
The 10 clusters separate into two broad shapes:

- **Evening-peak clusters (3, 4, 5, 7):** low overnight/daytime baseload,
  sharp rise from ~16:00 to a peak around 18:00–20:00 (~2.0–2.2 kW) — the
  "typical occupied household, home in the evening" shape, and the largest
  clusters by membership (761–951 households each).
- **Flatter / non-evening-peaked clusters (1, 6, 8, 9):** higher overnight
  base (1.0–1.75 kW), a milder or earlier peak, and in cluster 9's case a
  peak that is *higher overnight than in the evening* — consistent with
  always-on loads (electric storage hot water, pool pumps) or occupants home
  during the day.

All clusters show a small secondary bump around 07:00–08:00 (morning
routine), most visible in clusters 4, 5, 7.

## 5. Data-quality caveats

- **Pre-cleaned data:** zero missing values in the interval file means gaps
  were already filled before this data was published; we have no way to flag
  which intervals are original meter reads vs. imputed from this file alone.
- **Units inferred, not labelled:** the raw CSV has no header row describing
  units; kWh-per-half-hour was inferred from magnitude and cross-checked
  against the `eDaily` field, not confirmed from an Ausgrid data dictionary
  entry for this specific file (`Column Mappings.csv` documents the
  demographic table, not the interval-data file).
- **Sparse interview fields:** occupancy count and income band are only
  available for ~23% of households — not used in this sample's selection or
  characterisation beyond what's shown in §3.
- **Two clustering schemes coexist** (`3a/3h/3u` vs `ca/ch/cu`) with no
  documented difference in the raw files; `cluster_all` was chosen for being
  fully populated and having a documented match to `cluster_house`/
  `cluster_unit`, not because its methodology was confirmed.
- **One excluded household** (customer_ID redacted from the sample; 88%
  zero-valued intervals) suggests at least isolated metering gaps exist
  despite the file showing no explicit `NaN`s — worth another pass if a
  larger population sample (Phase 4) is built later.
- **Trial/control and real tariff exposure** is present in the demographic
  table (`TRIAL_CUSTOMER_TYPE`, `TARIFF_PRODUCT_CD`) but not used here; our
  simulator applies its own synthetic tariff (`config/tariff.yaml`), so the
  households' real historical billing tariff is out of scope unless a future
  phase wants to validate against it.
