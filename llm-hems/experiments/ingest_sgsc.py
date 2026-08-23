"""Phase 0: ingest SGSC -> select representative households -> freeze sample.

Responsibility (BUILD_SPEC.md §13 Phase 0): reads the raw SGSC files under
`data/raw/` (gitignored, ~946MB) and writes a small, version-controllable
working sample to `data/processed/` as parquet:

  households.parquet   -- one row per sampled household: id, cluster, dwelling
                           type, annual kWh, peak kW, a few demographic flags.
  load_curves.parquet  -- long-format 30-min load (kWh) for 2013 for just the
                           sampled households.

Selection method: stratify by (cluster_all x dwelling_type) -- the fully
populated, dwelling-agnostic 10-cluster labelling in
`demographic_lookup_3_groups.csv` crossed with `ASSRTD_DWELLING_TYPE_CD`
(NotUnit | Unit) -- drop households with >5% zero-valued intervals (likely
meter/logging gaps) or annual consumption outside the 1st-99th percentile,
then pick the household closest to each stratum's median annual kWh. This is
deterministic (no RNG) so the sample is reproducible from the raw data.

Run once against the raw data; the resulting parquet files are what the rest
of the project (and `data_profile.md`) depend on. Not part of the simulator
core -- data prep only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

ZERO_FRAC_MAX = 0.05
PCTILE_LO, PCTILE_HI = 0.01, 0.99


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    load = pd.read_csv(RAW / "interval load data" / "data_3_groups.csv")
    load["timestamp"] = pd.to_datetime(load["timestamp"])
    demo = pd.read_csv(RAW / "index files" / "demographic_lookup_3_groups.csv")
    return load, demo


def household_summary(load: pd.DataFrame) -> pd.DataFrame:
    hh_cols = load.columns[1:]
    data = load[hh_cols]
    return pd.DataFrame(
        {
            "customer_ID": [int(c) for c in hh_cols],
            "annual_kwh": data.sum().values,
            "zero_frac": (data == 0).mean().values,
            "peak_kw": (data.max() * 2).values,  # kWh/30min -> kW
        }
    )


def select_sample(summary: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    merged = summary.merge(
        demo[
            [
                "customer_ID",
                "cluster_all",
                "ASSRTD_DWELLING_TYPE_CD",
                "TRIAL_CUSTOMER_TYPE",
                "HAS_SOLAR",
                "HAS_AIRCON",
                "HAS_POOLPUMP",
                "NUM_OCCUPANTS",
                "HHOLD_INCOME_GROUP_CD",
            ]
        ],
        on="customer_ID",
        how="inner",
    )

    lo, hi = merged["annual_kwh"].quantile([PCTILE_LO, PCTILE_HI])
    eligible = merged[
        (merged["zero_frac"] < ZERO_FRAC_MAX)
        & (merged["annual_kwh"].between(lo, hi))
    ]

    picks = []
    for (cluster, dwelling), group in eligible.groupby(
        ["cluster_all", "ASSRTD_DWELLING_TYPE_CD"]
    ):
        if group.empty:
            continue
        med = group["annual_kwh"].median()
        closest = (group["annual_kwh"] - med).abs().idxmin()
        picks.append(group.loc[closest])

    sample = pd.DataFrame(picks).sort_values(["cluster_all", "ASSRTD_DWELLING_TYPE_CD"])
    return sample.reset_index(drop=True)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    load, demo = load_raw()
    summary = household_summary(load)
    sample = select_sample(summary, demo)

    sample_ids = sample["customer_ID"].astype(str).tolist()
    curves = load[["timestamp"] + sample_ids].melt(
        id_vars="timestamp", var_name="customer_ID", value_name="kwh"
    )
    curves["customer_ID"] = curves["customer_ID"].astype(int)

    sample.to_parquet(PROCESSED / "households.parquet", index=False)
    curves.to_parquet(PROCESSED / "load_curves.parquet", index=False)

    print(f"selected {len(sample)} households across "
          f"{sample['cluster_all'].nunique()} clusters "
          f"x {sample['ASSRTD_DWELLING_TYPE_CD'].nunique()} dwelling types")
    print(f"wrote {PROCESSED / 'households.parquet'} "
          f"({len(sample)} rows)")
    print(f"wrote {PROCESSED / 'load_curves.parquet'} "
          f"({len(curves)} rows)")


if __name__ == "__main__":
    main()
