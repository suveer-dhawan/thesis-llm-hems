"""Phase 0 (re-draw): freeze `data/processed/{households,load_curves}.parquet`
from the committed 24-household manifest (`config/household_sample.csv`).

Responsibility: unlike `ingest_sgsc.py` (which *selects* a sample from raw
SGSC data via stratified sampling), this script *freezes* an already-decided
sample -- the 24 interview-complete households from the sample audit -- by
pulling their rows out of `data/raw/` (gitignored, ~946MB) and writing the
same two processed-parquet artifacts the rest of the project depends on.
`config/household_sample.csv` is the manifest of record; re-run this script
whenever it changes.

Usage: python experiments/build_household_sample.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
MANIFEST = Path("config/household_sample.csv")

DEMO_COLUMNS = [
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


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    ids = manifest["customer_ID"].astype(int).tolist()
    id_strs = [str(i) for i in ids]

    load = pd.read_csv(RAW / "interval load data" / "data_3_groups.csv")
    load["timestamp"] = pd.to_datetime(load["timestamp"])
    demo = pd.read_csv(RAW / "index files" / "demographic_lookup_3_groups.csv")

    missing = [i for i in id_strs if i not in load.columns]
    if missing:
        raise ValueError(f"manifest customer_IDs missing from raw load data: {missing}")

    data = load[id_strs]
    summary = pd.DataFrame(
        {
            "customer_ID": ids,
            "annual_kwh": data.sum().values,
            "zero_frac": (data == 0).mean().values,
            "peak_kw": (data.max() * 2).values,  # kWh/30min -> kW
        }
    )
    households = summary.merge(demo[DEMO_COLUMNS], on="customer_ID", how="left")
    households = households.merge(
        manifest[["customer_ID", "income_tier", "climate_zone", "dwelling", "has_children"]],
        on="customer_ID",
        how="left",
    ).sort_values(["cluster_all", "ASSRTD_DWELLING_TYPE_CD"]).reset_index(drop=True)

    curves = load[["timestamp"] + id_strs].melt(
        id_vars="timestamp", var_name="customer_ID", value_name="kwh"
    )
    curves["customer_ID"] = curves["customer_ID"].astype(int)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    households.to_parquet(PROCESSED / "households.parquet", index=False)
    curves.to_parquet(PROCESSED / "load_curves.parquet", index=False)

    print(f"wrote {PROCESSED / 'households.parquet'} ({len(households)} rows)")
    print(f"wrote {PROCESSED / 'load_curves.parquet'} ({len(curves)} rows)")


if __name__ == "__main__":
    main()
