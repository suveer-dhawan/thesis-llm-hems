"""Phase 3 Part A: persona assignment (cost_sensitivity = energy burden).

Builds one persona per household in `config/household_sample.csv` from
`config/persona.yaml` (see `hems.environment.persona`) and writes:

- `config/persona_assignment.csv` -- the committed assignment the simulator
  reads (cost_sensitivity, archetype, band, washer window, plus the old
  income-tier archetype for comparison and two SGSC cross-check columns).
- `results/phase3/persona/persona_report.{md,json}` -- per-household burden
  table, A/B/C distribution, burden-vs-income-tier movement, and the
  cross-checks against the SGSC web-portal engagement group (Group 3) and the
  assumed electricity-use tier. Flags a lopsided distribution instead of
  silently switching anchors.

Needs `data/raw/index files/demographic_lookup_3_groups.csv` for the two
cross-check columns (Group, ASSRTD_ELECTRICITY_USE_GRP_CD).

Usage: python experiments/build_personas.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hems.environment.persona import (
    ASSIGNMENT_COLUMNS,
    DEFAULT_ASSIGNMENT_PATH,
    DEFAULT_PERSONA_CONFIG,
    build_persona_table,
    load_persona_config,
    load_weighted_average_rates,
    time_weighted_average_rate,
)
from hems.environment.tariff import Tariff

MANIFEST = Path("config/household_sample.csv")
HOUSEHOLDS = Path("data/processed/households.parquet")
LOAD_CURVES = Path("data/processed/load_curves.parquet")
DEMOGRAPHICS = Path("data/raw/index files/demographic_lookup_3_groups.csv")
REPORT_DIR = Path("results/phase3/persona")

# A bucket holding more than this share of households is flagged as lopsided.
LOPSIDED_SHARE = 0.5


def _crosstab_md(df: pd.DataFrame, row: str, col: str) -> str:
    ct = pd.crosstab(df[row], df[col], margins=True, margins_name="total")
    header = f"| {row} \\ {col} | " + " | ".join(str(c) for c in ct.columns) + " |"
    sep = "|" + "---|" * (len(ct.columns) + 1)
    lines = [header, sep]
    for idx, r in ct.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_PERSONA_CONFIG)
    parser.add_argument("--out", default=DEFAULT_ASSIGNMENT_PATH)
    args = parser.parse_args()

    cfg = load_persona_config(args.config)
    cs = cfg["cost_sensitivity"]
    manifest = pd.read_csv(MANIFEST)
    households = pd.read_parquet(HOUSEHOLDS)
    annual_kwh = households.set_index("customer_ID")["annual_kwh"]
    tariff = Tariff.from_yaml("config/tariff.yaml")

    tw_rate = time_weighted_average_rate(tariff, int(cs["rate_year"]))
    if cs["average_rate_method"] == "load_weighted":
        avg_rate = load_weighted_average_rates(
            tariff, pd.read_parquet(LOAD_CURVES), manifest["customer_ID"].tolist()
        )
    else:
        avg_rate = tw_rate

    table = build_persona_table(manifest, annual_kwh, avg_rate, cfg)

    demo = pd.read_csv(DEMOGRAPHICS, low_memory=False)[
        ["customer_ID", "Group", "ASSRTD_ELECTRICITY_USE_GRP_CD"]
    ].rename(
        columns={"Group": "engagement_group", "ASSRTD_ELECTRICITY_USE_GRP_CD": "elec_use_tier"}
    )
    table = table.merge(demo, on="customer_ID", how="left")
    table["web_portal"] = (table["engagement_group"] == 3).map(
        {True: "Group3_portal", False: "no_portal"}
    )

    out_cols = ASSIGNMENT_COLUMNS + ["archetype_fixed", "engagement_group", "elec_use_tier"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table[out_cols].to_csv(args.out, index=False, float_format="%.6g")

    # ---- report -------------------------------------------------------------
    labels = list(cs["archetype_labels"])
    dist = table["archetype"].value_counts().reindex(labels, fill_value=0)
    dist_fixed = table["archetype_fixed"].value_counts().reindex(labels, fill_value=0)
    moved = table[table["archetype"] != table["archetype_income_tier"]]
    lopsided_active = bool((dist / len(table)).max() > LOPSIDED_SHARE)
    lopsided_fixed = bool((dist_fixed / len(table)).max() > LOPSIDED_SHARE)
    tercile_edges = (
        table.sort_values("burden").groupby("archetype_terciles")["burden"].agg(["min", "max"])
    )

    rows = table.sort_values("burden")
    md = [
        "# Persona assignment -- energy-burden archetypes",
        "",
        f"- average tariff rate: **${tw_rate:.4f}/kWh** (time-weighted over {cs['rate_year']}, "
        f"CPP excluded; method in use: `{cs['average_rate_method']}`)",
        "- income midpoints ($/wk): "
        + ", ".join(f"{k}={v}" for k, v in cs["income_weekly_midpoint_aud"].items())
        + " (SGSC data dictionary bands; open-ended bands closed at the adjacent $400 width)",
        f"- bucketing in use: **{cs['bucketing']}**; fixed thresholds for comparison: "
        f"{cs['fixed_thresholds']}",
        "",
        "## Per-household burden",
        "",
        "| customer_ID | cluster | income | annual kWh | est. cost $ | income $ | burden % | "
        "archetype (burden) | archetype (fixed) | archetype (income tier) | moved | band C | "
        "schedule | Group | elec-use |",
        "|" + "---|" * 15,
    ]
    for _, r in rows.iterrows():
        md.append(
            f"| {r.customer_ID} | {r.cluster} | {r.income_tier} | {r.annual_kwh:,.0f} | "
            f"{r.est_annual_cost:,.0f} | {r.income_midpoint:,.0f} | {100 * r.burden:.2f} | "
            f"{r.archetype} | {r.archetype_fixed} | {r.archetype_income_tier} | "
            f"{'yes' if r.archetype != r.archetype_income_tier else ''} | "
            f"±{r.tolerance_band_c} | {r.schedule_profile} | {r.engagement_group} | "
            f"{r.elec_use_tier} |"
        )
    md += [
        "",
        "## Distribution",
        "",
        "| archetype | burden (in use) | fixed thresholds | income tier |",
        "|---|---|---|---|",
    ]
    dist_income = table["archetype_income_tier"].value_counts().reindex(labels, fill_value=0)
    for lab in labels:
        md.append(f"| {lab} | {dist[lab]} | {dist_fixed[lab]} | {dist_income[lab]} |")
    md += [
        "",
        "Tercile burden ranges: "
        + "; ".join(
            f"{a}: {100 * r['min']:.2f}%–{100 * r['max']:.2f}%" for a, r in tercile_edges.iterrows()
        ),
        "",
        f"Lopsided (> {LOPSIDED_SHARE:.0%} in one bucket)? burden-in-use: **{lopsided_active}**; "
        f"fixed thresholds: **{lopsided_fixed}**",
        "",
        f"## Burden archetype vs income-tier archetype ({len(moved)}/{len(table)} move)",
        "",
        _crosstab_md(table, "archetype_income_tier", "archetype"),
        "",
        "## Cross-check: web-portal engagement group (Group 3)",
        "",
        _crosstab_md(table, "archetype", "web_portal"),
        "",
        "## Cross-check: assumed electricity-use tier",
        "",
        _crosstab_md(table, "archetype", "elec_use_tier"),
        "",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "persona_report.md").write_text("\n".join(md))
    (REPORT_DIR / "persona_report.json").write_text(
        json.dumps(
            {
                "time_weighted_avg_rate": tw_rate,
                "bucketing": cs["bucketing"],
                "distribution": dist.to_dict(),
                "distribution_fixed": dist_fixed.to_dict(),
                "distribution_income_tier": dist_income.to_dict(),
                "n_moved_vs_income_tier": int(len(moved)),
                "lopsided_in_use": lopsided_active,
                "lopsided_fixed": lopsided_fixed,
                "tercile_ranges": {
                    a: [float(r["min"]), float(r["max"])] for a, r in tercile_edges.iterrows()
                },
            },
            indent=2,
        )
    )
    print("\n".join(md))
    print(f"\nwrote {args.out} and {REPORT_DIR}/persona_report.md")


if __name__ == "__main__":
    main()
