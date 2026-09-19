"""Analysis pass for probe_wtp_price_v2.py's raw CSV (+ a v1 cross-run
comparison). Not part of the production pipeline -- reporting only.

Usage:
  python experiments/analyze_probe_v2.py <v2_csv_path> [<v1_csv_path>]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

REVERSAL_TOLERANCE = 0.02


def linear_slope(rates, wtps) -> float:
    if len(set(rates)) < 2:
        return float("nan")
    slope, _ = np.polyfit(rates, wtps, 1)
    return float(slope)


def cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["appliance", "persona", "rate"])["wtp"].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    g["cv"] = g.apply(lambda r: r["std"] / r["mean"] if r["mean"] > 1e-9 else float("nan"), axis=1)
    return g


def main() -> None:
    v2_path = sys.argv[1]
    v1_path = sys.argv[2] if len(sys.argv) > 2 else None

    v2 = pd.read_csv(v2_path)
    main_grid = v2[v2["grid"] == "main_flag_off"]
    flag_grid = v2[v2["grid"] == "flag_on"]

    print("=" * 80)
    print("Q1 -- MAIN GRID (flag OFF, anchored framing)")
    print("=" * 80)
    for appliance in ("hvac", "washer"):
        for persona in ("comfort_first", "neutral"):
            sub = main_grid[(main_grid["appliance"] == appliance) & (main_grid["persona"] == persona)]
            if sub.empty:
                continue
            print(f"\n### {appliance} x {persona}\n")
            print("| price ($/kWh) | mean WTP | std WTP | CV |")
            print("|---|---|---|---|")
            stats = cell_stats(sub)
            means = []
            for _, row in stats.sort_values("rate").iterrows():
                cv_str = f"{row['cv']:.2f}" if row["cv"] == row["cv"] else "n/a"
                print(f"| {row['rate']} | {row['mean']:.3f} | {row['std']:.3f} | {cv_str} |")
                means.append((row["rate"], row["mean"]))
            slope = linear_slope(sub["rate"], sub["wtp"])
            wtp_range = max(m for _, m in means) - min(m for _, m in means)
            reversals = [
                (means[i][0], means[i + 1][0])
                for i in range(len(means) - 1)
                if means[i + 1][1] > means[i][1] + REVERSAL_TOLERANCE
            ]
            mean_cv = stats["cv"].dropna().mean()
            correct_sign = slope <= 0
            print(f"\nslope: {slope:.4f}  |  WTP range: ${wtp_range:.3f}  |  mean CV: {mean_cv:.3f}"
                  if mean_cv == mean_cv else f"\nslope: {slope:.4f}  |  WTP range: ${wtp_range:.3f}  |  mean CV: n/a")
            print(f"correct sign (non-positive slope): {correct_sign}  |  reversals: {reversals if reversals else 'none'}")

    if v1_path:
        print("\n" + "=" * 80)
        print("FRAMING EFFECT -- v1 (unanchored, flag off) vs v2 main grid (anchored, flag off)")
        print("=" * 80)
        v1 = pd.read_csv(v1_path)
        compare_prices = [0.22, 0.28, 0.55]
        for appliance in ("hvac", "washer"):
            for persona in ("comfort_first", "neutral"):
                print(f"\n### {appliance} x {persona}\n")
                print("| price | v1 mean (unanchored) | v2 mean (anchored) | delta |")
                print("|---|---|---|---|")
                for rate in compare_prices:
                    v1_cell = v1[(v1["appliance"] == appliance) & (v1["persona"] == persona)
                                 & (v1["price_label"].map({"off_peak": 0.22, "shoulder": 0.28, "peak": 0.55}.get) == rate)]
                    v2_cell = main_grid[(main_grid["appliance"] == appliance) & (main_grid["persona"] == persona)
                                        & (main_grid["rate"] == rate)]
                    v1_mean = v1_cell["wtp"].mean() if not v1_cell.empty else float("nan")
                    v2_mean = v2_cell["wtp"].mean() if not v2_cell.empty else float("nan")
                    delta = v2_mean - v1_mean
                    print(f"| {rate} | {v1_mean:.3f} | {v2_mean:.3f} | {delta:+.3f} |")

    print("\n" + "=" * 80)
    print("Q2 -- FLAG EFFECT ISOLATED (same price, flag off vs flag on)")
    print("=" * 80)
    for appliance in ("hvac", "washer"):
        for persona in ("comfort_first", "neutral"):
            print(f"\n### {appliance} x {persona}\n")
            print("| price | flag-OFF mean | flag-ON mean | delta (ON - OFF) |")
            print("|---|---|---|---|")
            for rate in (1.20, 2.50):
                off_cell = main_grid[(main_grid["appliance"] == appliance) & (main_grid["persona"] == persona)
                                      & (main_grid["rate"] == rate)]
                on_cell = flag_grid[(flag_grid["appliance"] == appliance) & (flag_grid["persona"] == persona)
                                     & (flag_grid["rate"] == rate)]
                off_mean = off_cell["wtp"].mean() if not off_cell.empty else float("nan")
                on_mean = on_cell["wtp"].mean() if not on_cell.empty else float("nan")
                delta = on_mean - off_mean
                print(f"| {rate} | {off_mean:.3f} | {on_mean:.3f} | {delta:+.3f} |")


if __name__ == "__main__":
    main()
