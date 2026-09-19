"""Analysis pass for probe_wtp_price_v3.py's raw CSV. Reporting only.

Usage:
  python experiments/analyze_probe_v3.py <v3_csv_path>
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


def main() -> None:
    path = sys.argv[1]
    df = pd.read_csv(path)
    main_df = df[df["phase"] == "main"]

    print("=" * 80)
    print("1. PER PERSONA x APPLIANCE x COMFORT STATE")
    print("=" * 80)
    for appliance in ("hvac", "washer"):
        for state in main_df[main_df["appliance"] == appliance]["comfort_state"].unique():
            for persona in ("comfort_first", "neutral", "price_conscious"):
                sub = main_df[(main_df["appliance"] == appliance) & (main_df["comfort_state"] == state)
                              & (main_df["persona"] == persona)].sort_values("rate")
                if sub.empty:
                    continue
                print(f"\n### {appliance} / {state} / {persona}\n")
                print("| price | wtp |")
                print("|---|---|")
                wtps = []
                for _, row in sub.iterrows():
                    print(f"| {row['rate']} | {row['wtp']:.3f} |")
                    wtps.append((row["rate"], row["wtp"]))
                slope = linear_slope(sub["rate"], sub["wtp"])
                reversals = [
                    (wtps[i][0], wtps[i + 1][0])
                    for i in range(len(wtps) - 1)
                    if wtps[i + 1][1] > wtps[i][1] + REVERSAL_TOLERANCE
                ]
                print(f"slope: {slope:.4f}  |  correct sign: {slope <= 0}  |  reversals: {reversals if reversals else 'none'}")

    print("\n" + "=" * 80)
    print("2. LOW-STAKES vs CONTESTED CONTRAST (per persona)")
    print("=" * 80)
    for appliance, (low_label, high_label) in [("hvac", ("comfortable", "contested")), ("washer", ("low_urgency", "high_urgency"))]:
        for persona in ("comfort_first", "neutral", "price_conscious"):
            low = main_df[(main_df["appliance"] == appliance) & (main_df["comfort_state"] == low_label) & (main_df["persona"] == persona)]
            high = main_df[(main_df["appliance"] == appliance) & (main_df["comfort_state"] == high_label) & (main_df["persona"] == persona)]
            if low.empty or high.empty:
                continue
            print(f"\n### {appliance} / {persona}: {low_label} vs {high_label}\n")
            print(f"{low_label} mean WTP: {low['wtp'].mean():.3f} (range ${low['wtp'].max()-low['wtp'].min():.3f})")
            print(f"{high_label} mean WTP: {high['wtp'].mean():.3f} (range ${high['wtp'].max()-high['wtp'].min():.3f})")
            print(f"floor multiple ({high_label}/{low_label}): "
                  f"{(high['wtp'].mean() / low['wtp'].mean()) if low['wtp'].mean() > 1e-9 else float('inf'):.2f}x")

    print("\n" + "=" * 80)
    print("3. PERSONA TRANSMISSION AT CONTESTED STATE")
    print("=" * 80)
    for appliance, state in [("hvac", "contested"), ("washer", "high_urgency")]:
        print(f"\n### {appliance} / {state}\n")
        print("| price | comfort_first | neutral | price_conscious |")
        print("|---|---|---|---|")
        sub = main_df[(main_df["appliance"] == appliance) & (main_df["comfort_state"] == state)]
        for rate in sorted(sub["rate"].unique()):
            row_vals = []
            for persona in ("comfort_first", "neutral", "price_conscious"):
                cell = sub[(sub["rate"] == rate) & (sub["persona"] == persona)]
                row_vals.append(f"{cell['wtp'].iloc[0]:.3f}" if not cell.empty else "n/a")
            print(f"| {rate} | " + " | ".join(row_vals) + " |")
        slopes = {}
        for persona in ("comfort_first", "neutral", "price_conscious"):
            psub = sub[sub["persona"] == persona].sort_values("rate")
            slopes[persona] = linear_slope(psub["rate"], psub["wtp"])
        print(f"\nslopes: {slopes}")

    print("\n" + "=" * 80)
    print("4. REASON STRINGS AT CONTESTED STATE (lowest vs highest price)")
    print("=" * 80)
    for appliance, state in [("hvac", "contested"), ("washer", "high_urgency")]:
        sub = main_df[(main_df["appliance"] == appliance) & (main_df["comfort_state"] == state)]
        lo_price, hi_price = sub["rate"].min(), sub["rate"].max()
        for persona in ("comfort_first", "neutral", "price_conscious"):
            lo = sub[(sub["persona"] == persona) & (sub["rate"] == lo_price)]
            hi = sub[(sub["persona"] == persona) & (sub["rate"] == hi_price)]
            print(f"\n### {appliance}/{state}/{persona}")
            if not lo.empty:
                print(f"  price={lo_price} wtp={lo['wtp'].iloc[0]:.3f}: {lo['reason'].iloc[0]}")
            if not hi.empty:
                print(f"  price={hi_price} wtp={hi['wtp'].iloc[0]:.3f}: {hi['reason'].iloc[0]}")

    print("\n" + "=" * 80)
    print("5. DETERMINISM CHECK")
    print("=" * 80)
    det = df[df["phase"] == "determinism_check"]
    for _, row in det.iterrows():
        orig = main_df[(main_df["appliance"] == row["appliance"]) & (main_df["persona"] == row["persona"])
                       & (main_df["comfort_state"] == row["comfort_state"]) & (main_df["rate"] == row["rate"])]
        if orig.empty:
            continue
        orig_wtp = orig["wtp"].iloc[0]
        identical = abs(orig_wtp - row["wtp"]) < 1e-9
        print(f"{row['appliance']}/{row['persona']}/{row['comfort_state']}/rate={row['rate']}: "
              f"original={orig_wtp:.3f} rerun={row['wtp']:.3f} identical={identical}")

    temp03 = df[df["phase"].astype(str).str.startswith("temp_")]
    if not temp03.empty:
        print("\n" + "=" * 80)
        print("OPTIONAL: temperature=0.3 recheck (single call per price, no seed)")
        print("=" * 80)
        for appliance in temp03["appliance"].unique():
            for persona in temp03[temp03["appliance"] == appliance]["persona"].unique():
                sub = temp03[(temp03["appliance"] == appliance) & (temp03["persona"] == persona)].sort_values("rate")
                print(f"\n### {appliance}/{persona} (T=0.3)\n")
                print("| price | wtp |")
                print("|---|---|")
                for _, row in sub.iterrows():
                    print(f"| {row['rate']} | {row['wtp']:.3f} |")
                slope = linear_slope(sub["rate"], sub["wtp"])
                print(f"slope: {slope:.4f}")


if __name__ == "__main__":
    main()
