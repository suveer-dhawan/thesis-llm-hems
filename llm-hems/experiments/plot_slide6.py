"""Slide-6 load-curve chart: real Phase 1 data, deck-styled.

Responsibility: renders a presentation-ready PNG of the base (SGSC) vs.
MILP total load curve for household 8596070 / 2013-01-18, using the exact
per-step arrays a real Phase 1 MILP run produces (`run_single.run_agent`),
restyled for the slide deck palette. This is a standalone slide asset, not
part of the `analysis.plots` overlays used by `run_single.py` -- those stay
generic (any household/agent, live titles); this one is fixed to one
household/day and deliberately drops MILP-specific labelling per the deck's
own context.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hems.analysis.metrics import peak_kw, peak_to_average_ratio, time_of_peak_clock
from experiments.run_single import load_environment, run_agent

HOUSEHOLD_ID = 8596070
DATE = dt.date(2013, 1, 18)
OUT_PATH = Path("results/phase1/slide6_peak_chart.png")

BG = "#FBFAF7"
INK = "#191817"
MUTED = "#6E6A63"
GRID = "#E5E0D8"
CLAY = "#9E6A50"

PEAK_PRICE_START_H = 15.0
PEAK_PRICE_END_H = 21.0


def _kw(load_kwh: list[float]) -> list[float]:
    return [v * 2 for v in load_kwh]


def main() -> None:
    household, tariff, hvac_params, washer_params = load_environment()
    assert household.params.household_id == HOUSEHOLD_ID

    result = run_agent("milp", household, tariff, hvac_params, washer_params, DATE, seed=42)

    base_curve = result.base_load_curve()
    total_curve = result.total_load_curve()
    base_kw = _kw(base_curve)
    total_kw = _kw(total_curve)
    hours = [i * 0.5 for i in range(len(base_kw))]

    base_peak = peak_kw(base_curve)
    base_peak_idx = max(range(len(base_kw)), key=lambda i: base_kw[i])
    base_peak_h = hours[base_peak_idx]

    total_peak = peak_kw(total_curve)
    total_peak_idx = max(range(len(total_kw)), key=lambda i: total_kw[i])
    total_peak_h = hours[total_peak_idx]

    base_par = peak_to_average_ratio(base_curve)
    total_par = peak_to_average_ratio(total_curve)
    print(
        f"[plot_slide6] base peak = {base_peak:.3f} kW @ "
        f"{time_of_peak_clock(result, base_curve)} (PAR {base_par:.2f})"
    )
    print(
        f"[plot_slide6] optimised peak = {total_peak:.3f} kW @ "
        f"{time_of_peak_clock(result, total_curve)} (PAR {total_par:.2f})"
    )

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 13

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ymax = max(total_kw) * 1.42
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 24)

    ax.axvspan(
        PEAK_PRICE_START_H, PEAK_PRICE_END_H, color=CLAY, alpha=0.07, zorder=0, lw=0
    )
    ax.text(
        (PEAK_PRICE_START_H + PEAK_PRICE_END_H) / 2,
        ymax * 0.965,
        "PEAK PRICE  3–9 PM",
        color=CLAY,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="top",
        zorder=1,
    )

    ax.plot(
        hours,
        base_kw,
        label="Typical household",
        color=MUTED,
        linestyle=(0, (5, 3)),
        linewidth=2.2,
        zorder=2,
    )
    ax.plot(
        hours,
        total_kw,
        label="Cost-optimised schedule",
        color=CLAY,
        linestyle="-",
        linewidth=2.8,
        zorder=3,
    )

    ax.scatter(
        [base_peak_h], [base_peak], s=32, color=MUTED, zorder=4, edgecolors=BG, linewidths=1.0
    )
    ax.scatter(
        [total_peak_h], [total_peak], s=42, color=CLAY, zorder=5, edgecolors=BG, linewidths=1.2
    )

    label_x = max(total_peak_h - 6.0, 1.5)
    label_y = ymax * 0.80
    ax.annotate(
        "taller, sharper peak",
        xy=(total_peak_h, total_peak),
        xytext=(label_x, label_y),
        color=CLAY,
        fontsize=11,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="-",
            color=CLAY,
            lw=1.1,
            connectionstyle="arc3,rad=0.25",
            shrinkA=2,
            shrinkB=6,
        ),
        zorder=6,
    )

    ax.set_ylabel("Household demand (kW)", color=INK)
    ax.set_xlabel("Time of day", color=INK)

    xticks = [0, 4, 8, 12, 16, 20, 24]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{h:02d}:00" for h in xticks], color=MUTED)
    ax.tick_params(axis="both", length=0, colors=MUTED)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    ax.yaxis.grid(True, color=GRID, alpha=0.7, linewidth=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="upper left", frameon=False, fontsize=12)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[plot_slide6] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
