"""Phase 0: data-profile plots for the SGSC sample and cluster structure.

Responsibility (BUILD_SPEC.md §13 Phase 0): produces the summary figures
referenced by `data_profile.md` --

  results/data_profile/representative_daily_curves.png
      A handful of sampled households' 30-min load on a summer and a
      winter weekday, from `data/processed/{households,load_curves}.parquet`.

  results/data_profile/avg_weekday_shape_by_cluster.png
      The average weekday (Mon-Fri) load shape per `cluster_all` label,
      computed across *all* households in each cluster -- this needs the
      full raw interval data (`data/raw/interval load data/data_3_groups.csv`,
      gitignored), not just the small committed sample, so re-run this
      script against the raw data if the figure needs regenerating.

Data prep only -- not part of the simulator core.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
OUT = Path("results/data_profile")

SUMMER_DAY = "2013-01-16"  # mid-Jan Wednesday -- Australian summer
WINTER_DAY = "2013-07-17"  # mid-Jul Wednesday -- Australian winter

# tab10-derived, colorblind-legible on both light and dark backgrounds
CLUSTER_COLORS = plt.get_cmap("tab10").colors


def plot_representative_daily_curves() -> None:
    hh = pd.read_parquet(PROCESSED / "households.parquet")
    curves = pd.read_parquet(PROCESSED / "load_curves.parquet")
    curves["timestamp"] = pd.to_datetime(curves["timestamp"])

    # one NotUnit + one Unit household from three spread-out clusters
    picks = (
        hh[hh["cluster_all"].isin([1, 5, 9])]
        .sort_values(["cluster_all", "ASSRTD_DWELLING_TYPE_CD"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, day, label in zip(axes, [SUMMER_DAY, WINTER_DAY], ["Summer (16 Jan)", "Winter (17 Jul)"]):
        day_curves = curves[curves["timestamp"].dt.date.astype(str) == day]
        for i, row in picks.reset_index(drop=True).iterrows():
            series = day_curves[day_curves["customer_ID"] == row["customer_ID"]].sort_values("timestamp")
            hours = series["timestamp"].dt.hour + series["timestamp"].dt.minute / 60
            ax.plot(
                hours,
                series["kwh"] * 2,  # kWh/30min -> kW
                color=CLUSTER_COLORS[int(row["cluster_all"]) - 1],
                linestyle="-" if row["ASSRTD_DWELLING_TYPE_CD"] == "NotUnit" else "--",
                label=f"cluster {row['cluster_all']} ({row['ASSRTD_DWELLING_TYPE_CD']})",
                linewidth=1.6,
            )
        ax.set_title(label)
        ax.set_xlabel("hour of day")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("load (kW)")
    axes[1].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("Representative household load curves -- SGSC sample (2013)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "representative_daily_curves.png", dpi=150)
    plt.close(fig)


def plot_avg_weekday_shape_by_cluster() -> None:
    load = pd.read_csv(RAW / "interval load data" / "data_3_groups.csv")
    load["timestamp"] = pd.to_datetime(load["timestamp"])
    demo = pd.read_csv(RAW / "index files" / "demographic_lookup_3_groups.csv")

    weekday = load[load["timestamp"].dt.weekday < 5].copy()
    weekday["slot"] = weekday["timestamp"].dt.hour * 2 + weekday["timestamp"].dt.minute // 30

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cluster in sorted(demo["cluster_all"].dropna().unique()):
        ids = demo.loc[demo["cluster_all"] == cluster, "customer_ID"].astype(str)
        ids = [c for c in ids if c in weekday.columns]
        cluster_mean_kw = weekday.groupby("slot")[ids].mean().mean(axis=1) * 2
        hours = cluster_mean_kw.index / 2
        ax.plot(
            hours,
            cluster_mean_kw.values,
            color=CLUSTER_COLORS[int(cluster) - 1],
            label=f"cluster {int(cluster)} (n={len(ids)})",
            linewidth=1.8,
        )
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean load (kW)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Average weekday load shape by cluster (SGSC, all of 2013, n=6031 households)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "avg_weekday_shape_by_cluster.png", dpi=150)
    plt.close(fig)


def main() -> None:
    plot_representative_daily_curves()
    plot_avg_weekday_shape_by_cluster()
    print(f"wrote plots to {OUT}/")


if __name__ == "__main__":
    main()
