"""Plotting.

Responsibility: renders the load-curve overlays used across §11-13 (per-run
base/total curve, the multi-agent comparison overlay) to PNG for each run's
results folder. The four-way overlay (adding the LLM agent) lands in
Phase 2/3 -- Phase 1 only has base/MILP/noisy-MILP to compare.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hems.environment.simulator import RunResult

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _kw(load_kwh: list[float]) -> list[float]:
    return [v * 2 for v in load_kwh]  # kWh-per-half-hour -> kW, per unit convention


def plot_load_curve(result: RunResult, out_path: str | Path, title: str | None = None) -> None:
    """Single-run base vs. total load curve, with the flexible layer shaded."""
    clocks = [s.clock for s in result.steps]
    base_kw = _kw(result.base_load_curve())
    total_kw = _kw(result.total_load_curve())

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(clocks, base_kw, label="base (SGSC)", color="0.35", linestyle="--")
    ax.plot(clocks, total_kw, label=f"total ({result.agent_name})", color=_PALETTE[1])
    ax.fill_between(
        clocks, base_kw, total_kw, color=_PALETTE[1], alpha=0.15, label="flexible layer"
    )
    ax.set_xticks(clocks[::4])
    ax.set_xticklabels(clocks[::4], rotation=45, ha="right")
    ax.set_ylabel("kW")
    ax.set_xlabel("time of day")
    ax.set_title(title or f"household {result.household_id} | {result.date} | {result.agent_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overlay(
    results: dict[str, RunResult], out_path: str | Path, title: str | None = None
) -> None:
    """Base load once, plus each agent's total load curve, cost in the legend.

    `results` maps a display label (e.g. "milp", "noisy_milp") to its
    `RunResult`; all entries must share the same household and date.
    """
    any_result = next(iter(results.values()))
    clocks = [s.clock for s in any_result.steps]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        clocks, _kw(any_result.base_load_curve()), label="base (SGSC)", color="0.4", linestyle="--"
    )
    for i, (label, result) in enumerate(results.items()):
        ax.plot(
            clocks,
            _kw(result.total_load_curve()),
            label=f"{label} total (${result.total_cost:.2f})",
            color=_PALETTE[i % len(_PALETTE)],
        )
    ax.set_xticks(clocks[::4])
    ax.set_xticklabels(clocks[::4], rotation=45, ha="right")
    ax.set_ylabel("kW")
    ax.set_xlabel("time of day")
    ax.set_title(title or f"household {any_result.household_id} | {any_result.date}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_four_way(
    base_day: dict[str, list[float]],
    cpp_day: dict[str, list[float]],
    clocks: list[str],
    cpp_window: tuple[int, int],
    out_path: str | Path,
    title: str,
) -> None:
    """Four-way daily load-curve overlay (BUILD_SPEC.md §11), baseline vs CPP day.

    Each dict maps a series label -- "SGSC base", "MILP", "noisy-MILP", "LLM"
    -- to a kWh-per-half-hour curve (one household, or a population sum). The
    CPP window `[start, end)` (step indices) is shaded on the CPP panel.
    """
    styles = {
        "SGSC base": {"color": "0.45", "linestyle": "--", "linewidth": 1.4},
        "MILP": {"color": _PALETTE[0], "linewidth": 1.6},
        "noisy-MILP": {"color": _PALETTE[1], "linewidth": 1.6},
        "LLM": {"color": _PALETTE[2], "linewidth": 2.2},
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, (panel, curves) in zip(axes, [("baseline day (no CPP)", base_day), ("CPP day", cpp_day)]):
        for label, curve in curves.items():
            ax.plot(clocks, _kw(curve), label=label, **styles.get(label, {}))
        if panel == "CPP day":
            ax.axvspan(cpp_window[0] - 0.5, cpp_window[1] - 0.5, color="#C44E52", alpha=0.12,
                       label="CPP window")
        ax.set_title(panel)
        ax.set_xticks(range(0, len(clocks), 4))
        ax.set_xticklabels(clocks[::4], rotation=45, ha="right")
        ax.set_xlabel("time of day")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("kW")
    axes[1].legend(loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
