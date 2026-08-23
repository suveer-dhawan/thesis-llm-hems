"""One household, one agent.

Responsibility (BUILD_SPEC.md §3, §16 P0): CLI entry point that runs a
single household through the full simulator loop for one agent and writes
results to `results/phase1/`: a load-curve plot, a per-step decision log
(parquet), and a `summary.json` of metrics (BUILD_SPEC.md §13).

Only `--agent stub` is wired up so far -- MILP/noisy-MILP land in Part 1b.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from hems.agents.base import DecisionAgent
from hems.agents.stub_agent import StubAgent
from hems.analysis.metrics import summary_dict
from hems.analysis.plots import plot_load_curve
from hems.environment.appliances import HVACParams, WasherParams
from hems.environment.household import Household
from hems.environment.simulator import RunResult, run_simulation
from hems.environment.tariff import Tariff

AGENT_NAMES = ["stub"]

DEFAULT_DATE = "2013-01-18"  # hot summer weekday for household 8596070, see BUILD_SPEC.md
DEFAULT_RESULTS_DIR = "results/phase1"


def load_environment(
    household_path: str = "config/household.yaml",
    appliances_path: str = "config/appliances.yaml",
    tariff_path: str = "config/tariff.yaml",
    load_curves_path: str = "data/processed/load_curves.parquet",
) -> tuple[Household, Tariff, HVACParams, WasherParams]:
    household = Household.from_config(household_path, load_curves_path)
    tariff = Tariff.from_yaml(tariff_path)
    hvac_params = HVACParams.from_yaml(appliances_path)
    washer_params = WasherParams.from_yaml(appliances_path, household_path)
    return household, tariff, hvac_params, washer_params


def build_agent(name: str) -> DecisionAgent:
    if name not in AGENT_NAMES:
        raise NotImplementedError(
            f"agent '{name}' is not implemented yet; available: {AGENT_NAMES}"
        )
    return StubAgent()


def write_run_outputs(result: RunResult, out_dir: Path, prefix: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_dict(result)
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    result.to_dataframe().to_parquet(out_dir / f"{prefix}_decision_log.parquet")
    plot_load_curve(result, out_dir / f"{prefix}_load_curve.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="stub", choices=AGENT_NAMES, help="agent to run")
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="YYYY-MM-DD simulation day (default: Phase 1 hot weekday)",
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    date = dt.date.fromisoformat(args.date)
    household, tariff, hvac_params, washer_params = load_environment()
    out_dir = Path(args.results_dir)

    agent = build_agent(args.agent)
    result = run_simulation(household, tariff, hvac_params, washer_params, agent, date)
    summary = write_run_outputs(result, out_dir, args.agent)
    print(
        f"[run_single] agent={result.agent_name} household={result.household_id} "
        f"date={result.date} "
        f"base_kwh={summary['base_kwh_total']} flexible_kwh={summary['flexible_kwh_total']} "
        f"flexible_share_pct={summary['flexible_share_pct']} total_cost=${summary['total_cost']} "
        f"comfort_violations={summary['comfort_violation_steps']} "
        f"washer_completed={summary['washer_completed']}"
    )


if __name__ == "__main__":
    main()
