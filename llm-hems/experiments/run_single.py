"""One household, one agent.

Responsibility (BUILD_SPEC.md §3, §16 P0/P1): CLI entry point that runs a
single household through the full simulator loop for one agent and writes
results to `results/phase1/`: a load-curve plot, a per-step decision log
(parquet), and a `summary.json` of metrics (BUILD_SPEC.md §13).

`--agent stub` runs the deterministic environment alone (Part 1a). `--agent
milp` / `--agent noisy_milp` additionally run the *other* MILP baseline for
comparison (deterministic, no LLM spend either way) and write a 3-way
overlay plot -- base (SGSC) / MILP total / noisy-MILP total -- plus each
run's total cost (Part 1b, BUILD_SPEC.md §13 Phase 1 definition of done).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import yaml

from hems.agents.base import DecisionAgent
from hems.agents.llm_agent import LLMAgent
from hems.agents.milp_agent import MILPAgent
from hems.agents.noisy_milp_agent import NoisyMILPAgent
from hems.agents.stub_agent import StubAgent
from hems.analysis.metrics import summary_dict
from hems.analysis.plots import plot_load_curve, plot_overlay
from hems.environment.appliances import HVACParams, WasherParams
from hems.environment.household import Household
from hems.environment.simulator import RunResult, run_simulation
from hems.environment.tariff import Tariff
from hems.llm.client import build_llm_client  # noqa: F401 -- re-exported for older scripts
from hems.provenance import write_provenance

AGENT_NAMES = ["stub", "milp", "noisy_milp", "llm"]

DEFAULT_DATE = "2013-01-18"  # hot summer weekday for household 8596070, see BUILD_SPEC.md
DEFAULT_RESULTS_DIR = "results/phase1"
DEFAULT_LLM_CACHE_PATH = "results/llm_cache.json"
DEFAULT_EXPERIMENT_CONFIG = "config/experiment.yaml"


def load_llm_config(experiment_path: str = DEFAULT_EXPERIMENT_CONFIG) -> dict:
    """The `llm:` block of experiment.yaml -- provider selection + per-provider settings."""
    return yaml.safe_load(Path(experiment_path).read_text()).get("llm", {})


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


def build_agent(
    name: str,
    household: Household,
    tariff: Tariff,
    hvac_params: HVACParams,
    washer_params: WasherParams,
    date: dt.date,
    seed: int,
    dry_run: bool = False,
    cache_path: str | None = DEFAULT_LLM_CACHE_PATH,
    provider: str | None = None,
) -> DecisionAgent:
    if name == "stub":
        return StubAgent()
    if name == "milp":
        return MILPAgent(household, tariff, hvac_params, washer_params, date)
    if name == "noisy_milp":
        return NoisyMILPAgent(household, tariff, hvac_params, washer_params, date, seed=seed)
    if name == "llm":
        if dry_run:
            return LLMAgent(dry_run=True)
        llm_cfg = load_llm_config()
        resolved_provider = provider or llm_cfg.get("provider", "matilda")
        client = build_llm_client(resolved_provider, llm_cfg, cache_path)
        return LLMAgent(client=client)
    raise NotImplementedError(f"agent '{name}' is not implemented yet; available: {AGENT_NAMES}")


def run_agent(
    name: str,
    household: Household,
    tariff: Tariff,
    hvac_params: HVACParams,
    washer_params: WasherParams,
    date: dt.date,
    seed: int,
    dry_run: bool = False,
    cache_path: str | None = DEFAULT_LLM_CACHE_PATH,
    provider: str | None = None,
) -> tuple[RunResult, DecisionAgent]:
    """Builds the agent and runs it through the simulator.

    For `noisy_milp`, the agent's *own* jittered comfort/deadline params
    (its heterogeneous household's real preferences, not a belief/reality
    gap -- see `noisy_milp_agent.py`) are fed back into `run_simulation`
    so its accounting is internally consistent. Returns the agent alongside
    the result so callers (e.g. `--agent llm`) can pull agent-specific state
    such as the LLM decision log. `provider` (llm agent only) overrides
    `llm.provider` from experiment.yaml -- see `load_llm_config`.
    """
    agent = build_agent(
        name, household, tariff, hvac_params, washer_params, date, seed, dry_run, cache_path,
        provider,
    )
    run_hvac_params = hvac_params
    run_washer_params = washer_params
    if isinstance(agent, NoisyMILPAgent):
        run_hvac_params = agent.jittered_hvac_params
        run_washer_params = agent.jittered_washer_params
    result = run_simulation(household, tariff, run_hvac_params, run_washer_params, agent, date)
    return result, agent


def write_run_outputs(result: RunResult, out_dir: Path, prefix: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(out_dir)
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
    parser.add_argument("--seed", type=int, default=42, help="noisy_milp jitter seed")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="llm agent only: use a deterministic stub instead of calling the API (§15, no spend)",
    )
    parser.add_argument(
        "--cache-path",
        default=DEFAULT_LLM_CACHE_PATH,
        help="llm agent only: disk cache of (provider, prompt) -> response, reused across runs",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["matilda", "gemini"],
        help="llm agent only: override llm.provider from experiment.yaml",
    )
    args = parser.parse_args()

    date = dt.date.fromisoformat(args.date)
    household, tariff, hvac_params, washer_params = load_environment()
    out_dir = Path(args.results_dir)

    result, agent = run_agent(
        args.agent,
        household,
        tariff,
        hvac_params,
        washer_params,
        date,
        args.seed,
        dry_run=args.dry_run,
        cache_path=args.cache_path,
        provider=args.provider,
    )
    summary = write_run_outputs(result, out_dir, args.agent)
    print(
        f"[run_single] agent={result.agent_name} household={result.household_id} "
        f"date={result.date} "
        f"base_kwh={summary['base_kwh_total']} flexible_kwh={summary['flexible_kwh_total']} "
        f"flexible_share_pct={summary['flexible_share_pct']} total_cost=${summary['total_cost']} "
        f"comfort_violations={summary['comfort_violation_steps']} "
        f"washer_completed={summary['washer_completed']}"
    )

    if isinstance(agent, LLMAgent):
        # Distinct filename from write_run_outputs' "{prefix}_decision_log.parquet"
        # (the simulator's per-step StepLog, with indoor_temp_c/in_band/etc.) --
        # this is the agent's own prompt/wtp/reason log (BUILD_SPEC.md §6.3).
        agent.log_to_dataframe().to_parquet(out_dir / "llm_agent_log.parquet")
        (out_dir / "llm_meta.json").write_text(
            json.dumps(
                {
                    "provider": agent.provider,
                    "model_id": agent.model_id,
                    "sdk_version": agent.sdk_version,
                },
                indent=2,
            )
        )
        print(
            f"[run_single] llm decision log: {len(agent.log)} steps logged "
            f"(provider={agent.provider} model_id={agent.model_id} dry_run={args.dry_run})"
        )

    if args.agent in ("milp", "noisy_milp"):
        other = "noisy_milp" if args.agent == "milp" else "milp"
        other_result, _ = run_agent(
            other, household, tariff, hvac_params, washer_params, date, args.seed
        )
        write_run_outputs(other_result, out_dir, other)

        milp_result = result if args.agent == "milp" else other_result
        noisy_result = other_result if args.agent == "milp" else result

        plot_overlay(
            {"milp": milp_result, "noisy_milp": noisy_result},
            out_dir / "comparison_overlay.png",
            title=f"household {result.household_id} | {result.date} | MILP vs noisy-MILP",
        )
        comparison = {
            "household_id": result.household_id,
            "date": result.date.isoformat(),
            "base_kwh_total": round(milp_result.base_kwh_total, 4),
            "milp": {
                "total_cost": round(milp_result.total_cost, 4),
                "flexible_share_pct": round(milp_result.flexible_share * 100, 2),
                "washer_completed_by_step": milp_result.washer_completed_by_step,
            },
            "noisy_milp": {
                "total_cost": round(noisy_result.total_cost, 4),
                "flexible_share_pct": round(noisy_result.flexible_share * 100, 2),
                "washer_completed_by_step": noisy_result.washer_completed_by_step,
                "seed": args.seed,
            },
        }
        (out_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2))
        print(
            f"[run_single] comparison: milp_cost=${comparison['milp']['total_cost']} "
            f"noisy_milp_cost=${comparison['noisy_milp']['total_cost']} "
            f"milp_washer_step={comparison['milp']['washer_completed_by_step']} "
            f"noisy_milp_washer_step={comparison['noisy_milp']['washer_completed_by_step']}"
        )

    if args.agent == "llm":
        milp_result, _ = run_agent(
            "milp", household, tariff, hvac_params, washer_params, date, args.seed
        )
        write_run_outputs(milp_result, out_dir, "milp")

        plot_overlay(
            {"milp": milp_result, "llm": result},
            out_dir / "llm_vs_milp_overlay.png",
            title=f"household {result.household_id} | {result.date} | MILP vs LLM",
        )
        comparison = {
            "household_id": result.household_id,
            "date": result.date.isoformat(),
            "base_kwh_total": round(milp_result.base_kwh_total, 4),
            "milp": {
                "total_cost": round(milp_result.total_cost, 4),
                "flexible_share_pct": round(milp_result.flexible_share * 100, 2),
                "washer_completed_by_step": milp_result.washer_completed_by_step,
            },
            "llm": {
                "total_cost": round(result.total_cost, 4),
                "flexible_share_pct": round(result.flexible_share * 100, 2),
                "washer_completed_by_step": result.washer_completed_by_step,
                "dry_run": args.dry_run,
            },
        }
        (out_dir / "llm_vs_milp_summary.json").write_text(json.dumps(comparison, indent=2))
        print(
            f"[run_single] comparison: milp_cost=${comparison['milp']['total_cost']} "
            f"llm_cost=${comparison['llm']['total_cost']} "
            f"milp_flexible_share_pct={comparison['milp']['flexible_share_pct']} "
            f"llm_flexible_share_pct={comparison['llm']['flexible_share_pct']}"
        )


if __name__ == "__main__":
    main()
