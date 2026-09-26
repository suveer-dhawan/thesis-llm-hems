"""Phase 3 runner: matched baseline/CPP days, calibration, scenario, budget.

Responsibility (BUILD_SPEC.md §10-13 Phase 3): the library behind
`experiments/run_phase3.py` and `experiments/calibrate.py`. Everything is
driven by the `phase3:` block of `config/experiment.yaml` plus the persona
assignment (`config/persona_assignment.csv`).

- Agents: MILP, noisy-MILP, the rule-based satisficer (the LLM agent's halt
  with a formula WTP) and the LLM. The rule-based agent and the LLM get the
  same global_scale and archetype factor. `phase3.band_mode` sets every
  agent's comfort band (`fixed` +/-fixed_band_c, or each `persona`'s own).
- `Phase3Runner.run_day` simulates one household x agent x seed x wtp_scale
  on one date under one tariff variant, memoised so a step that re-asks for
  the same run (calibration grid point == frozen scale, overlay == elasticity
  run, ...) never re-simulates it -- and, for the LLM, never re-prompts it.
- `run_pair` is the §10 unit: same date, baseline tariff (no CPP) vs CPP
  tariff (exactly one event), everything else fixed.
- `stratified_split` / `fit_scale` implement the pre-registered calibration.
- `run_scenario` + `scenario_verdict` implement the §12 discriminating
  scenario.

LLM cost control: one `LLMClient` per seed, all sharing one append-only disk
cache (`phase3.llm_cache_path`, overridable per run for sharding);
`dry_run=True` swaps in the deterministic stub so the whole pipeline runs
with zero spend.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd
import yaml

from hems.agents.llm_agent import LLMAgent
from hems.agents.milp_agent import MILPAgent
from hems.agents.noisy_milp_agent import NoisyMILPAgent
from hems.agents.rule_based_agent import RuleBasedParams, RuleBasedSatisficerAgent
from hems.analysis.elasticity import ElasticityRecord, household_elasticity, pooled_elasticity
from hems.environment.appliances import HVACParams, WasherParams, _clock_to_step
from hems.environment.household import Household
from hems.environment.persona import (
    ThermalHeterogeneity,
    load_assignment,
    load_persona_environment,
)
from hems.environment.simulator import RunResult, run_simulation
from hems.environment.tariff import Tariff
from hems.llm.client import LLMClient, build_llm_client

AGENTS = ("milp", "noisy_milp", "rule_based", "llm")
WTP_AGENTS = ("rule_based", "llm")  # emit a WTP through the shared halt -> scale + factor apply
DETERMINISTIC_AGENTS = ("milp", "rule_based")  # no seed dependence
BAND_MODES = ("fixed", "persona")
DEFAULT_EXPERIMENT_CONFIG = "config/experiment.yaml"


@dataclass(frozen=True)
class Phase3Config:
    raw: dict  # the resolved `phase3:` block (overrides applied), written into every results folder
    llm_cfg: dict
    experiment_path: str
    dry_run: bool
    thermal: ThermalHeterogeneity = ThermalHeterogeneity()
    full: dict = field(default_factory=dict)  # the whole experiment.yaml, as loaded

    @classmethod
    def from_yaml(
        cls,
        path: str = DEFAULT_EXPERIMENT_CONFIG,
        dry_run: bool = False,
        band_mode: str | None = None,
        cache_path: str | None = None,
        thermal_enabled: bool | None = None,
    ) -> "Phase3Config":
        """Load experiment.yaml; `band_mode` / `cache_path` / `thermal_enabled`
        override the file's values and are written back into `raw` / `thermal`
        so the resolved config records what actually ran."""
        cfg = yaml.safe_load(Path(path).read_text())
        raw = dict(cfg["phase3"])
        if band_mode is not None:
            raw["band_mode"] = band_mode
        if cache_path is not None:
            raw["llm_cache_path"] = cache_path
        if raw.get("band_mode", "persona") not in BAND_MODES:
            raise ValueError(f"phase3.band_mode must be one of {BAND_MODES}")
        thermal = ThermalHeterogeneity.from_config(cfg.get("thermal_heterogeneity"))
        if thermal_enabled is not None:
            thermal = replace(thermal, enabled=thermal_enabled)
        return cls(
            raw=raw,
            llm_cfg=cfg.get("llm", {}),
            experiment_path=path,
            dry_run=dry_run,
            thermal=thermal,
            full=cfg,
        )

    @property
    def dates(self) -> list[dt.date]:
        return [dt.date.fromisoformat(d) for d in self.raw["dates"]]

    @property
    def cpp_steps(self) -> tuple[int, int]:
        return _clock_to_step(self.raw["cpp"]["start"]), _clock_to_step(self.raw["cpp"]["end"])

    @property
    def rebound_steps(self) -> int:
        return int(round(float(self.raw["rebound_hours"]) * 2))

    @property
    def band_mode(self) -> str:
        return str(self.raw.get("band_mode", "persona"))

    @property
    def fixed_band_c(self) -> float:
        return float(self.raw["fixed_band_c"])

    @property
    def washer_enabled(self) -> bool:
        return bool(self.raw.get("washer_enabled", True))

    @property
    def llm_seeds(self) -> list[int]:
        return [int(s) for s in self.raw["llm_seeds"]]

    @property
    def noisy_milp_seeds(self) -> list[int]:
        return [int(s) for s in self.raw["noisy_milp_seeds"]]

    @property
    def primary_seed(self) -> int:
        """The seed elasticity / overlay / scenario run at (first LLM seed)."""
        return self.llm_seeds[0]

    def seeds_for(self, agent: str) -> list[int]:
        """Seeds an agent is run at: one for deterministic agents, the lists otherwise."""
        if agent in DETERMINISTIC_AGENTS:
            return [self.primary_seed]
        return self.noisy_milp_seeds if agent == "noisy_milp" else self.llm_seeds

    @property
    def conway_range(self) -> tuple[float, float]:
        lo, hi = (float(x) for x in self.raw["conway_range"])
        return lo, hi

    @property
    def calibration(self) -> dict:
        return self.raw["calibration"]

    @property
    def calibration_enabled(self) -> bool:
        return bool(self.calibration.get("enabled", True))

    def rule_based_params(self) -> RuleBasedParams:
        return RuleBasedParams.from_yaml(self.experiment_path, band_mode=self.band_mode)

    def resolved(self) -> dict:
        """Everything that shapes a run, for `config_resolved.yaml`."""
        rb = self.rule_based_params()
        resolved = {
            "phase3": self.raw,
            "band": {
                "mode": self.band_mode,
                "fixed_band_c": self.fixed_band_c if self.band_mode == "fixed" else None,
            },
            "seeds": {
                "primary": self.primary_seed,
                "llm": self.llm_seeds,
                "noisy_milp": self.noisy_milp_seeds,
                "deterministic_agents": list(DETERMINISTIC_AGENTS),
            },
            "wtp_scaling": {
                "global_scale": self.global_scale,
                "archetype_factors": self.archetype_factors,
                "applies_to": list(WTP_AGENTS),
                "calibration_enabled": self.calibration_enabled,
            },
            "washer_enabled": self.washer_enabled,
            "thermal_heterogeneity": dataclasses.asdict(self.thermal),
            "rule_based": {
                **(self.full.get("rule_based") or {}),
                "band_mode_used": self.band_mode,
                "hvac_k_used": rb.k_by_orientation,
                "washer_w_max_used": rb.w_max,
            },
            "noisy_milp": self.full.get("noisy_milp"),
            "llm": self.llm_cfg,
            "agents": list(AGENTS),
            "dry_run": self.dry_run,
            "experiment_path": self.experiment_path,
        }
        # plain JSON types (tuples -> lists), so it reads back from YAML identically
        return json.loads(json.dumps(resolved))

    @property
    def fit_seed(self) -> int:
        return int(self.calibration["fit_seed"])

    @property
    def global_scale(self) -> float:
        return float(self.raw.get("wtp_scaling", {}).get("global_scale", 1.0))

    @property
    def archetype_factors(self) -> dict[str, float]:
        factors = self.raw.get("wtp_scaling", {}).get("archetype_factors", {})
        return {str(k): float(v) for k, v in factors.items()}


@dataclass
class DayRun:
    household_id: int
    agent: str
    seed: int
    wtp_scale: float
    date: dt.date
    label: str  # "base" | "cpp" | scenario variant name
    result: RunResult
    llm_log: pd.DataFrame | None = None
    rule_log: pd.DataFrame | None = None  # rule-based satisficer's per-decision log
    prompted: int = 0  # LLM decisions that needed a model answer
    live_calls: int = 0  # of those, how many went to the API (not cache, not stub)


@dataclass
class Phase3Runner:
    cfg: Phase3Config
    assignment: pd.DataFrame = field(init=False)
    tariff: Tariff = field(init=False)

    def __post_init__(self) -> None:
        self.assignment = load_assignment(self.cfg.raw["persona_assignment"])
        self.tariff = Tariff.from_yaml("config/tariff.yaml")
        self._envs: dict[int, tuple[Household, HVACParams, WasherParams]] = {}
        self.rule_params = self.cfg.rule_based_params()
        self._clients: dict[int, LLMClient] = {}
        self._runs: dict[tuple, DayRun] = {}

    # -- lookups -----------------------------------------------------------

    def archetype(self, household_id: int) -> str:
        return str(
            self.assignment.loc[self.assignment["customer_ID"] == household_id, "archetype"].iloc[0]
        )

    def wtp_factor(self, household_id: int) -> float:
        """The household's deterministic cost-sensitivity WTP factor (1.0 if unset)."""
        return self.cfg.archetype_factors.get(self.archetype(household_id), 1.0)

    def env(self, household_id: int) -> tuple[Household, HVACParams, WasherParams]:
        """Household + HVAC/washer params, with the run's band mode applied."""
        if household_id not in self._envs:
            household, hvac, washer = load_persona_environment(
                household_id, self.assignment, thermal=self.cfg.thermal
            )
            if self.cfg.band_mode == "fixed":
                hvac = replace(hvac, tolerance_band_c=self.cfg.fixed_band_c)
            if not self.cfg.washer_enabled:
                # A zero-length cycle is complete from step 0: every agent's
                # washer decision is forced off (no prompt), MILP schedules nothing.
                washer = replace(washer, duration_steps=0, power_kw=0.0)
            self._envs[household_id] = (household, hvac, washer)
        return self._envs[household_id]

    def client(self, seed: int) -> LLMClient:
        if seed not in self._clients:
            provider = self.cfg.llm_cfg.get("provider", "gemini")
            self._clients[seed] = build_llm_client(
                provider, self.cfg.llm_cfg, self.cfg.raw["llm_cache_path"], seed=seed
            )
        return self._clients[seed]

    def tariff_for(self, date: dt.date, label: str) -> Tariff:
        if label == "base":
            return self.tariff.with_cpp_events([])
        if label == "cpp":
            c = self.cfg.raw["cpp"]
            return self.tariff.with_cpp_events(
                [self.tariff.make_cpp_event(date, c["start"], c["end"], severity=c["severity"])]
            )
        s = self.cfg.raw["scenario"]["spike"]
        return self.tariff.with_cpp_events(
            [self.tariff.make_cpp_event(date, s["start"], s["end"], severity=s["severity"])]
        )

    @property
    def runs(self) -> list[DayRun]:
        return list(self._runs.values())

    # -- simulation --------------------------------------------------------

    def run_day(
        self,
        household_id: int,
        agent_name: str,
        date: dt.date,
        label: str,
        seed: int,
        wtp_scale: float = 1.0,
    ) -> DayRun:
        """One simulated day, memoised on every input that can change it."""
        scale_key = wtp_scale if agent_name in WTP_AGENTS else 1.0
        key = (
            household_id,
            agent_name,
            date,
            label,
            seed if agent_name not in DETERMINISTIC_AGENTS else 0,
            scale_key,
        )
        if key in self._runs:
            return self._runs[key]

        household, hvac, washer = self.env(household_id)
        if label in self.cfg.raw["scenario"]["variants"]:
            v = self.cfg.raw["scenario"]["variants"][label]
            washer = replace(
                washer,
                released_step=_clock_to_step(v["washer_release"]),
                deadline_step=_clock_to_step(v["washer_deadline"]),
            )
        tariff = self.tariff_for(date, label)

        if agent_name == "milp":
            agent = MILPAgent(household, tariff, hvac, washer, date)
        elif agent_name == "noisy_milp":
            agent = NoisyMILPAgent(
                household,
                tariff,
                hvac,
                washer,
                date,
                seed=seed,
                jitter_config_path=self.cfg.experiment_path,
            )
            # The jittered params are this household's real preferences (see
            # noisy_milp_agent.py), so they drive its own accounting too.
            hvac, washer = agent.jittered_hvac_params, agent.jittered_washer_params
        elif agent_name == "rule_based":
            agent = RuleBasedSatisficerAgent(
                self.rule_params, wtp_scale=wtp_scale, wtp_factor=self.wtp_factor(household_id)
            )
        elif agent_name == "llm":
            factor = self.wtp_factor(household_id)
            agent = (
                LLMAgent(dry_run=True, wtp_scale=wtp_scale, wtp_factor=factor)
                if self.cfg.dry_run
                else LLMAgent(client=self.client(seed), wtp_scale=wtp_scale, wtp_factor=factor)
            )
        else:
            raise ValueError(f"unknown agent '{agent_name}' -- choose from {AGENTS}")

        result = run_simulation(household, tariff, hvac, washer, agent, date)
        run = DayRun(household_id, agent_name, seed, scale_key, date, label, result)
        if isinstance(agent, LLMAgent):
            run.llm_log = agent.log_to_dataframe()
            prompted = run.llm_log[~run.llm_log["forced"]]
            run.prompted = len(prompted)
            run.live_calls = int((~prompted["cached"] & ~prompted["dry_run"]).sum())
        elif isinstance(agent, RuleBasedSatisficerAgent):
            run.rule_log = agent.log_to_dataframe()
        self._runs[key] = run
        return run

    def run_pair(
        self, household_id: int, agent_name: str, date: dt.date, seed: int, wtp_scale: float = 1.0
    ) -> tuple[DayRun, DayRun]:
        return (
            self.run_day(household_id, agent_name, date, "base", seed, wtp_scale),
            self.run_day(household_id, agent_name, date, "cpp", seed, wtp_scale),
        )

    def elasticity_records(
        self, household_ids: list[int], agent_name: str, seed: int, wtp_scale: float = 1.0
    ) -> list[ElasticityRecord]:
        records = []
        for hid in household_ids:
            for date in self.cfg.dates:
                base, cpp = self.run_pair(hid, agent_name, date, seed, wtp_scale)
                rec = household_elasticity(base.result, cpp.result, seed, self.archetype(hid))
                rec.wtp_scale = wtp_scale if agent_name in WTP_AGENTS else None
                records.append(rec)
        return records

    def llm_call_stats(self) -> dict:
        llm_runs = [r for r in self.runs if r.agent == "llm"]
        return {
            "prompted_decisions": sum(r.prompted for r in llm_runs),
            "live_calls_this_process": sum(r.live_calls for r in llm_runs),
            "client_stats_by_seed": {s: dict(c.stats) for s, c in self._clients.items()},
            "model_versions_by_seed": {
                s: dict(c.model_versions_seen) for s, c in self._clients.items()
            },
            "client_settings_by_seed": {s: c.settings for s, c in self._clients.items()},
            "dry_run": self.cfg.dry_run,
        }


# --------------------------------------------------------------------------
# Calibration (pre-registered -- see experiment.yaml phase3.calibration)
# --------------------------------------------------------------------------


def stratified_split(
    assignment: pd.DataFrame, household_ids: list[int], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Per archetype: shuffle (seeded), send round(n * fraction) to calibration.

    Clamped so an archetype with >= 2 households always lands on both sides;
    an archetype with one household goes to calibration.
    """
    rng = random.Random(seed)
    sub = assignment[assignment["customer_ID"].isin(household_ids)]
    cal: list[int] = []
    held: list[int] = []
    for archetype in sorted(sub["archetype"].unique()):
        ids = sorted(int(i) for i in sub.loc[sub["archetype"] == archetype, "customer_ID"])
        rng.shuffle(ids)
        n_cal = round(len(ids) * fraction)
        if len(ids) >= 2:
            n_cal = min(max(n_cal, 1), len(ids) - 1)
        else:
            n_cal = len(ids)
        cal += ids[:n_cal]
        held += ids[n_cal:]
    return sorted(cal), sorted(held)


@dataclass
class ScaleFit:
    scale: float
    elasticity: float
    in_range: bool
    grid: pd.DataFrame


def fit_scale(runner: Phase3Runner, calibration_ids: list[int]) -> ScaleFit:
    cal = runner.cfg.calibration
    lo, hi = runner.cfg.conway_range
    target = float(cal["target_elasticity"])
    method = runner.cfg.raw["elasticity_method"]
    rows = []
    for scale in (float(s) for s in cal["scale_grid"]):
        recs = runner.elasticity_records(calibration_ids, "llm", runner.cfg.fit_seed, scale)
        e = pooled_elasticity(recs, "total", method)
        rows.append(
            {
                "wtp_scale": scale,
                "elasticity_total": e,
                "elasticity_flexible": pooled_elasticity(recs, "flexible", method),
                "in_range": lo <= e <= hi,
                "distance_to_target": abs(e - target) if e == e else float("inf"),
            }
        )
    grid = pd.DataFrame(rows)
    pool = grid[grid["in_range"]] if grid["in_range"].any() else grid
    best = pool.sort_values(["distance_to_target", "wtp_scale"]).iloc[0]
    return ScaleFit(
        scale=float(best["wtp_scale"]),
        elasticity=float(best["elasticity_total"]),
        in_range=bool(best["in_range"]),
        grid=grid,
    )


# --------------------------------------------------------------------------
# Discriminating scenario (BUILD_SPEC §12)
# --------------------------------------------------------------------------


def scenario_verdict(run: DayRun, spike: tuple[int, int], deadline_step: int, variant: str) -> dict:
    """Did the washer do what the scenario logic says a coherent agent does?"""
    steps = run.result.steps
    on = [s.t for s in steps if s.washer_on]
    ran_in_spike = any(spike[0] <= t < spike[1] for t in on)
    done_by = run.result.washer_completed_by_step
    on_time = done_by is not None and done_by < deadline_step
    if variant == "slack_remains":
        expected = "defer past the spike, finish on time"
        coherent = on_time and not ran_in_spike
    else:
        expected = "run through the spike, finish on time"
        coherent = on_time
    return {
        "household_id": run.household_id,
        "agent": run.agent,
        "seed": run.seed,
        "variant": variant,
        "washer_start": steps[on[0]].clock if on else None,
        "ran_in_spike": ran_in_spike,
        "finished_on_time": on_time,
        "expected": expected,
        "coherent": coherent,
    }


def run_scenario(
    runner: Phase3Runner,
    household_ids: list[int],
    wtp_scale: float,
    seeds_by_agent: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Each agent at its own seeds (default: the primary seed for everyone)."""
    seeds_by_agent = seeds_by_agent or {a: [runner.cfg.primary_seed] for a in AGENTS}
    sc = runner.cfg.raw["scenario"]
    spike = (_clock_to_step(sc["spike"]["start"]), _clock_to_step(sc["spike"]["end"]))
    date = runner.cfg.dates[0]
    rows = []
    for variant, v in sc["variants"].items():
        deadline = _clock_to_step(v["washer_deadline"])
        for hid in household_ids:
            for agent in AGENTS:
                for seed in seeds_by_agent[agent]:
                    run = runner.run_day(hid, agent, date, variant, seed, wtp_scale)
                    rows.append(scenario_verdict(run, spike, deadline, variant))
    return pd.DataFrame(rows)
