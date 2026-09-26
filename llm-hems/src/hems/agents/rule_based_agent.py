"""Rule-based satisficer: the LLM agent with the LLM replaced by a formula.

A no-LLM control for the thesis agent. It emits a WTP from a fixed rule and
routes it through the *same* Python halt (`llm_agent.satisficing_halt`:
`U = wtp_scale * wtp_factor * wtp - cost_now`, run iff `U >= 0`) and the same
forced-washer handling (`llm_agent.forced_washer_outcome`), so the only thing
that differs from `LLMAgent` is where the WTP comes from. If this reproduces
the LLM's staggering / bounded elasticity, the LLM is not what produces them.

  HVAC   : WTP = k[comfort_orientation] * max(0, T_in - band_ceiling)
           band_ceiling = setpoint + tolerance_band from the context, so a
           fixed-band run moves the ceiling the rule sees.
  washer : WTP = w_max * urgency,  urgency = clamp(1 - slack / slack_0, 0, 1)
           slack_0 = slack at the release step (deadline - release - duration),
           so urgency rises linearly from 0 at release to 1 at slack 0 (the
           last start that still meets the deadline).

`k` and `w_max` are level-matched to the LLM by a least-squares fit through
the origin (`fit_through_origin`) of raw LLM WTP on the rule's regressor --
see `experiments/fit_rule_based.py` for the data and the fit, and
`experiment.yaml` -> `rule_based:` for the frozen values.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from hems.agents.base import Decision, DecisionContext
from hems.agents.llm_agent import forced_washer_outcome, satisficing_halt

DEFAULT_EXPERIMENT_CONFIG = "config/experiment.yaml"


def hvac_over_ceiling_c(appliance: dict) -> float:
    """Degrees C above the comfort-band ceiling (0 inside or below the band)."""
    ceiling = appliance["setpoint_c"] + appliance["tolerance_band_c"]
    return max(0.0, appliance["indoor_temp_c"] - ceiling)


def washer_urgency(appliance: dict) -> float:
    """0 at the release step, rising linearly to 1 at zero slack (clamped to [0, 1])."""
    slack_0 = appliance["deadline_step"] - appliance["released_step"] - appliance["duration_steps"]
    if slack_0 <= 0:
        return 1.0
    return min(1.0, max(0.0, 1.0 - appliance["slack_steps"] / slack_0))


def fit_through_origin(x, y) -> float:
    """Least-squares slope of `y ~ b * x` (no intercept): `sum(x*y) / sum(x^2)`."""
    x, y = list(map(float, x)), list(map(float, y))
    sxx = sum(v * v for v in x)
    if sxx == 0:
        raise ValueError("regressor is all zero -- slope undefined")
    return sum(a * b for a, b in zip(x, y)) / sxx


@dataclass(frozen=True)
class RuleBasedParams:
    k_by_orientation: dict[str, float]  # $ per degC over the ceiling
    w_max: float  # $ at zero slack

    @classmethod
    def from_yaml(
        cls, path: str | Path = DEFAULT_EXPERIMENT_CONFIG, band_mode: str = "persona"
    ) -> "RuleBasedParams":
        """`band_mode` picks the k fitted on prompts at that band: `persona` ->
        `rule_based.hvac_k`, `fixed` -> `rule_based.hvac_k_fixed_band`."""
        cfg = yaml.safe_load(Path(path).read_text())["rule_based"]
        keys = {"persona": "hvac_k", "fixed": "hvac_k_fixed_band"}
        if band_mode not in keys:
            raise ValueError(f"unknown band_mode '{band_mode}' (persona | fixed)")
        return cls(
            k_by_orientation={str(o): float(v) for o, v in cfg[keys[band_mode]].items()},
            w_max=float(cfg["washer_w_max"]),
        )


@dataclass
class RuleDecisionLogEntry:
    t: int
    clock: str
    appliance: str
    regressor: float  # over-ceiling degC (HVAC) or urgency 0..1 (washer); NaN if forced
    wtp: float
    cost_now: float
    utility: float
    run: bool
    forced: bool
    wtp_scale: float
    wtp_factor: float


class RuleBasedSatisficerAgent:
    """Formula WTP -> the LLM agent's Python halt. No model, no API."""

    def __init__(self, params: RuleBasedParams, wtp_scale: float = 1.0, wtp_factor: float = 1.0):
        self.params = params
        self.wtp_scale = float(wtp_scale)
        self.wtp_factor = float(wtp_factor)
        self.log: list[RuleDecisionLogEntry] = []

    def wtp(self, ctx: DecisionContext) -> tuple[float, float]:
        """`(regressor, wtp)` for a non-forced decision."""
        a = ctx.appliance
        if a["type"] == "hvac":
            orientation = ctx.household.get("comfort_orientation", "balanced")
            if orientation not in self.params.k_by_orientation:
                raise ValueError(f"no rule-based k for comfort_orientation '{orientation}'")
            x = hvac_over_ceiling_c(a)
            return x, self.params.k_by_orientation[orientation] * x
        if a["type"] == "washer":
            u = washer_urgency(a)
            return u, self.params.w_max * u
        raise ValueError(f"unknown appliance type: {a['type']}")

    def decide(self, ctx: DecisionContext) -> Decision:
        if ctx.appliance["type"] == "washer":
            forced = forced_washer_outcome(ctx)
            if forced is not None:
                run, reason = forced
                self._log(ctx, float("nan"), float("nan"), float("nan"), run, forced=True)
                return Decision(run=run, wtp=None, reason=reason)
        x, wtp = self.wtp(ctx)
        utility, run = satisficing_halt(wtp, ctx.cost_now, self.wtp_scale, self.wtp_factor)
        self._log(ctx, x, wtp, utility, run, forced=False)
        return Decision(run=run, wtp=wtp, reason="rule-based satisficer")

    def name(self) -> str:
        return "rule_based"

    def _log(self, ctx, x, wtp, utility, run, forced) -> None:
        self.log.append(
            RuleDecisionLogEntry(
                t=ctx.t,
                clock=ctx.clock,
                appliance=ctx.appliance["type"],
                regressor=x,
                wtp=wtp,
                cost_now=ctx.cost_now,
                utility=utility,
                run=run,
                forced=forced,
                wtp_scale=self.wtp_scale,
                wtp_factor=self.wtp_factor,
            )
        )

    def log_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(e) for e in self.log])
