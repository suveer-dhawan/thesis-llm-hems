"""Hybrid satisficing agent (the thesis).

Responsibility (BUILD_SPEC.md §6.3): per appliance per interval, prompt the
LLM for a structured `wtp` (willingness-to-pay) + one-line `reason` given
comfort state, time, and household persona -- no cost comparison is asked
of the LLM. The prompt is deliberately **price-blind**: it never states the
electricity rate or a CPP flag (see `agents/prompts/*.txt`), so the WTP
reflects comfort/urgency valuation alone. Python still owns `ctx.rate` /
`ctx.is_cpp` / `ctx.cost_now` for accounting -- it computes `U = wtp_scale *
wtp - cost_now` and applies the deterministic halt rule: run iff `U >= 0`.
`wtp_scale` is the single global factor Phase 3 fits on the calibration
subset (`experiments/calibrate.py`) and then freezes; it defaults to 1.0.

Washer steps where Python already fixes the outcome -- before the release
step, mid-cycle (non-interruptible), or after completion -- are decided
without a prompt (`forced=True` in the log): the simulator would override
whatever the model said, so asking costs a call and changes nothing. Every
prompt, wtp, reasoning string, computed cost, utility, run/defer decision,
and provider/model identity is logged (`self.log`, dumped via
`log_to_dataframe()`).

Each response also carries a comfort-state echo (`llm/schemas.py`) that the
model restates alongside its answer; `_detect_misreads` compares it against
the real `DecisionContext` input so a misread (wrong indoor temp, wrong
comfort band, wrong deadline slack, ...) is caught programmatically rather
than by manually reading `reason` strings -- see
`experiments/reliability_battery.py`.
"""

from __future__ import annotations

import dataclasses
import string
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from hems.agents.base import Decision, DecisionContext
from hems.llm.client import LLMClient
from hems.llm.schemas import HVACWTPResponse, WasherWTPResponse

PROMPTS_DIR = Path(__file__).parent / "prompts"

_ECHO_TOLERANCE_C = 0.15  # float round-trip / formatting slack, not a real "close enough"


def _load_template(name: str) -> string.Template:
    return string.Template((PROMPTS_DIR / name).read_text())


_HVAC_TEMPLATE = _load_template("hvac.txt")
_SHIFTABLE_TEMPLATE = _load_template("shiftable.txt")


@dataclass
class LLMDecisionLogEntry:
    t: int
    clock: str
    appliance: str
    prompt: str
    wtp: float  # raw model output, before wtp_scale
    reason: str
    cost_now: float
    utility: float
    run: bool
    cached: bool
    forced: bool
    wtp_scale: float
    dry_run: bool
    provider: str
    model_id: str
    sdk_version: str
    misreads: list[str] = field(default_factory=list)


class LLMAgent:
    """Per appliance, per interval: prompt for wtp, then Python halts.

    `client=None, dry_run=False` builds a real `LLMClient` for `provider`
    ("matilda" | "gemini", see `llm/client.py`) from the environment.
    `dry_run=True` skips the API entirely and uses a deterministic heuristic
    stand-in (§15: pipeline testing without spend) -- everything downstream
    (halt rule, logging, accounting) runs identically either way.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        provider: str = "matilda",
        dry_run: bool = False,
        cache_path: str | Path | None = None,
        wtp_scale: float = 1.0,
    ):
        self._dry_run = dry_run
        self._client = None if dry_run else (client or LLMClient(provider=provider, cache_path=cache_path))
        self.wtp_scale = float(wtp_scale)
        self.log: list[LLMDecisionLogEntry] = []

    @property
    def provider(self) -> str:
        return "dry_run" if self._dry_run else self._client.provider

    @property
    def model_id(self) -> str:
        return "dry_run" if self._dry_run else self._client.model_id

    @property
    def sdk_version(self) -> str:
        return "dry_run" if self._dry_run else self._client.sdk_version

    def decide(self, ctx: DecisionContext) -> Decision:
        appliance_type = ctx.appliance["type"]
        if appliance_type == "washer":
            forced = self._forced_washer_decision(ctx)
            if forced is not None:
                return forced
        if appliance_type == "hvac":
            prompt = self._build_hvac_prompt(ctx)
            schema = HVACWTPResponse
        elif appliance_type == "washer":
            prompt = self._build_washer_prompt(ctx)
            schema = WasherWTPResponse
        else:
            raise ValueError(f"unknown appliance type: {appliance_type}")

        if self._dry_run:
            wtp_response, cached = self._dry_run_response(ctx, schema), False
        else:
            wtp_response, cached = self._client.complete_json(prompt, schema)

        utility = self.wtp_scale * wtp_response.wtp - ctx.cost_now
        run = utility >= 0
        misreads = self._detect_misreads(appliance_type, ctx, wtp_response)

        self.log.append(
            LLMDecisionLogEntry(
                t=ctx.t,
                clock=ctx.clock,
                appliance=appliance_type,
                prompt=prompt,
                wtp=wtp_response.wtp,
                reason=wtp_response.reason,
                cost_now=ctx.cost_now,
                utility=utility,
                run=run,
                cached=cached,
                forced=False,
                wtp_scale=self.wtp_scale,
                dry_run=self._dry_run,
                provider=self.provider,
                model_id=self.model_id,
                sdk_version=self.sdk_version,
                misreads=misreads,
            )
        )
        return Decision(run=run, wtp=wtp_response.wtp, reason=wtp_response.reason)

    def name(self) -> str:
        return "llm"

    def _forced_washer_decision(self, ctx: DecisionContext) -> Decision | None:
        """The washer outcome when Python fixes it regardless of the agent, else None."""
        a = ctx.appliance
        if a["steps_completed"] >= a["duration_steps"]:
            run, reason = False, "forced: cycle already complete"
        elif a["steps_completed"] > 0:
            run, reason = True, "forced: non-interruptible cycle in progress"
        elif ctx.t < a["released_step"]:
            run, reason = False, "forced: before release step"
        else:
            return None
        self.log.append(
            LLMDecisionLogEntry(
                t=ctx.t,
                clock=ctx.clock,
                appliance="washer",
                prompt="",
                wtp=float("nan"),
                reason=reason,
                cost_now=ctx.cost_now,
                utility=float("nan"),
                run=run,
                cached=False,
                forced=True,
                wtp_scale=self.wtp_scale,
                dry_run=self._dry_run,
                provider=self.provider,
                model_id=self.model_id,
                sdk_version=self.sdk_version,
            )
        )
        return Decision(run=run, wtp=None, reason=reason)

    @property
    def prompted_decisions(self) -> int:
        """Decisions that needed a model answer (live, cached, or dry-run stub)."""
        return sum(1 for e in self.log if not e.forced)

    def log_to_dataframe(self) -> pd.DataFrame:
        """The full per-step LLM decision log for `results/*/llm_decision_log.parquet`."""
        return pd.DataFrame([dataclasses.asdict(e) for e in self.log])

    # -- misread detection ---------------------------------------------------

    def _detect_misreads(self, appliance_type: str, ctx: DecisionContext, resp) -> list[str]:
        """Compares the model's state-echo against the real input.

        Returns the list of field names that don't match (empty if the
        response correctly restated everything it was given). Dry-run stub
        responses always echo the true values by construction, so this is a
        no-op there.
        """
        if self._dry_run:
            return []
        a = ctx.appliance
        mismatches: list[str] = []
        if appliance_type == "hvac":
            band_lo = a["setpoint_c"] - a["tolerance_band_c"]
            band_hi = a["setpoint_c"] + a["tolerance_band_c"]
            if abs(resp.echo_indoor_temp_c - a["indoor_temp_c"]) > _ECHO_TOLERANCE_C:
                mismatches.append("indoor_temp_c")
            if abs(resp.echo_band_lo_c - band_lo) > _ECHO_TOLERANCE_C:
                mismatches.append("band_lo_c")
            if abs(resp.echo_band_hi_c - band_hi) > _ECHO_TOLERANCE_C:
                mismatches.append("band_hi_c")
        else:
            steps_until_deadline = a["deadline_step"] - ctx.t
            if resp.echo_steps_until_deadline != steps_until_deadline:
                mismatches.append("steps_until_deadline")
            if resp.echo_slack_steps != a["slack_steps"]:
                mismatches.append("slack_steps")
        return mismatches

    # -- prompt building ----------------------------------------------------

    def _build_hvac_prompt(self, ctx: DecisionContext) -> str:
        a = ctx.appliance
        setpoint = a["setpoint_c"]
        band = a["tolerance_band_c"]
        indoor = a["indoor_temp_c"]
        band_lo = setpoint - band
        band_hi = setpoint + band
        if indoor > band_hi:
            comfort_fact = (
                f"indoor is {indoor:.1f}C; comfort band is {band_lo:.1f}-{band_hi:.1f}C; "
                f"you are {indoor - band_hi:.1f}C OVER the ceiling"
            )
        elif indoor < band_lo:
            comfort_fact = (
                f"indoor is {indoor:.1f}C; comfort band is {band_lo:.1f}-{band_hi:.1f}C; "
                f"you are {band_lo - indoor:.1f}C UNDER the floor"
            )
        else:
            comfort_fact = (
                f"indoor is {indoor:.1f}C; comfort band is {band_lo:.1f}-{band_hi:.1f}C; "
                f"you are WITHIN the band (no violation)"
            )
        return _HVAC_TEMPLATE.substitute(
            thermal_sensitivity=ctx.household.get("thermal_sensitivity", "moderate"),
            price_sensitivity=ctx.household.get("price_sensitivity", "moderate"),
            clock=ctx.clock,
            indoor_temp_c=f"{indoor:.1f}",
            band_lo_c=f"{band_lo:.1f}",
            band_hi_c=f"{band_hi:.1f}",
            comfort_fact=comfort_fact,
        )

    def _build_washer_prompt(self, ctx: DecisionContext) -> str:
        a = ctx.appliance
        steps_remaining = a["duration_steps"] - a["steps_completed"]
        steps_until_deadline = a["deadline_step"] - ctx.t
        return _SHIFTABLE_TEMPLATE.substitute(
            price_sensitivity=ctx.household.get("price_sensitivity", "moderate"),
            clock=ctx.clock,
            steps_completed=a["steps_completed"],
            duration_steps=a["duration_steps"],
            steps_remaining=steps_remaining,
            already_started="yes" if a["steps_completed"] > 0 else "no",
            minutes_until_deadline=max(steps_until_deadline, 0) * 30,
            steps_until_deadline=steps_until_deadline,
            slack_steps=a["slack_steps"],
        )

    # -- dry run --------------------------------------------------------

    # Dry-run stub: persona price_sensitivity -> WTP multiplier, so archetypes
    # differ and the Phase 3 pipeline (calibration, per-archetype tables) has
    # something non-degenerate to chew on. Not a behavioural claim.
    _STUB_SENSITIVITY_MULT = {"low": 1.5, "moderate": 1.0, "high": 0.6}

    def _dry_run_response(self, ctx: DecisionContext, schema):
        """Deterministic stand-in for `--dry-run`: no API call, no spend.

        A cheap heuristic that exercises context-building, the halt rule,
        logging, and accounting end to end -- not meant to resemble real
        WTP behaviour. Echoes the true input values by construction.
        """
        a = ctx.appliance
        mult = self._STUB_SENSITIVITY_MULT.get(ctx.household.get("price_sensitivity"), 1.0)
        if a["type"] == "hvac":
            over_ceiling = a["indoor_temp_c"] - (a["setpoint_c"] + a["tolerance_band_c"])
            wtp = mult * 0.8 * max(0.0, over_ceiling + 0.5)
            return schema(
                wtp=wtp,
                reason="dry-run stub: rises as indoor temp nears/exceeds the ceiling",
                echo_indoor_temp_c=a["indoor_temp_c"],
                echo_band_lo_c=a["setpoint_c"] - a["tolerance_band_c"],
                echo_band_hi_c=a["setpoint_c"] + a["tolerance_band_c"],
            )
        steps_until_deadline = a["deadline_step"] - ctx.t
        slack = a["slack_steps"]
        wtp = 3.0 if slack <= 0 else mult * max(0.0, 0.8 - 0.15 * slack)
        return schema(
            wtp=wtp,
            reason="dry-run stub: proportional to shrinking slack",
            echo_steps_until_deadline=steps_until_deadline,
            echo_slack_steps=slack,
        )
