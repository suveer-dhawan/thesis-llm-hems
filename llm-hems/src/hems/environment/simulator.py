"""The 30-min orchestration loop.

Responsibility (BUILD_SPEC.md §4): runs one household over a horizon
(default one day = 48 steps), stepping through tariff state and each
controllable appliance, calling `agent.decide(ctx)`, applying the run/defer
decision to update physical state and cost, logging every decision, and
accumulating the aggregate load curve into a `RunResult`.

Units and the base/flexible split follow BUILD_SPEC.md "Base load vs
flexible layer": the household's SGSC curve is the fixed, non-controllable
`base` load; HVAC + washer form the `flexible` layer added on top; `total =
base + flexible` at every step. `cost = energy_kwh * rate` applies to all
consumption, base and flexible alike, since a real bill doesn't distinguish
them -- only the *flexible* portion is within the agent's control.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass

import pandas as pd

from hems.agents.base import DecisionAgent, DecisionContext
from hems.environment.appliances import (
    HVACParams,
    WasherParams,
    hvac_in_band,
    hvac_step,
    synthetic_outdoor_temp_c,
    washer_step,
)
from hems.environment.household import Household
from hems.environment.tariff import Tariff, step_to_clock

STEPS_PER_DAY = 48


@dataclass
class StepLog:
    t: int
    clock: str
    rate: float
    is_cpp: bool
    base_kwh: float
    hvac_on: bool
    hvac_energy_kwh: float
    hvac_cost: float
    hvac_wtp: float | None
    indoor_temp_c: float
    outdoor_temp_c: float
    in_band: bool
    washer_on: bool
    washer_energy_kwh: float
    washer_cost: float
    washer_wtp: float | None
    washer_steps_completed: int
    flexible_kwh: float
    total_kwh: float
    total_cost: float


@dataclass
class RunResult:
    household_id: int
    date: dt.date
    agent_name: str
    steps: list[StepLog]
    hvac_band_lo_c: float
    hvac_band_hi_c: float
    washer_duration_steps: int
    washer_deadline_step: int

    @property
    def base_kwh_total(self) -> float:
        return sum(s.base_kwh for s in self.steps)

    @property
    def flexible_kwh_total(self) -> float:
        return sum(s.flexible_kwh for s in self.steps)

    @property
    def total_kwh_total(self) -> float:
        return sum(s.total_kwh for s in self.steps)

    @property
    def total_cost(self) -> float:
        return sum(s.total_cost for s in self.steps)

    @property
    def flexible_cost(self) -> float:
        return sum(s.hvac_cost + s.washer_cost for s in self.steps)

    @property
    def flexible_share(self) -> float:
        total = self.total_kwh_total
        return self.flexible_kwh_total / total if total > 0 else 0.0

    @property
    def comfort_violation_steps(self) -> int:
        return sum(1 for s in self.steps if not s.in_band)

    @property
    def washer_completed(self) -> bool:
        return self.steps[-1].washer_steps_completed >= self.washer_duration_steps

    @property
    def washer_completed_by_step(self) -> int | None:
        for s in self.steps:
            if s.washer_steps_completed >= self.washer_duration_steps:
                return s.t
        return None

    def base_load_curve(self) -> list[float]:
        return [s.base_kwh for s in self.steps]

    def flexible_load_curve(self) -> list[float]:
        return [s.flexible_kwh for s in self.steps]

    def total_load_curve(self) -> list[float]:
        return [s.total_kwh for s in self.steps]

    def to_dataframe(self) -> pd.DataFrame:
        """The full per-step decision log, e.g. for a `results/*/decision_log.parquet`."""
        return pd.DataFrame([dataclasses.asdict(s) for s in self.steps])


def run_simulation(
    household: Household,
    tariff: Tariff,
    hvac_params: HVACParams,
    washer_params: WasherParams,
    agent: DecisionAgent,
    date: dt.date,
    day_index: int = 0,
    initial_indoor_temp_c: float | None = None,
) -> RunResult:
    base_curve = household.base_kwh_for_date(date)
    state = household.new_state(
        initial_indoor_temp_c if initial_indoor_temp_c is not None else hvac_params.setpoint_c
    )
    household_ctx = {
        "thermal_sensitivity": household.params.thermal_sensitivity,
        "price_sensitivity": household.params.price_sensitivity,
        "comfort_orientation": household.params.comfort_orientation,
    }

    steps: list[StepLog] = []
    for t in range(STEPS_PER_DAY):
        clock = step_to_clock(t)
        rate_result = tariff.rate_at(t, date)
        outdoor_temp_c = synthetic_outdoor_temp_c(
            t, hvac_params.outdoor_min_c, hvac_params.outdoor_max_c, hvac_params.outdoor_peak_hour
        )

        hvac_ctx = DecisionContext(
            t=t,
            clock=clock,
            day_index=day_index,
            appliance={
                "type": "hvac",
                "indoor_temp_c": state.indoor_temp_c,
                "outdoor_temp_c": outdoor_temp_c,
                "setpoint_c": hvac_params.setpoint_c,
                "tolerance_band_c": hvac_params.tolerance_band_c,
                "rated_kw": hvac_params.rated_kw,
            },
            cost_now=hvac_params.rated_kw * hvac_params.dt_hours * rate_result.rate,
            rate=rate_result.rate,
            is_cpp=rate_result.is_cpp,
            household=household_ctx,
        )
        hvac_decision = agent.decide(hvac_ctx)
        new_indoor_temp, hvac_energy = hvac_step(
            hvac_params, state.indoor_temp_c, outdoor_temp_c, hvac_decision.run
        )
        state.indoor_temp_c = new_indoor_temp
        hvac_cost = hvac_energy * rate_result.rate

        # Slack = steps_until_deadline - steps_remaining, computed once here
        # (design principle §1: the LLM never does arithmetic) so the
        # shiftable prompt can state urgency explicitly instead of asking
        # the model to derive it from two separate numbers -- see
        # BUILD_SPEC.md §8 and `agents/llm_agent.py._build_washer_prompt`.
        steps_remaining = washer_params.duration_steps - state.washer.steps_completed
        steps_until_deadline = washer_params.deadline_step - t
        slack_steps = steps_until_deadline - steps_remaining

        washer_ctx = DecisionContext(
            t=t,
            clock=clock,
            day_index=day_index,
            appliance={
                "type": "washer",
                "steps_completed": state.washer.steps_completed,
                "duration_steps": washer_params.duration_steps,
                "released_step": washer_params.released_step,
                "deadline_step": washer_params.deadline_step,
                "slack_steps": slack_steps,
                "interruptible": False,
                "rated_kw": washer_params.power_kw,
            },
            cost_now=washer_params.power_kw * washer_params.dt_hours * rate_result.rate,
            rate=rate_result.rate,
            is_cpp=rate_result.is_cpp,
            household=household_ctx,
        )
        washer_decision = agent.decide(washer_ctx)
        # Python enforces the release window regardless of what the agent asks
        # for (design principle §2.1) -- non-interruptibility is enforced
        # inside `washer_step` itself.
        washer_requested = washer_decision.run and t >= washer_params.released_step
        state.washer, washer_energy, washer_on = washer_step(
            washer_params, state.washer, washer_requested
        )
        washer_cost = washer_energy * rate_result.rate

        base_kwh = base_curve[t]
        flexible_kwh = hvac_energy + washer_energy
        total_kwh = base_kwh + flexible_kwh
        total_cost = total_kwh * rate_result.rate

        steps.append(
            StepLog(
                t=t,
                clock=clock,
                rate=rate_result.rate,
                is_cpp=rate_result.is_cpp,
                base_kwh=base_kwh,
                hvac_on=hvac_decision.run,
                hvac_energy_kwh=hvac_energy,
                hvac_cost=hvac_cost,
                hvac_wtp=hvac_decision.wtp,
                indoor_temp_c=new_indoor_temp,
                outdoor_temp_c=outdoor_temp_c,
                in_band=hvac_in_band(hvac_params, new_indoor_temp),
                washer_on=washer_on,
                washer_energy_kwh=washer_energy,
                washer_cost=washer_cost,
                washer_wtp=washer_decision.wtp,
                washer_steps_completed=state.washer.steps_completed,
                flexible_kwh=flexible_kwh,
                total_kwh=total_kwh,
                total_cost=total_cost,
            )
        )

    return RunResult(
        household_id=household.params.household_id,
        date=date,
        agent_name=agent.name(),
        steps=steps,
        hvac_band_lo_c=hvac_params.band_lo_c,
        hvac_band_hi_c=hvac_params.band_hi_c,
        washer_duration_steps=washer_params.duration_steps,
        washer_deadline_step=washer_params.deadline_step,
    )
