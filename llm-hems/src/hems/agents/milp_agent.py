"""MILP optimal cost-minimiser (hyper-rational baseline).

Responsibility (BUILD_SPEC.md §6.1): PuLP-based (CBC solver) horizon
optimisation minimising total flexible-load cost, subject to shiftable-load
contiguity/deadline constraints and hard HVAC comfort-band constraints
(linearised RC dynamics, so it stays a mixed-integer *linear* program).
Solves once at construction; `decide()` reads back the precomputed
per-step schedule, so the same `simulator.run_simulation` loop drives the
actual physics/accounting for MILP, noisy-MILP, and the LLM agent alike
(apples-to-apples, §6.1).

Comfort floor: the band floor only limits *AC-induced* cooling. With the AC
off the house can drift below `band_lo_c` on its own overnight (outdoor min
22C), which no control can prevent and which used to make the hard band
infeasible. So the floor at step t is `min(band_lo_c, free_drift[t])`, where
`free_drift` is the precomputed AC-off trajectory from the same initial temp
(`free_drift_temps`) -- a constant per step, so the constraint stays linear.
Where the free drift never goes below the floor, the constraint is exactly
the old one. No soft penalties.
"""

from __future__ import annotations

import datetime as dt
import random

import pulp

from hems.agents.base import Decision, DecisionContext
from hems.environment.appliances import HVACParams, WasherParams, synthetic_outdoor_temp_c
from hems.environment.household import Household
from hems.environment.simulator import STEPS_PER_DAY
from hems.environment.tariff import Tariff


def free_drift_temps(hp: HVACParams, initial_temp_c: float) -> list[float]:
    """Indoor temps T[0..48] with the AC off all day (same RC update as the MILP)."""
    gain = hp.dt_hours / (hp.r * hp.c)
    temps = [initial_temp_c]
    for t in range(STEPS_PER_DAY):
        t_out = synthetic_outdoor_temp_c(
            t, hp.outdoor_min_c, hp.outdoor_max_c, hp.outdoor_peak_hour
        )
        temps.append(temps[-1] + gain * (t_out - temps[-1]))
    return temps


class MILPAgent:
    def __init__(
        self,
        household: Household,
        tariff: Tariff,
        hvac_params: HVACParams,
        washer_params: WasherParams,
        date: dt.date,
        initial_indoor_temp_c: float | None = None,
        price_noise_std: float = 0.0,
        rng: random.Random | None = None,
    ):
        self.hvac_params = hvac_params
        self.washer_params = washer_params
        self._initial_temp = (
            initial_indoor_temp_c if initial_indoor_temp_c is not None else hvac_params.start_temp_c
        )
        true_rates = [tariff.rate_at(t, date).rate for t in range(STEPS_PER_DAY)]
        if price_noise_std > 0.0:
            if rng is None:
                raise ValueError("price_noise_std > 0 requires an rng")
            self.perceived_rates = [
                max(0.0, r + rng.gauss(0.0, price_noise_std)) for r in true_rates
            ]
        else:
            self.perceived_rates = true_rates

        self.hvac_on, self.washer_on = self._solve()

    def _solve(self) -> tuple[list[bool], list[bool]]:
        hp, wp = self.hvac_params, self.washer_params
        prob = pulp.LpProblem("hems_milp", pulp.LpMinimize)

        on_hvac = {t: pulp.LpVariable(f"on_hvac_{t}", cat="Binary") for t in range(STEPS_PER_DAY)}

        valid_starts = list(range(wp.released_step, wp.latest_start_step + 1))
        if not valid_starts:
            raise ValueError("washer has no feasible start step within [released, deadline]")
        start_washer = {s: pulp.LpVariable(f"start_washer_{s}", cat="Binary") for s in valid_starts}
        prob += pulp.lpSum(start_washer.values()) == 1

        def washer_on_expr(t: int):
            covering = [s for s in valid_starts if s <= t < s + wp.duration_steps]
            return pulp.lpSum(start_washer[s] for s in covering)

        # Floor per step: the band floor, relaxed to the AC-off drift where the
        # house would fall below it unaided (see module docstring).
        drift = free_drift_temps(hp, self._initial_temp)
        self.floor_c = [min(hp.band_lo_c, drift[t + 1]) for t in range(STEPS_PER_DAY)]

        temp = {t: pulp.LpVariable(f"temp_{t}") for t in range(STEPS_PER_DAY + 1)}
        prob += temp[0] == self._initial_temp

        gain_coeff = hp.dt_hours / (hp.r * hp.c)
        cool_coeff = (hp.cop * hp.rated_kw * hp.dt_hours) / hp.c

        for t in range(STEPS_PER_DAY):
            t_out = synthetic_outdoor_temp_c(
                t, hp.outdoor_min_c, hp.outdoor_max_c, hp.outdoor_peak_hour
            )
            prob += temp[t + 1] == (
                temp[t] + gain_coeff * (t_out - temp[t]) - cool_coeff * on_hvac[t]
            )
            prob += temp[t + 1] >= self.floor_c[t]
            prob += temp[t + 1] <= hp.band_hi_c

        cost_terms = []
        for t in range(STEPS_PER_DAY):
            hvac_cost = hp.rated_kw * hp.dt_hours * self.perceived_rates[t]
            washer_cost = wp.power_kw * wp.dt_hours * self.perceived_rates[t]
            cost_terms.append(hvac_cost * on_hvac[t])
            cost_terms.append(washer_cost * washer_on_expr(t))
        prob += pulp.lpSum(cost_terms)

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"MILP did not solve to optimality: {pulp.LpStatus[status]}")

        hvac_on = [bool(round(on_hvac[t].value())) for t in range(STEPS_PER_DAY)]
        washer_on = []
        for t in range(STEPS_PER_DAY):
            covering = [s for s in valid_starts if s <= t < s + wp.duration_steps]
            washer_on.append(any(round(start_washer[s].value()) == 1 for s in covering))
        return hvac_on, washer_on

    def decide(self, ctx: DecisionContext) -> Decision:
        if ctx.appliance["type"] == "hvac":
            return Decision(run=self.hvac_on[ctx.t], reason="milp schedule")
        return Decision(run=self.washer_on[ctx.t], reason="milp schedule")

    def name(self) -> str:
        return "milp"
