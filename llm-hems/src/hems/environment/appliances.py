"""HVAC thermal model + shiftable load.

Responsibility (BUILD_SPEC.md §7): the two controllable appliance models for
the core thesis -- HVAC (first-order RC thermal model, hard comfort band)
and the shiftable washing-machine load (non-interruptible, released/deadline
window).

These are the *flexible layer* added on top of a household's fixed SGSC base
load -- see BUILD_SPEC.md "Base load vs flexible layer" for the design
rationale and unit convention (kW = kWh-per-half-hour-cell * 2;
cost = energy_kwh * rate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

STEPS_PER_DAY = 48


# --------------------------------------------------------------------------
# HVAC (first-order RC thermal model)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HVACParams:
    rated_kw: float
    cop: float
    r: float  # thermal resistance, degC per kW
    c: float  # thermal capacitance, kWh per degC (so r * c has units of hours)
    setpoint_c: float
    tolerance_band_c: float
    outdoor_min_c: float
    outdoor_max_c: float
    outdoor_peak_hour: float = 15.0  # hour-of-day (0-24) of the daily temperature max
    dt_hours: float = 0.5
    # Indoor temp at t=0; None = start at the setpoint. Set per household by
    # persona.apply_thermal_heterogeneity, read by the simulator and the MILP.
    initial_indoor_temp_c: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HVACParams":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        h = cfg["hvac"]
        return cls(
            rated_kw=float(h["rated_kw"]),
            cop=float(h["cop"]),
            r=float(h["r"]),
            c=float(h["c"]),
            setpoint_c=float(h["setpoint_c"]),
            tolerance_band_c=float(h["tolerance_band_c"]),
            outdoor_min_c=float(h["outdoor_temp_min_c"]),
            outdoor_max_c=float(h["outdoor_temp_max_c"]),
            outdoor_peak_hour=float(h.get("outdoor_temp_peak_hour", 15.0)),
        )

    @property
    def start_temp_c(self) -> float:
        """Indoor temp at t=0: the household's own initial temp, else the setpoint."""
        return self.setpoint_c if self.initial_indoor_temp_c is None else self.initial_indoor_temp_c

    @property
    def band_lo_c(self) -> float:
        return self.setpoint_c - self.tolerance_band_c

    @property
    def band_hi_c(self) -> float:
        return self.setpoint_c + self.tolerance_band_c


def synthetic_outdoor_temp_c(
    t: int,
    min_c: float,
    max_c: float,
    peak_hour: float = 15.0,
    steps_per_day: int = STEPS_PER_DAY,
) -> float:
    """Synthetic diurnal outdoor temperature: a single cosine cycle per day.

    SGSC has no weather data, so this is a config-driven placeholder curve
    (BUILD_SPEC.md §7, CONFIRM) -- minimum overnight (12h before `peak_hour`),
    maximum at `peak_hour`.
    """
    hour = (t % steps_per_day) * (24.0 / steps_per_day)
    mean = (min_c + max_c) / 2.0
    amplitude = (max_c - min_c) / 2.0
    return mean + amplitude * math.cos(2.0 * math.pi * (hour - peak_hour) / 24.0)


def hvac_step(
    params: HVACParams, indoor_temp_c: float, outdoor_temp_c: float, on: bool
) -> tuple[float, float]:
    """One 30-min RC-model update.

    `T_in[t+1] = T_in[t] + (dt/(R*C)) * (T_out[t] - T_in[t])
                 - (COP * P_hvac * on * dt) / C`  (BUILD_SPEC.md §7)

    Returns `(new_indoor_temp_c, energy_kwh)`. Electrical energy drawn is
    `rated_kw * dt` when on regardless of COP -- COP scales the *cooling
    effect* delivered per unit of electrical energy, not the electrical draw
    itself.
    """
    delta = (params.dt_hours / (params.r * params.c)) * (outdoor_temp_c - indoor_temp_c)
    energy_kwh = 0.0
    if on:
        delta -= (params.cop * params.rated_kw * params.dt_hours) / params.c
        energy_kwh = params.rated_kw * params.dt_hours
    return indoor_temp_c + delta, energy_kwh


def hvac_in_band(params: HVACParams, indoor_temp_c: float) -> bool:
    return params.band_lo_c <= indoor_temp_c <= params.band_hi_c


# --------------------------------------------------------------------------
# Shiftable load: washing machine (non-interruptible, 4-step run)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WasherParams:
    duration_steps: int
    power_kw: float
    released_step: int
    deadline_step: int
    dt_hours: float = 0.5

    @classmethod
    def from_yaml(cls, appliances_path: str | Path, household_path: str | Path) -> "WasherParams":
        with open(appliances_path) as f:
            a_cfg = yaml.safe_load(f)["shiftable_washing_machine"]
        with open(household_path) as f:
            h_cfg = yaml.safe_load(f)

        released_step = a_cfg.get("released_step")
        if released_step is None:
            released_step = _clock_to_step(h_cfg["occupied_hours"]["start"])
        deadline_step = a_cfg.get("deadline_step")
        if deadline_step is None:
            deadline_step = _clock_to_step(h_cfg["deadline"])

        return cls(
            duration_steps=int(a_cfg["duration_steps"]),
            power_kw=float(a_cfg["power_kw"]),
            released_step=int(released_step),
            deadline_step=int(deadline_step),
        )

    @property
    def energy_kwh(self) -> float:
        """Total energy the wash cycle consumes, independent of *when* it runs."""
        return self.power_kw * self.dt_hours * self.duration_steps

    @property
    def latest_start_step(self) -> int:
        """Last step the cycle could start on and still finish by the deadline."""
        return self.deadline_step - self.duration_steps


def _clock_to_step(clock: str) -> int:
    h, m = clock.split(":")
    return int(h) * 2 + (1 if m == "30" else 0)


@dataclass
class WasherState:
    steps_completed: int = 0
    started: bool = False


def washer_done(params: WasherParams, state: WasherState) -> bool:
    return state.steps_completed >= params.duration_steps


def washer_step(
    params: WasherParams, state: WasherState, requested_on: bool
) -> tuple[WasherState, float, bool]:
    """One 30-min washer update.

    Non-interruptible (BUILD_SPEC.md §7): once started, `requested_on=False`
    is ignored until the cycle completes. Returns
    `(new_state, energy_kwh, actually_on)`.
    """
    if washer_done(params, state):
        return state, 0.0, False

    actually_on = requested_on or state.started
    if not actually_on:
        return state, 0.0, False

    new_state = WasherState(steps_completed=state.steps_completed + 1, started=True)
    energy_kwh = params.power_kw * params.dt_hours
    return new_state, energy_kwh, True
