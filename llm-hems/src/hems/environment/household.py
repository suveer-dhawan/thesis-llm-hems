"""Base load + physical state container.

Responsibility (BUILD_SPEC.md §8): holds the SGSC-derived uncontrollable
base load profile for a household plus the physical state container that
appliances and agents read/write each step, and the interpretable
comfort/persona params from `config/household.yaml` (thermal_sensitivity,
price_sensitivity, deadline, occupied_hours).

See BUILD_SPEC.md "Base load vs flexible layer": the SGSC curve is a
whole-home aggregate that cannot be disaggregated, so it is treated as the
household's fixed, non-controllable base load. HVAC + washer are simulated
as an explicit flexible layer added on top by `environment/simulator.py`,
not subtracted out of this curve.
"""

from __future__ import annotations

import datetime as dt
import functools
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from hems.environment.appliances import WasherState

DEFAULT_LOAD_CURVES_PATH = "data/processed/load_curves.parquet"
STEPS_PER_DAY = 48


@dataclass(frozen=True)
class HouseholdParams:
    household_id: int
    thermal_sensitivity: str
    price_sensitivity: str
    deadline: str  # "HH:MM"
    occupied_hours_start: str
    occupied_hours_end: str
    # Persona-schema fields (config/persona.yaml -> config/persona_assignment.csv,
    # see environment/persona.py). None = fall back to appliances.yaml /
    # household.yaml defaults, which is what the single-household
    # config/household.yaml path does.
    archetype: str | None = None
    tolerance_band_c: float | None = None
    washer_release: str | None = None  # "HH:MM"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HouseholdParams":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            household_id=int(cfg["household_id"]),
            thermal_sensitivity=cfg["thermal_sensitivity"],
            price_sensitivity=cfg["price_sensitivity"],
            deadline=cfg["deadline"],
            occupied_hours_start=cfg["occupied_hours"]["start"],
            occupied_hours_end=cfg["occupied_hours"]["end"],
        )


@functools.lru_cache(maxsize=2)
def _read_load_curves(path: str) -> pd.DataFrame:
    """The processed long-format curves, read once per process (Phase 3 builds
    hundreds of `Household`s from the same 420k-row file)."""
    return pd.read_parquet(path)


@dataclass
class HouseholdState:
    """Mutable physical state carried step-to-step through the simulator loop."""

    indoor_temp_c: float
    washer: WasherState = field(default_factory=WasherState)


class Household:
    """A household's fixed SGSC base load, keyed by `customer_ID`."""

    def __init__(
        self,
        params: HouseholdParams,
        load_curves_path: str | Path = DEFAULT_LOAD_CURVES_PATH,
    ):
        self.params = params
        curves = _read_load_curves(str(load_curves_path))
        mine = curves[curves["customer_ID"] == params.household_id].sort_values("timestamp")
        if mine.empty:
            raise ValueError(f"no load-curve rows for household_id={params.household_id}")
        self._base = mine.set_index("timestamp")["kwh"]

    @classmethod
    def from_config(
        cls,
        household_path: str | Path = "config/household.yaml",
        load_curves_path: str | Path = DEFAULT_LOAD_CURVES_PATH,
    ) -> "Household":
        return cls(HouseholdParams.from_yaml(household_path), load_curves_path)

    def base_kwh_for_date(self, date: dt.date) -> list[float]:
        """The household's fixed base load (kWh per half-hour cell), 48 values for `date`."""
        day_index = pd.date_range(pd.Timestamp(date), periods=STEPS_PER_DAY, freq="30min")
        values = self._base.reindex(day_index)
        if values.isna().any():
            missing = [ts.isoformat() for ts in values[values.isna()].index]
            raise ValueError(f"missing base-load intervals for {date}: {missing}")
        return values.tolist()

    def new_state(self, initial_indoor_temp_c: float) -> HouseholdState:
        return HouseholdState(indoor_temp_c=initial_indoor_temp_c)
