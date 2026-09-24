"""Generic persona schema -> SGSC instantiation (decision memo Phase 2.7).

Responsibility: turn `config/persona.yaml` + the committed household sample
into one persona per household, and hand the simulator the household/HVAC/
washer params that persona implies. Three schema axes:

- **cost_sensitivity** via *energy burden*: `est_annual_cost / income_midpoint`,
  with `est_annual_cost = annual kWh x average tariff rate` (rate derived from
  `config/tariff.yaml`, not typed in). Burden is bucketed L/M/H (terciles by
  default, or fixed thresholds) -> archetype A (low burden) / B / C (high).
  The archetype never reaches the LLM: it acts in Python as a per-household
  WTP factor (`phase3.wtp_scaling.archetype_factors` in experiment.yaml).
- **thermal_tolerance** -> HVAC comfort band (ordered rules on children /
  aircon type) -> `tolerance_band_c` (MILP constraint + prompt state) and
  `comfort_orientation` (the LLM-facing persona sentence).
- **schedule_profile** -> washer release/deadline window, from the SGSC
  load-shape cluster.

`build_persona_table` is pure (DataFrames in, DataFrame out) so the bucketing
is unit-testable; `experiments/build_personas.py` writes its output to
`config/persona_assignment.csv`, and `load_persona_environment` is what the
Phase 3 runner calls to simulate one household under its persona.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from hems.environment.appliances import HVACParams, WasherParams, _clock_to_step
from hems.environment.household import DEFAULT_LOAD_CURVES_PATH, Household, HouseholdParams
from hems.environment.tariff import STEPS_PER_DAY, Tariff

DEFAULT_PERSONA_CONFIG = "config/persona.yaml"
DEFAULT_ASSIGNMENT_PATH = "config/persona_assignment.csv"


def load_persona_config(path: str | Path = DEFAULT_PERSONA_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# cost_sensitivity: energy burden
# --------------------------------------------------------------------------


def _year_dates(year: int) -> list[dt.date]:
    start = dt.date(year, 1, 1)
    return [start + dt.timedelta(days=i) for i in range((dt.date(year + 1, 1, 1) - start).days)]


def time_weighted_average_rate(tariff: Tariff, year: int) -> float:
    """Mean $/kWh over every 30-min step of `year`, CPP events excluded."""
    standing = tariff.with_cpp_events([])
    rates = [standing.rate_at(t, d).rate for d in _year_dates(year) for t in range(STEPS_PER_DAY)]
    return sum(rates) / len(rates)


def load_weighted_average_rates(
    tariff: Tariff, curves: pd.DataFrame, household_ids: list[int]
) -> dict[int, float]:
    """Each household's kWh-weighted mean TOU rate over its own metered year."""
    standing = tariff.with_cpp_events([])
    out: dict[int, float] = {}
    rate_cache: dict[tuple[dt.date, int], float] = {}
    for hid in household_ids:
        mine = curves[curves["customer_ID"] == hid]
        cost = 0.0
        for ts, kwh in zip(mine["timestamp"], mine["kwh"]):
            key = (ts.date(), ts.hour * 2 + (1 if ts.minute >= 30 else 0))
            if key not in rate_cache:
                rate_cache[key] = standing.rate_at(key[1], key[0]).rate
            cost += kwh * rate_cache[key]
        total = mine["kwh"].sum()
        out[hid] = cost / total if total > 0 else float("nan")
    return out


def bucket_burden(
    burden: pd.Series,
    method: str,
    labels: list[str],
    fixed_thresholds: list[float] | None = None,
) -> pd.Series:
    """Burden -> archetype label (labels[0] = lowest burden).

    `terciles` (or any quantile split, one bucket per label) is rank-based, so
    ties can't produce uneven buckets and the split is balanced by
    construction. `fixed` uses the absolute thresholds (fractions of income):
    `burden < t0 -> labels[0]`, `t0 <= burden < t1 -> labels[1]`, ...
    """
    if method == "terciles":
        ranks = burden.rank(method="first")
        return pd.qcut(ranks, q=len(labels), labels=labels).astype(str)
    if method == "fixed":
        if fixed_thresholds is None or len(fixed_thresholds) != len(labels) - 1:
            raise ValueError("fixed bucketing needs len(labels) - 1 thresholds")
        edges = [float("-inf"), *fixed_thresholds, float("inf")]
        return pd.cut(burden, bins=edges, labels=labels, right=False).astype(str)
    raise ValueError(f"unknown bucketing method '{method}' (terciles | fixed)")


# --------------------------------------------------------------------------
# thermal_tolerance and schedule_profile
# --------------------------------------------------------------------------


def _thermal_rule(row: pd.Series, cfg: dict) -> dict | None:
    for rule in cfg["rules"]:
        if all(str(row.get(k)) == str(v) for k, v in rule["when"].items()):
            return rule
    return None


def thermal_band(row: pd.Series, cfg: dict) -> float:
    """First matching rule's band; `default_tolerance_band_c` if none match."""
    rule = _thermal_rule(row, cfg)
    return float(rule["tolerance_band_c"] if rule else cfg["default_tolerance_band_c"])


def comfort_orientation(row: pd.Series, cfg: dict) -> str:
    """First matching rule's orientation; `default_comfort_orientation` if none match."""
    rule = _thermal_rule(row, cfg)
    return str(rule["comfort_orientation"] if rule else cfg["default_comfort_orientation"])


def thermal_descriptor(band_c: float, cfg: dict) -> str:
    mapping = {float(k): v for k, v in cfg["thermal_sensitivity_by_band"].items()}
    if band_c not in mapping:
        raise ValueError(f"no thermal_sensitivity descriptor configured for band {band_c}")
    return mapping[band_c]


def schedule_profile(cluster: int, cfg: dict) -> str:
    by_cluster = {int(k): v for k, v in (cfg.get("profile_by_cluster") or {}).items()}
    return by_cluster.get(int(cluster), cfg["default_profile"])


# --------------------------------------------------------------------------
# The persona table
# --------------------------------------------------------------------------


def build_persona_table(
    manifest: pd.DataFrame,
    annual_kwh: pd.Series,
    avg_rate: float | dict[int, float],
    cfg: dict,
) -> pd.DataFrame:
    """One persona row per manifest household.

    `annual_kwh` is indexed by customer_ID (real metered 2013 totals);
    `avg_rate` is one $/kWh for everyone (time-weighted) or a per-household
    dict (load-weighted).
    """
    cs = cfg["cost_sensitivity"]
    tt = cfg["thermal_tolerance"]
    sp = cfg["schedule_profile"]
    labels = list(cs["archetype_labels"])

    df = manifest.copy()
    df["customer_ID"] = df["customer_ID"].astype(int)
    df["annual_kwh"] = df["customer_ID"].map(annual_kwh).astype(float)
    if df["annual_kwh"].isna().any():
        missing = df.loc[df["annual_kwh"].isna(), "customer_ID"].tolist()
        raise ValueError(f"no annual kWh for households {missing}")
    if isinstance(avg_rate, dict):
        df["avg_rate"] = df["customer_ID"].map(avg_rate)
    else:
        df["avg_rate"] = float(avg_rate)
    df["est_annual_cost"] = df["annual_kwh"] * df["avg_rate"]

    weekly = {k: float(v) for k, v in cs["income_weekly_midpoint_aud"].items()}
    unknown = set(df["income_tier"]) - set(weekly)
    if unknown:
        raise ValueError(f"no income midpoint configured for tiers {sorted(unknown)}")
    df["income_midpoint"] = df["income_tier"].map(weekly) * float(cs["weeks_per_year"])
    df["burden"] = df["est_annual_cost"] / df["income_midpoint"]

    df["archetype_terciles"] = bucket_burden(df["burden"], "terciles", labels)
    df["archetype_fixed"] = bucket_burden(df["burden"], "fixed", labels, cs["fixed_thresholds"])
    df["archetype"] = df["archetype_fixed" if cs["bucketing"] == "fixed" else "archetype_terciles"]
    df["cost_sensitivity"] = df["archetype"].map(dict(zip(labels, ["L", "M", "H"])))
    df["archetype_income_tier"] = df["income_tier"].map(cs["income_tier_archetype"])
    df["price_sensitivity"] = df["archetype"].map(cs["price_sensitivity_by_archetype"])

    df["tolerance_band_c"] = df.apply(lambda r: thermal_band(r, tt), axis=1)
    df["thermal_sensitivity"] = df["tolerance_band_c"].map(lambda b: thermal_descriptor(b, tt))
    df["comfort_orientation"] = df.apply(lambda r: comfort_orientation(r, tt), axis=1)

    df["schedule_profile"] = df["cluster"].map(lambda c: schedule_profile(c, sp))
    df["washer_release"] = df["schedule_profile"].map(lambda p: sp["profiles"][p]["washer_release"])
    df["washer_deadline"] = df["schedule_profile"].map(
        lambda p: sp["profiles"][p]["washer_deadline"]
    )
    return df


ASSIGNMENT_COLUMNS = [
    "customer_ID",
    "cluster",
    "income_tier",
    "annual_kwh",
    "avg_rate",
    "est_annual_cost",
    "income_midpoint",
    "burden",
    "cost_sensitivity",
    "archetype",
    "archetype_income_tier",
    "price_sensitivity",
    "tolerance_band_c",
    "thermal_sensitivity",
    "comfort_orientation",
    "schedule_profile",
    "washer_release",
    "washer_deadline",
]


# --------------------------------------------------------------------------
# Simulator wiring
# --------------------------------------------------------------------------


def load_assignment(path: str | Path = DEFAULT_ASSIGNMENT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def persona_household_params(household_id: int, assignment: pd.DataFrame) -> HouseholdParams:
    rows = assignment[assignment["customer_ID"] == int(household_id)]
    if rows.empty:
        raise ValueError(f"household {household_id} has no persona assignment")
    r = rows.iloc[0]
    return HouseholdParams(
        household_id=int(household_id),
        thermal_sensitivity=str(r["thermal_sensitivity"]),
        price_sensitivity=str(r["price_sensitivity"]),
        deadline=str(r["washer_deadline"]),
        occupied_hours_start=str(r["washer_release"]),
        occupied_hours_end=str(r["washer_deadline"]),
        archetype=str(r["archetype"]),
        tolerance_band_c=float(r["tolerance_band_c"]),
        washer_release=str(r["washer_release"]),
        comfort_orientation=str(r["comfort_orientation"]),
    )


def apply_persona(
    params: HouseholdParams, hvac_params: HVACParams, washer_params: WasherParams
) -> tuple[HVACParams, WasherParams]:
    """Overrides the appliance defaults with whatever the persona sets."""
    if params.tolerance_band_c is not None:
        hvac_params = replace(hvac_params, tolerance_band_c=params.tolerance_band_c)
    if params.washer_release is not None:
        washer_params = replace(
            washer_params,
            released_step=_clock_to_step(params.washer_release),
            deadline_step=_clock_to_step(params.deadline),
        )
    if washer_params.latest_start_step < washer_params.released_step:
        raise ValueError(f"household {params.household_id}: washer window too short for its cycle")
    return hvac_params, washer_params


def load_persona_environment(
    household_id: int,
    assignment: pd.DataFrame,
    appliances_path: str | Path = "config/appliances.yaml",
    household_defaults_path: str | Path = "config/household.yaml",
    load_curves_path: str | Path = DEFAULT_LOAD_CURVES_PATH,
) -> tuple[Household, HVACParams, WasherParams]:
    """Household + persona-adjusted HVAC/washer params for one simulation."""
    params = persona_household_params(household_id, assignment)
    household = Household(params, load_curves_path)
    hvac = HVACParams.from_yaml(appliances_path)
    washer = WasherParams.from_yaml(appliances_path, household_defaults_path)
    hvac, washer = apply_persona(params, hvac, washer)
    return household, hvac, washer
