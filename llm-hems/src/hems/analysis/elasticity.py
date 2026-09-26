"""Elasticity estimator -- the key metric.

Responsibility (BUILD_SPEC.md §10): short-run own-price arc elasticity of
demand over the CPP-priced window, comparing a baseline day (no CPP) against
a CPP day for the same household:

    elasticity = %dQ / %dP = ((Q_cpp - Q_base) / Q_base) / ((P_cpp - P_base) / P_base)

`Q` is energy in the affected window. Phase 3 runs the baseline and CPP day
on the *same calendar date* with only the tariff differing (see
`Tariff.with_cpp_events`), so base load and weather cancel and `dQ` is
purely the agent's response. `P` is the time-averaged rate over the window.

Quantities are kept per component -- `hvac`, `washer`, `flexible` (= hvac +
washer), and `total` (= base + flexible, the whole-household figure the
success criterion is judged on, §4 "Base load vs flexible layer") -- so the
same pair yields per-appliance and aggregate numbers. Aggregation across
households sums Q before taking the ratio (a pooled arc elasticity, not a
mean of ratios), which keeps near-zero `Q_base` households from dominating.

`method="base"` is the spec formula above (default). `method="midpoint"` is
the symmetric arc (midpoint) formula, reported for robustness.

Conway 2020's Australian short-run range is -0.026 to -0.47 (§10). Runs take
it from `phase3.conway_range` in experiment.yaml; `CONWAY_RANGE` below is only
the default for callers that don't pass one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from hems.environment.simulator import RunResult

CONWAY_RANGE = (-0.47, -0.026)
COMPONENTS = ("hvac", "washer", "flexible", "total")


def arc_elasticity(
    q_base: float, q_cpp: float, p_base: float, p_cpp: float, method: str = "base"
) -> float:
    """%dQ / %dP. NaN when undefined (no baseline demand, or no price change)."""
    if method == "base":
        if q_base == 0 or p_base == 0 or p_cpp == p_base:
            return float("nan")
        return ((q_cpp - q_base) / q_base) / ((p_cpp - p_base) / p_base)
    if method == "midpoint":
        q_mid, p_mid = (q_base + q_cpp) / 2, (p_base + p_cpp) / 2
        if q_mid == 0 or p_mid == 0 or p_cpp == p_base:
            return float("nan")
        return ((q_cpp - q_base) / q_mid) / ((p_cpp - p_base) / p_mid)
    raise ValueError(f"unknown elasticity method '{method}' (base | midpoint)")


def in_conway_range(e: float, conway_range: tuple[float, float] = CONWAY_RANGE) -> bool:
    lo, hi = conway_range
    return not math.isnan(e) and lo <= e <= hi


def cpp_window_steps(result: RunResult) -> list[int]:
    """Steps flagged `is_cpp` on a CPP-day run."""
    steps = [s.t for s in result.steps if s.is_cpp]
    if not steps:
        raise ValueError("run has no CPP steps -- pass the CPP-day result")
    return steps


def window_quantities(result: RunResult, window: list[int]) -> dict[str, float]:
    """Energy (kWh) per component, and mean rate, over `window` steps."""
    by_t = {s.t: s for s in result.steps}
    rows = [by_t[t] for t in window]
    hvac = sum(s.hvac_energy_kwh for s in rows)
    washer = sum(s.washer_energy_kwh for s in rows)
    base = sum(s.base_kwh for s in rows)
    return {
        "hvac": hvac,
        "washer": washer,
        "flexible": hvac + washer,
        "total": base + hvac + washer,
        "price": sum(s.rate for s in rows) / len(rows),
    }


@dataclass
class ElasticityRecord:
    """One household x agent x seed: window quantities for both days."""

    household_id: int
    agent: str
    seed: int
    archetype: str | None
    window_start: str
    window_end: str
    p_base: float
    p_cpp: float
    q_base: dict[str, float]
    q_cpp: dict[str, float]
    date: str = ""
    wtp_scale: float | None = None

    def elasticity(self, component: str = "total", method: str = "base") -> float:
        return arc_elasticity(
            self.q_base[component], self.q_cpp[component], self.p_base, self.p_cpp, method
        )

    def to_row(self, method: str = "base") -> dict:
        row = {k: v for k, v in asdict(self).items() if k not in ("q_base", "q_cpp")}
        for c in COMPONENTS:
            row[f"q_base_{c}"] = self.q_base[c]
            row[f"q_cpp_{c}"] = self.q_cpp[c]
            row[f"elasticity_{c}"] = self.elasticity(c, method)
        return row


def household_elasticity(
    base_result: RunResult,
    cpp_result: RunResult,
    seed: int,
    archetype: str | None = None,
) -> ElasticityRecord:
    """Pair a baseline-day and CPP-day run of the same household and agent."""
    if base_result.household_id != cpp_result.household_id:
        raise ValueError("baseline and CPP runs are for different households")
    window = cpp_window_steps(cpp_result)
    qb = window_quantities(base_result, window)
    qc = window_quantities(cpp_result, window)
    by_t = {s.t: s for s in cpp_result.steps}
    end_minutes = (window[-1] + 1) * 30
    return ElasticityRecord(
        household_id=cpp_result.household_id,
        agent=cpp_result.agent_name,
        seed=seed,
        archetype=archetype,
        window_start=by_t[window[0]].clock,
        window_end=f"{end_minutes // 60:02d}:{end_minutes % 60:02d}",
        p_base=qb.pop("price"),
        p_cpp=qc.pop("price"),
        q_base=qb,
        q_cpp=qc,
        date=cpp_result.date.isoformat(),
    )


def pooled_elasticity(
    records: list[ElasticityRecord], component: str = "total", method: str = "base"
) -> float:
    """Aggregate arc elasticity: sum Q over households, then take the ratio.

    Price is the mean window price across records -- identical for every
    household in Phase 3, since all share one tariff and one CPP window.
    """
    if not records:
        return float("nan")
    qb = sum(r.q_base[component] for r in records)
    qc = sum(r.q_cpp[component] for r in records)
    pb = sum(r.p_base for r in records) / len(records)
    pc = sum(r.p_cpp for r in records) / len(records)
    return arc_elasticity(qb, qc, pb, pc, method)


def elasticity_table(
    records: list[ElasticityRecord],
    group_by: str | None = None,
    method: str = "base",
    conway_range: tuple[float, float] = CONWAY_RANGE,
) -> pd.DataFrame:
    """Pooled elasticity per component, optionally per `group_by` value.

    `group_by` is any `ElasticityRecord` attribute (`agent`, `archetype`,
    `household_id`, `seed`); `None` pools everything into one row.
    """
    if group_by is None:
        groups = {"all": records}
    else:
        groups: dict = {}
        for r in records:
            groups.setdefault(getattr(r, group_by), []).append(r)
    rows = []
    for key, recs in groups.items():
        row = {group_by or "group": key, "n": len(recs)}
        for c in COMPONENTS:
            row[f"elasticity_{c}"] = pooled_elasticity(recs, c, method)
        row["in_conway_total"] = in_conway_range(row["elasticity_total"], conway_range)
        rows.append(row)
    return pd.DataFrame(rows)


def records_to_frame(records: list[ElasticityRecord], method: str = "base") -> pd.DataFrame:
    return pd.DataFrame([r.to_row(method) for r in records])
