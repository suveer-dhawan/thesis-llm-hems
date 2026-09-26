"""PAR, peak kW, cost, load curves.

Responsibility (BUILD_SPEC.md §11): Peak-to-Average Ratio, peak demand (kW)
and time-of-peak, total cost ($) per household, and the per-run summary
dict written to `summary.json`. The four-way overlay and the
behavioural-coherence check (§12) build on top of this once the LLM agent
exists (Phase 2/3) -- Phase 1 only has base/MILP/noisy-MILP to compare.
"""

from __future__ import annotations

from hems.environment.simulator import RunResult


def peak_to_average_ratio(load_kwh: list[float]) -> float:
    if not load_kwh:
        return float("nan")
    avg = sum(load_kwh) / len(load_kwh)
    return max(load_kwh) / avg if avg > 0 else float("nan")


def peak_kw(load_kwh: list[float]) -> float:
    """Half-hour kWh cells -> instantaneous kW (BUILD_SPEC.md unit convention)."""
    return max(load_kwh) * 2 if load_kwh else 0.0


def time_of_peak_clock(result: RunResult, load_kwh: list[float] | None = None) -> str:
    load = load_kwh if load_kwh is not None else result.total_load_curve()
    idx = max(range(len(load)), key=lambda i: load[i])
    return result.steps[idx].clock


def summary_dict(result: RunResult) -> dict:
    total_curve = result.total_load_curve()
    return {
        "household_id": result.household_id,
        "date": result.date.isoformat(),
        "agent": result.agent_name,
        "base_kwh_total": round(result.base_kwh_total, 4),
        "flexible_kwh_total": round(result.flexible_kwh_total, 4),
        "total_kwh_total": round(result.total_kwh_total, 4),
        "flexible_share_pct": round(result.flexible_share * 100, 2),
        "base_cost": round(result.total_cost - result.flexible_cost, 4),
        "flexible_cost": round(result.flexible_cost, 4),
        "total_cost": round(result.total_cost, 4),
        "peak_kw": round(peak_kw(total_curve), 3),
        "time_of_peak": time_of_peak_clock(result, total_curve),
        "par": round(peak_to_average_ratio(total_curve), 3),
        "comfort_violation_steps": result.comfort_violation_steps,
        "hvac_band_c": [result.hvac_band_lo_c, result.hvac_band_hi_c],
        "washer_completed": result.washer_completed,
        "washer_completed_by_step": result.washer_completed_by_step,
        "washer_deadline_step": result.washer_deadline_step,
    }


def curve_metrics(load_kwh: list[float], clocks: list[str]) -> dict:
    """PAR, peak kW, and time-of-peak for any curve (e.g. a population sum)."""
    idx = max(range(len(load_kwh)), key=lambda i: load_kwh[i])
    return {
        "par": round(peak_to_average_ratio(load_kwh), 3),
        "peak_kw": round(peak_kw(load_kwh), 3),
        "time_of_peak": clocks[idx],
    }


def degree_hours_above_ceiling(result: RunResult) -> float:
    """Sum over steps of max(0, indoor - band ceiling) x step length (degC.h).

    Indoor temp is the logged end-of-step value; the ceiling is the run's own
    (for noisy-MILP, its jittered band).
    """
    dt_h = 24 / len(result.steps)
    return sum(max(0.0, s.indoor_temp_c - result.hvac_band_hi_c) for s in result.steps) * dt_h


def max_exceedance_c(result: RunResult) -> float:
    """Largest indoor overshoot above the band ceiling over the day (degC, 0 if never)."""
    return max(0.0, max(s.indoor_temp_c for s in result.steps) - result.hvac_band_hi_c)


def rebound_peak_kw(load_kwh: list[float], start_step: int, n_steps: int) -> float:
    """Peak kW in the `n_steps` after an event ends (`start_step` = first post-event step).

    Compared CPP day vs baseline day, this is the rebound: deferred load
    landing just after the price spike (BUILD_SPEC.md §13 Phase 4).
    """
    window = load_kwh[start_step : start_step + n_steps]
    return peak_kw(window) if window else float("nan")
