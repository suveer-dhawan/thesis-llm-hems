"""TOU + CPP cost calculator.

Responsibility (BUILD_SPEC.md §9): expose `rate_at(t, date)` returning
`(rate, is_cpp)` for a given 30-min step, driven entirely by
`config/tariff.yaml` (peak/shoulder/off-peak windows and CPP event
schedule).

Units: `rate` is $/kWh. `cost = energy_kwh * rate` -- see BUILD_SPEC.md
"Base load vs flexible layer" for the project-wide unit convention.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import yaml

_DAY_ABBREV = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
STEPS_PER_DAY = 48
MINUTES_PER_STEP = 30


def step_to_clock(t: int) -> str:
    """Step index (0..47, wraps for t outside that range) -> "HH:MM"."""
    minutes = (t % STEPS_PER_DAY) * MINUTES_PER_STEP
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _clock_to_minutes(clock: str) -> int:
    h, m = clock.split(":")
    return int(h) * 60 + int(m)


def _time_in_window(t_minutes: int, start: str, end: str) -> bool:
    """Half-open [start, end); end <= start wraps past midnight."""
    start_m, end_m = _clock_to_minutes(start), _clock_to_minutes(end)
    if start_m == end_m:
        return False
    if start_m < end_m:
        return start_m <= t_minutes < end_m
    return t_minutes >= start_m or t_minutes < end_m


@dataclass(frozen=True)
class RateResult:
    rate: float
    is_cpp: bool


@dataclass(frozen=True)
class Window:
    days: tuple[str, ...]
    start: str
    end: str

    def matches(self, weekday_abbrev: str, t_minutes: int) -> bool:
        return weekday_abbrev in self.days and _time_in_window(t_minutes, self.start, self.end)


@dataclass(frozen=True)
class CppEvent:
    date: dt.date
    start: str
    end: str

    def matches(self, date: dt.date, t_minutes: int) -> bool:
        return date == self.date and _time_in_window(t_minutes, self.start, self.end)


class Tariff:
    """TOU schedule + CPP events, loaded from `config/tariff.yaml`."""

    def __init__(self, config: dict):
        rates = config["rates"]
        self.peak_rate = float(rates["peak"]["rate"])
        self.peak_windows = self._parse_windows(rates["peak"].get("windows", []))
        self.shoulder_rate = float(rates["shoulder"]["rate"])
        self.shoulder_windows = self._parse_windows(rates["shoulder"].get("windows", []))
        self.off_peak_rate = float(rates["off_peak"]["rate"])
        self.off_peak_windows = self._parse_windows(rates["off_peak"].get("windows", []))

        cpp = config.get("cpp", {})
        self.cpp_rate = float(cpp["rate"]) if "rate" in cpp else None
        self.cpp_events = [
            CppEvent(
                date=dt.date.fromisoformat(e["date"]),
                start=e["start"],
                end=e["end"],
            )
            for e in cpp.get("events", [])
        ]

    @staticmethod
    def _parse_windows(windows: list[dict]) -> list[Window]:
        return [
            Window(days=tuple(w["days"]), start=w["start"], end=w["end"]) for w in windows
        ]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Tariff":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def rate_at(self, t: int, date: dt.date) -> RateResult:
        """Rate for 30-min step `t` on calendar `date`.

        CPP events take priority over the normal TOU schedule. Within TOU,
        peak > off-peak > shoulder (shoulder is the default / catch-all).
        """
        t_minutes = (t % STEPS_PER_DAY) * MINUTES_PER_STEP

        for event in self.cpp_events:
            if event.matches(date, t_minutes):
                return RateResult(rate=self.cpp_rate, is_cpp=True)

        weekday = _DAY_ABBREV[date.weekday()]

        if any(w.matches(weekday, t_minutes) for w in self.peak_windows):
            return RateResult(rate=self.peak_rate, is_cpp=False)
        if any(w.matches(weekday, t_minutes) for w in self.off_peak_windows):
            return RateResult(rate=self.off_peak_rate, is_cpp=False)
        if any(w.matches(weekday, t_minutes) for w in self.shoulder_windows):
            return RateResult(rate=self.shoulder_rate, is_cpp=False)
        return RateResult(rate=self.shoulder_rate, is_cpp=False)
