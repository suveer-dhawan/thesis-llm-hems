"""Tests for `hems.environment.tariff`.

Responsibility (BUILD_SPEC.md §9): unit-test `tariff.rate_at(t, date)`
against the configured peak/shoulder/off-peak table (including the
summer/winter-only peak season) and the CPP event schedule.
"""

from __future__ import annotations

import datetime as dt

import pytest

from hems.environment.tariff import Tariff, step_to_clock

CONFIG_PATH = "config/tariff.yaml"

SUMMER_WEEKDAY = dt.date(2013, 1, 14)  # January -> summer peak season, no CPP event
SHOULDER_SEASON_DAY = dt.date(2013, 4, 15)  # April -> no peak season at all
SATURDAY = dt.date(2013, 1, 19)  # weekend, still summer peak season
CPP_DAY = dt.date(2013, 1, 18)  # the Phase 1 demo CPP event date, also summer


def clock_to_step(clock: str) -> int:
    h, m = clock.split(":")
    return int(h) * 2 + (1 if m == "30" else 0)


@pytest.fixture
def tariff() -> Tariff:
    return Tariff.from_yaml(CONFIG_PATH)


def test_step_to_clock_roundtrip():
    assert step_to_clock(0) == "00:00"
    assert step_to_clock(1) == "00:30"
    assert step_to_clock(28) == "14:00"
    assert step_to_clock(47) == "23:30"


def test_off_peak_overnight(tariff: Tariff):
    result = tariff.rate_at(clock_to_step("02:00"), SUMMER_WEEKDAY)
    assert result.rate == pytest.approx(0.22)
    assert result.is_cpp is False


def test_peak_summer_afternoon(tariff: Tariff):
    result = tariff.rate_at(clock_to_step("18:00"), SUMMER_WEEKDAY)
    assert result.rate == pytest.approx(0.55)
    assert result.is_cpp is False


def test_peak_window_is_half_open(tariff: Tariff):
    # 15:00 is in the peak window, 21:00 (the end) is not.
    assert tariff.rate_at(clock_to_step("15:00"), SUMMER_WEEKDAY).rate == pytest.approx(0.55)
    assert tariff.rate_at(clock_to_step("21:00"), SUMMER_WEEKDAY).rate != pytest.approx(0.55)


def test_shoulder_morning(tariff: Tariff):
    result = tariff.rate_at(clock_to_step("10:00"), SUMMER_WEEKDAY)
    assert result.rate == pytest.approx(0.28)
    assert result.is_cpp is False


def test_shoulder_early_evening(tariff: Tariff):
    result = tariff.rate_at(clock_to_step("21:30"), SUMMER_WEEKDAY)
    assert result.rate == pytest.approx(0.28)


def test_peak_applies_on_weekends_too(tariff: Tariff):
    # Current Ausgrid-style TOU periods apply every day, not weekday-only.
    result = tariff.rate_at(clock_to_step("18:00"), SATURDAY)
    assert result.rate == pytest.approx(0.55)


def test_no_peak_season_falls_back_to_shoulder(tariff: Tariff):
    # April is outside both summer (Nov-Mar) and winter (Jun-Aug) peak
    # months, so 15:00-21:00 has no matching window -> shoulder default.
    result = tariff.rate_at(clock_to_step("18:00"), SHOULDER_SEASON_DAY)
    assert result.rate == pytest.approx(0.28)
    assert result.is_cpp is False


def test_cpp_event_overrides_peak(tariff: Tariff):
    # 17:00-20:00 on the CPP day overrides what would otherwise be peak.
    result = tariff.rate_at(clock_to_step("18:00"), CPP_DAY)
    assert result.rate == pytest.approx(1.20)  # moderate severity
    assert result.is_cpp is True


def test_cpp_event_is_half_open(tariff: Tariff):
    assert tariff.rate_at(clock_to_step("17:00"), CPP_DAY).is_cpp is True
    assert tariff.rate_at(clock_to_step("20:00"), CPP_DAY).is_cpp is False


def test_cpp_event_scoped_to_its_date(tariff: Tariff):
    # Same clock time, non-CPP date -> normal peak schedule applies.
    result = tariff.rate_at(clock_to_step("18:00"), SUMMER_WEEKDAY)
    assert result.is_cpp is False
    assert result.rate == pytest.approx(0.55)


def test_overnight_boundary_wraps_midnight(tariff: Tariff):
    # off-peak window is 22:00 -> 07:00; 06:59 is inside, 07:00 is not.
    assert tariff.rate_at(clock_to_step("06:30"), SUMMER_WEEKDAY).rate == pytest.approx(0.22)
    assert tariff.rate_at(clock_to_step("07:00"), SUMMER_WEEKDAY).rate != pytest.approx(0.22)


def test_severe_cpp_rate_is_configured_and_higher(tariff: Tariff):
    # Not wired to a default event, but must be available for the
    # sensitivity variant (BUILD_SPEC.md CONFIRM).
    assert tariff.cpp_severity_rates["severe"] == pytest.approx(2.50)
    assert tariff.cpp_severity_rates["severe"] > tariff.cpp_severity_rates["moderate"]


def test_demo_day_is_within_summer_peak_season(tariff: Tariff):
    # Confirms the Phase 1 hot-day demo (January) actually exercises the
    # summer peak rate outside the CPP window, not the shoulder default.
    result = tariff.rate_at(clock_to_step("15:30"), CPP_DAY)
    assert result.rate == pytest.approx(0.55)
    assert result.is_cpp is False
