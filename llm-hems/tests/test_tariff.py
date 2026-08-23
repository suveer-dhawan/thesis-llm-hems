"""Tests for `hems.environment.tariff`.

Responsibility (BUILD_SPEC.md §9): unit-test `tariff.rate_at(t, date)`
against the configured peak/shoulder/off-peak table and CPP event schedule.
"""

from __future__ import annotations

import datetime as dt

import pytest

from hems.environment.tariff import Tariff, step_to_clock

CONFIG_PATH = "config/tariff.yaml"

MONDAY = dt.date(2013, 1, 14)  # a plain weekday, no CPP event scheduled
SATURDAY = dt.date(2013, 1, 19)  # weekend
CPP_DAY = dt.date(2013, 1, 18)  # the Phase 1 demo CPP event date


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


def test_off_peak_overnight_weekday(tariff: Tariff):
    result = tariff.rate_at(clock_to_step("02:00"), MONDAY)
    assert result.rate == pytest.approx(0.12)
    assert result.is_cpp is False


def test_peak_weekday_afternoon(tariff: Tariff):
    result = tariff.rate_at(clock_to_step("15:00"), MONDAY)
    assert result.rate == pytest.approx(0.55)
    assert result.is_cpp is False


def test_peak_window_is_half_open(tariff: Tariff):
    # 14:00 is in the peak window, 20:00 (the end) is not.
    assert tariff.rate_at(clock_to_step("14:00"), MONDAY).rate == pytest.approx(0.55)
    assert tariff.rate_at(clock_to_step("20:00"), MONDAY).rate != pytest.approx(0.55)


def test_shoulder_weekday_morning(tariff: Tariff):
    result = tariff.rate_at(clock_to_step("10:00"), MONDAY)
    assert result.rate == pytest.approx(0.22)
    assert result.is_cpp is False


def test_off_peak_overnight_weekend(tariff: Tariff):
    result = tariff.rate_at(clock_to_step("23:00"), SATURDAY)
    assert result.rate == pytest.approx(0.12)


def test_peak_window_weekday_only(tariff: Tariff):
    # Saturday 15:00 has no peak window configured for weekends -> shoulder.
    result = tariff.rate_at(clock_to_step("15:00"), SATURDAY)
    assert result.rate == pytest.approx(0.22)


def test_cpp_event_overrides_peak(tariff: Tariff):
    # 17:00-20:00 on the CPP day overrides what would otherwise be peak.
    result = tariff.rate_at(clock_to_step("18:00"), CPP_DAY)
    assert result.rate == pytest.approx(1.25)
    assert result.is_cpp is True


def test_cpp_event_is_half_open(tariff: Tariff):
    assert tariff.rate_at(clock_to_step("17:00"), CPP_DAY).is_cpp is True
    assert tariff.rate_at(clock_to_step("20:00"), CPP_DAY).is_cpp is False


def test_cpp_event_scoped_to_its_date(tariff: Tariff):
    # Same clock time, non-CPP date -> normal peak schedule applies.
    result = tariff.rate_at(clock_to_step("18:00"), MONDAY)
    assert result.is_cpp is False
    assert result.rate == pytest.approx(0.55)


def test_overnight_boundary_wraps_midnight(tariff: Tariff):
    # off-peak window is 22:00 -> 07:00; 06:59 is inside, 07:00 is not.
    assert tariff.rate_at(clock_to_step("06:30"), MONDAY).rate == pytest.approx(0.12)
    assert tariff.rate_at(clock_to_step("07:00"), MONDAY).rate != pytest.approx(0.12)
