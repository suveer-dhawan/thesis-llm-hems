"""Tests for `hems.environment.tariff`.

Responsibility (BUILD_SPEC.md §9): unit-test `tariff.rate_at(t)`
against the configured peak/shoulder/off-peak table and CPP event
schedule. No tariff logic exists yet -- placeholder only.
"""

import pytest


@pytest.mark.skip(reason="tariff.py not implemented yet")
def test_rate_at_placeholder():
    pass
