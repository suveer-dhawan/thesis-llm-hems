"""Tests for `hems.environment.simulator`.

Responsibility (BUILD_SPEC.md §4): verify the 30-min orchestration
loop produces a correct `RunResult` (load curve, total cost, decision
log) given a stub agent. No simulator logic exists yet -- placeholder
only.
"""

import pytest


@pytest.mark.skip(reason="simulator.py not implemented yet")
def test_simulator_loop_placeholder():
    pass
