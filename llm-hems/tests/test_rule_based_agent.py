"""RuleBasedSatisficerAgent: formula WTP, the LLM agent's own halt, no model.

Checks the two WTP rules, that the halt and forced-washer paths are the ones
`LLMAgent` uses (same decisions for the same WTP), the least-squares
level-matching helper, the config wiring, and a full persona-household
simulation with no LLM client anywhere.
"""

from __future__ import annotations

import datetime as dt

import pytest

from hems.agents import llm_agent
from hems.agents.base import DecisionContext
from hems.agents.llm_agent import LLMAgent, satisficing_halt
from hems.agents.rule_based_agent import (
    RuleBasedParams,
    RuleBasedSatisficerAgent,
    fit_through_origin,
    hvac_over_ceiling_c,
    washer_urgency,
)
from hems.environment.persona import load_assignment, load_persona_environment
from hems.environment.simulator import run_simulation
from hems.environment.tariff import Tariff

PARAMS = RuleBasedParams(
    k_by_orientation={"comfort_prioritising": 3.0, "balanced": 0.6, "comfort_flexible": 0.5},
    w_max=0.9,
)


def _hvac_ctx(indoor=27.0, band=2.0, rate=0.55, orientation="balanced"):
    return DecisionContext(
        t=34,
        clock="17:00",
        day_index=0,
        appliance={
            "type": "hvac",
            "indoor_temp_c": indoor,
            "outdoor_temp_c": 31.0,
            "setpoint_c": 24.0,
            "tolerance_band_c": band,
            "rated_kw": 2.0,
        },
        cost_now=2.0 * 0.5 * rate,
        rate=rate,
        is_cpp=rate > 0.55,
        household={"comfort_orientation": orientation},
    )


def _washer_ctx(t=36, released=32, deadline=44, duration=4, completed=0, rate=0.55):
    slack = (deadline - t) - (duration - completed)
    return DecisionContext(
        t=t,
        clock="18:00",
        day_index=0,
        appliance={
            "type": "washer",
            "steps_completed": completed,
            "duration_steps": duration,
            "released_step": released,
            "deadline_step": deadline,
            "slack_steps": slack,
            "interruptible": False,
            "rated_kw": 0.8,
        },
        cost_now=0.8 * 0.5 * rate,
        rate=rate,
        is_cpp=rate > 0.55,
        household={"comfort_orientation": "balanced"},
    )


# -- HVAC rule ---------------------------------------------------------------


@pytest.mark.parametrize("indoor", [22.0, 25.0, 26.0])
def test_hvac_wtp_is_zero_at_or_below_ceiling(indoor):
    agent = RuleBasedSatisficerAgent(PARAMS)
    d = agent.decide(_hvac_ctx(indoor=indoor))
    assert d.wtp == 0.0 and d.run is False


@pytest.mark.parametrize("orientation", sorted(PARAMS.k_by_orientation))
def test_hvac_wtp_is_k_times_degrees_over_ceiling(orientation):
    agent = RuleBasedSatisficerAgent(PARAMS)
    d = agent.decide(_hvac_ctx(indoor=27.5, orientation=orientation))
    assert d.wtp == pytest.approx(PARAMS.k_by_orientation[orientation] * 1.5)
    assert agent.log[-1].regressor == pytest.approx(1.5)


def test_hvac_ceiling_follows_the_context_band():
    """A fixed-band run moves the ceiling the rule sees."""
    assert hvac_over_ceiling_c(_hvac_ctx(indoor=26.5, band=1.5).appliance) == pytest.approx(1.0)
    assert hvac_over_ceiling_c(_hvac_ctx(indoor=26.5, band=2.5).appliance) == 0.0


def test_hvac_unknown_orientation_raises():
    with pytest.raises(ValueError):
        RuleBasedSatisficerAgent(PARAMS).decide(_hvac_ctx(orientation="stoic"))


def test_hvac_defers_under_cpp_at_the_same_state():
    """1C over, balanced: WTP 0.6 clears the 0.55 peak cost but not the 1.20 CPP cost."""
    agent = RuleBasedSatisficerAgent(PARAMS)
    assert agent.decide(_hvac_ctx(indoor=27.0, rate=0.55)).run is True
    assert agent.decide(_hvac_ctx(indoor=27.0, rate=1.20)).run is False


# -- washer rule ---------------------------------------------------------------


def test_washer_urgency_is_linear_from_release_to_zero_slack():
    # release 32, deadline 44, duration 4 -> slack_0 = 8
    urg = [washer_urgency(_washer_ctx(t=t).appliance) for t in range(32, 41)]
    assert urg == pytest.approx([i / 8 for i in range(9)])
    assert washer_urgency(_washer_ctx(t=42).appliance) == 1.0  # past latest start: clamped


def test_washer_wtp_reaches_w_max_at_zero_slack():
    agent = RuleBasedSatisficerAgent(PARAMS)
    assert agent.decide(_washer_ctx(t=32)).wtp == 0.0
    assert agent.decide(_washer_ctx(t=40)).wtp == pytest.approx(PARAMS.w_max)


def test_washer_zero_slack_window_is_full_urgency():
    assert washer_urgency(_washer_ctx(t=40, released=40, deadline=44).appliance) == 1.0


@pytest.mark.parametrize(
    ("kwargs", "run"),
    [({"t": 20}, False), ({"completed": 2}, True), ({"completed": 4}, False)],
)
def test_washer_forced_steps_match_llm_agent(kwargs, run):
    ctx = _washer_ctx(**kwargs)
    rule, llm = RuleBasedSatisficerAgent(PARAMS), LLMAgent(dry_run=True)
    d_rule, d_llm = rule.decide(ctx), llm.decide(ctx)
    assert d_rule.run == d_llm.run == run
    assert d_rule.reason == d_llm.reason
    assert rule.log[-1].forced and llm.log[-1].forced


# -- the shared halt -----------------------------------------------------------


def test_halt_rule_is_the_llm_agents():
    assert satisficing_halt(0.6, 0.55) == pytest.approx((0.05, True))
    assert satisficing_halt(0.5, 0.55)[1] is False
    assert satisficing_halt(0.5, 0.55, wtp_scale=2.0, wtp_factor=0.7) == pytest.approx((0.15, True))
    assert satisficing_halt(0.55, 0.55)[1] is True  # U == 0 runs


def test_agent_calls_the_shared_halt(monkeypatch):
    seen = []

    def spy(wtp, cost_now, scale, factor):
        seen.append((wtp, cost_now, scale, factor))
        return satisficing_halt(wtp, cost_now, scale, factor)

    monkeypatch.setattr("hems.agents.rule_based_agent.satisficing_halt", spy)
    RuleBasedSatisficerAgent(PARAMS, wtp_scale=1.0, wtp_factor=1.0).decide(_hvac_ctx(indoor=27.0))
    assert seen == [(pytest.approx(0.6), pytest.approx(0.55), 1.0, 1.0)]
    assert llm_agent.satisficing_halt is not spy  # the LLM agent's own symbol is untouched


def test_scale_and_factor_enter_the_halt():
    ctx = _hvac_ctx(indoor=27.0, rate=0.55)  # wtp 0.6, cost 0.55
    assert RuleBasedSatisficerAgent(PARAMS, wtp_scale=1.0, wtp_factor=0.7).decide(ctx).run is False
    assert RuleBasedSatisficerAgent(PARAMS, wtp_scale=1.0, wtp_factor=1.0).decide(ctx).run is True


# -- fitting and config --------------------------------------------------------


def test_fit_through_origin_recovers_slope():
    x = [0.0, 0.5, 1.0, 2.0, 3.0]
    assert fit_through_origin(x, [2.5 * v for v in x]) == pytest.approx(2.5)
    # least squares, not a mean of ratios: sum(xy) / sum(x^2)
    assert fit_through_origin([1.0, 2.0], [1.0, 1.0]) == pytest.approx(3.0 / 5.0)
    with pytest.raises(ValueError):
        fit_through_origin([0.0, 0.0], [1.0, 2.0])


def test_params_load_from_experiment_config():
    p = RuleBasedParams.from_yaml()
    assert set(p.k_by_orientation) == {"comfort_prioritising", "balanced", "comfort_flexible"}
    k = p.k_by_orientation
    assert k["comfort_prioritising"] > k["balanced"] > k["comfort_flexible"] > 0
    assert p.w_max > 0


# -- end to end ----------------------------------------------------------------


def test_full_day_on_a_persona_household_without_llm():
    household, hvac, washer = load_persona_environment(8596070, load_assignment())
    tariff = Tariff.from_yaml("config/tariff.yaml").with_cpp_events([])
    agent = RuleBasedSatisficerAgent(RuleBasedParams.from_yaml())
    result = run_simulation(household, tariff, hvac, washer, agent, dt.date(2013, 1, 18))
    assert result.agent_name == "rule_based"
    assert result.washer_completed
    log = agent.log_to_dataframe()
    assert len(log) == 96  # 48 steps x (HVAC + washer)
    hvac_log = log[log["appliance"] == "hvac"]
    assert (hvac_log["run"] == (hvac_log["utility"] >= 0)).all()
