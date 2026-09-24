"""LLMAgent schema selection + misread detection.

No live API calls -- `dry_run=True` exercises the full `decide()` path
(prompt building, schema selection, the halt rule, logging) with the
deterministic stub response, and directly checks `_detect_misreads` against
hand-built responses to confirm each echoed field is actually compared
against the real input.

The prompt is price-blind (BUILD_SPEC.md §2.3): `DecisionContext.rate` /
`is_cpp` / `cost_now` are still used by Python for the halt rule, but never
appear in `_build_hvac_prompt` / `_build_washer_prompt`'s output, so these
tests also assert the price/CPP fields are absent from the rendered prompt.
"""

from __future__ import annotations

from hems.agents.base import DecisionContext
import pytest

from hems.agents.llm_agent import COMFORT_ORIENTATION_TEXT, LLMAgent
from hems.llm.schemas import HVACWTPResponse, WasherWTPResponse

HOUSEHOLD = {
    "thermal_sensitivity": "moderate",
    "price_sensitivity": "moderate",
    "comfort_orientation": "balanced",
}

# Words that would leak price / cost / money framing into the persona line.
_MONEY_WORDS = ("price", "cost", "money", "cheap", "expensive", "bill", "$")


def _hvac_ctx(indoor_temp_c=27.0, is_cpp=False, rate=0.55, household=None):
    return DecisionContext(
        t=30,
        clock="15:00",
        day_index=0,
        appliance={
            "type": "hvac",
            "indoor_temp_c": indoor_temp_c,
            "outdoor_temp_c": 33.0,
            "setpoint_c": 24.0,
            "tolerance_band_c": 2.0,
            "rated_kw": 2.0,
        },
        cost_now=2.0 * 0.5 * rate,
        rate=rate,
        is_cpp=is_cpp,
        household=household or HOUSEHOLD,
    )


def _washer_ctx(
    t=10, deadline_step=20, is_cpp=False, rate=0.55, duration_steps=4, steps_completed=0
):
    steps_remaining = duration_steps - steps_completed
    steps_until_deadline = deadline_step - t
    slack_steps = steps_until_deadline - steps_remaining
    return DecisionContext(
        t=t,
        clock="05:00",
        day_index=0,
        appliance={
            "type": "washer",
            "steps_completed": steps_completed,
            "duration_steps": duration_steps,
            "released_step": 0,
            "deadline_step": deadline_step,
            "slack_steps": slack_steps,
            "interruptible": False,
            "rated_kw": 0.8,
        },
        cost_now=0.8 * 0.5 * rate,
        rate=rate,
        is_cpp=is_cpp,
        household=HOUSEHOLD,
    )


def test_dry_run_hvac_never_misreads():
    agent = LLMAgent(dry_run=True)
    ctx = _hvac_ctx(indoor_temp_c=30.0, is_cpp=True)
    agent.decide(ctx)
    entry = agent.log[-1]
    assert entry.misreads == []
    assert entry.provider == "dry_run"


def test_dry_run_washer_never_misreads():
    agent = LLMAgent(dry_run=True)
    ctx = _washer_ctx(t=18, deadline_step=20)
    agent.decide(ctx)
    entry = agent.log[-1]
    assert entry.misreads == []


def test_hvac_prompt_is_price_blind():
    agent = LLMAgent(dry_run=True)
    ctx = _hvac_ctx(indoor_temp_c=30.0, is_cpp=True, rate=1.20)
    prompt = agent._build_hvac_prompt(ctx)
    assert "1.20" not in prompt
    assert "1.2" not in prompt
    assert "CPP" not in prompt
    assert "price sensitivity" not in prompt.lower()  # cost sensitivity is applied in Python


@pytest.mark.parametrize("orientation", sorted(COMFORT_ORIENTATION_TEXT))
def test_hvac_prompt_renders_comfort_orientation_without_money_wording(orientation):
    ctx = _hvac_ctx(household={**HOUSEHOLD, "comfort_orientation": orientation})
    prompt = LLMAgent(dry_run=True)._build_hvac_prompt(ctx)
    text = COMFORT_ORIENTATION_TEXT[orientation]
    assert f"This household {text}" in prompt
    assert not any(w in text.lower() for w in _MONEY_WORDS)
    # the numeric band the MILP uses is still stated in the prompt's state block
    assert "22.0 to 26.0" in prompt


def test_hvac_prompt_rejects_unknown_orientation():
    ctx = _hvac_ctx(household={**HOUSEHOLD, "comfort_orientation": "stoic"})
    with pytest.raises(ValueError):
        LLMAgent(dry_run=True)._build_hvac_prompt(ctx)


def test_washer_prompt_is_price_blind_and_states_slack():
    agent = LLMAgent(dry_run=True)
    # steps_until_deadline=4, steps_remaining=4 -> zero slack (the bug case
    # from the v2/v3 probes: the model said "plenty of slack" here without
    # an explicit slack figure).
    ctx = _washer_ctx(t=16, deadline_step=20, duration_steps=4, steps_completed=0, rate=1.20)
    prompt = agent._build_washer_prompt(ctx)
    assert "1.20" not in prompt
    assert "CPP" not in prompt
    assert "price sensitivity" not in prompt.lower()
    assert "buffer before deadline" in prompt
    assert "0 half-hour steps of slack" in prompt


def _misread_checker() -> LLMAgent:
    """An agent whose `_dry_run` flag is forced off so `_detect_misreads`
    actually compares fields, without needing a real client/API key --
    only `_detect_misreads` is exercised here, never `decide()`."""
    agent = LLMAgent(dry_run=True)
    agent._dry_run = False
    return agent


def test_detect_misreads_hvac_flags_every_field():
    agent = _misread_checker()
    ctx = _hvac_ctx(indoor_temp_c=27.0, is_cpp=False)
    # band is [22.0, 26.0] for setpoint=24.0, tolerance=2.0
    bad_response = HVACWTPResponse(
        wtp=1.0,
        reason="wrong on every count",
        echo_indoor_temp_c=32.0,  # actually the outdoor temp -- classic mixup
        echo_band_lo_c=24.0,  # wrong: should be 22.0
        echo_band_hi_c=26.0,  # correct
    )
    misreads = agent._detect_misreads("hvac", ctx, bad_response)
    assert set(misreads) == {"indoor_temp_c", "band_lo_c"}


def test_detect_misreads_hvac_clean_response():
    agent = _misread_checker()
    ctx = _hvac_ctx(indoor_temp_c=27.0, is_cpp=False)
    good_response = HVACWTPResponse(
        wtp=1.0,
        reason="correct",
        echo_indoor_temp_c=27.0,
        echo_band_lo_c=22.0,
        echo_band_hi_c=26.0,
    )
    assert agent._detect_misreads("hvac", ctx, good_response) == []


def test_detect_misreads_washer():
    agent = _misread_checker()
    # 10 steps until deadline, 4 remaining -> slack 6
    ctx = _washer_ctx(t=10, deadline_step=20, is_cpp=True)
    bad_response = WasherWTPResponse(
        wtp=1.0,
        reason="wrong deadline and slack",
        echo_steps_until_deadline=4,
        echo_slack_steps=0,
    )
    assert set(agent._detect_misreads("washer", ctx, bad_response)) == {
        "steps_until_deadline",
        "slack_steps",
    }

    good_response = WasherWTPResponse(
        wtp=1.0, reason="correct", echo_steps_until_deadline=10, echo_slack_steps=6
    )
    assert agent._detect_misreads("washer", ctx, good_response) == []


def test_wtp_scale_multiplies_wtp_in_halt_rule():
    ctx = _hvac_ctx(indoor_temp_c=26.2, rate=0.55)  # stub wtp = 0.8 * 0.7 = 0.56 vs cost 0.55
    assert LLMAgent(dry_run=True, wtp_scale=1.0).decide(ctx).run
    assert not LLMAgent(dry_run=True, wtp_scale=0.5).decide(ctx).run


def test_wtp_factor_composes_with_wtp_scale():
    ctx = _hvac_ctx(indoor_temp_c=26.2, rate=0.55)  # stub wtp 0.56 vs cost 0.55
    assert not LLMAgent(dry_run=True, wtp_scale=1.0, wtp_factor=0.7).decide(ctx).run
    assert LLMAgent(dry_run=True, wtp_scale=0.8, wtp_factor=1.3).decide(ctx).run  # 1.04x
    agent = LLMAgent(dry_run=True, wtp_scale=2.0, wtp_factor=0.7)
    agent.decide(ctx)
    e = agent.log[-1]
    assert e.utility == pytest.approx(2.0 * 0.7 * e.wtp - e.cost_now)
    assert (e.wtp_scale, e.wtp_factor) == (2.0, 0.7)


def test_washer_forced_states_skip_the_prompt():
    agent = LLMAgent(dry_run=True)
    # mid-cycle: must continue, no prompt
    assert agent.decide(_washer_ctx(t=12, steps_completed=2)).run
    # completed: nothing to do
    assert not agent.decide(_washer_ctx(t=15, steps_completed=4)).run
    # before release
    ctx = _washer_ctx(t=3)
    ctx.appliance["released_step"] = 5
    assert not agent.decide(ctx).run
    assert all(e.forced and e.prompt == "" for e in agent.log)
    assert agent.prompted_decisions == 0
    agent.decide(_washer_ctx(t=10))  # released, not started -> a real decision
    assert agent.prompted_decisions == 1
