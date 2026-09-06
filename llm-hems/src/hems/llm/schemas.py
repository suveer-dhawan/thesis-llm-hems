"""Pydantic I/O contracts for LLM calls.

Responsibility: structured-output schema for the LLM satisficing agent's
per-interval, per-appliance call (§6.3) -- the `wtp` + `reason` response
contract that `llm_agent.py` gets back from `llm/client.py` and turns into a
run/defer decision. `wtp` and `reason` are the *only* things the LLM's
answer is used for downstream; Python owns every other number (§1).

Both schemas also carry a state-echo: the model restates the exact input
fields it says it is acting on (indoor temp + comfort band + CPP flag for
HVAC; deadline slack + CPP flag for the washer). This exists purely so a
misread is detectable programmatically -- `agents/llm_agent.py` compares the
echo against the real `DecisionContext` input and logs any mismatch. This
was added after `experiments/phase2_diagnostics.py` found the (then
Matilda-only, pre-hardened-prompt) agent misreading its own input state in
3 of 4 controlled calls: a wrong comfort band, a wrong indoor temp, an
indoor/outdoor temperature mix-up, and an invented active CPP event when
`is_cpp=no` was stated explicitly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HVACWTPResponse(BaseModel):
    """The LLM's per-interval HVAC valuation, plus an echo of the state it used."""

    wtp: float = Field(ge=0, description="Willingness-to-pay in dollars for this interval only.")
    reason: str = Field(description="One-line reason for the wtp value.")
    echo_indoor_temp_c: float = Field(
        description="Restate the indoor temperature (C) from the prompt's CURRENT STATE "
        "block, exactly as given -- do not use the outdoor temperature or any other number."
    )
    echo_band_lo_c: float = Field(
        description="Restate the comfort band's lower bound (C) from the prompt, exactly as given."
    )
    echo_band_hi_c: float = Field(
        description="Restate the comfort band's upper bound (C) from the prompt, exactly as given."
    )
    echo_is_cpp: bool = Field(
        description="Restate whether a CPP event is active right now, exactly as stated in "
        "the prompt -- do not infer this from the price."
    )


class WasherWTPResponse(BaseModel):
    """The LLM's per-interval washer valuation, plus an echo of the state it used."""

    wtp: float = Field(ge=0, description="Willingness-to-pay in dollars for this interval only.")
    reason: str = Field(description="One-line reason for the wtp value.")
    echo_steps_until_deadline: int = Field(
        description="Restate the number of half-hour steps remaining until the deadline "
        "from the prompt, exactly as given."
    )
    echo_is_cpp: bool = Field(
        description="Restate whether a CPP event is active right now, exactly as stated in "
        "the prompt -- do not infer this from the price."
    )
