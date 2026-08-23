"""DecisionAgent interface.

Responsibility (BUILD_SPEC.md §5): the common contract every agent
(MILP, noisy-MILP, LLM, stub) implements, and the `DecisionContext`
the simulator loop builds and hands to `agent.decide(ctx)` each
appliance/interval. This is the interface only -- no agent logic
lives here.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class Decision(BaseModel):
    """The agent's run/defer verdict for one appliance, one interval."""

    run: bool
    wtp: float | None = None  # LLM only; None for MILP/stub
    reason: str | None = None


class DecisionContext(BaseModel):
    """Everything an agent receives to decide -- Python-computed.

    `appliance` and `household` are left as free-form dicts at the
    scaffolding stage; §7/§8 typed state containers will replace them
    once appliance physics and household params are implemented.
    """

    t: int
    clock: str  # "HH:MM"
    day_index: int
    appliance: dict[str, Any]
    cost_now: float
    rate: float
    is_cpp: bool
    rate_forecast: list[float] | None = None
    household: dict[str, Any]


class DecisionAgent(Protocol):
    """Interface every agent (MILP, noisy-MILP, LLM, stub) implements."""

    def decide(self, ctx: DecisionContext) -> Decision: ...

    def name(self) -> str: ...
