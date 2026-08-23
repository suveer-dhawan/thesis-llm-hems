"""Stub agent that always defers.

Responsibility: minimal `DecisionAgent` implementation used to prove
the simulator loop runs end-to-end before any real agent logic
(MILP, noisy-MILP, LLM) exists. Referenced by `run_single.py --agent
stub` per BUILD_SPEC.md §16 (P0 definition of done). Always returns
`run=False` -- no decision logic.
"""

from __future__ import annotations

from hems.agents.base import Decision, DecisionContext


class StubAgent:
    """Always defers: never runs a controllable appliance."""

    def decide(self, ctx: DecisionContext) -> Decision:
        return Decision(run=False, reason="stub agent: always defers")

    def name(self) -> str:
        return "stub"
