"""The 30-min orchestration loop.

Responsibility (BUILD_SPEC.md §4): runs one household over a horizon
(default one day = 48 steps), stepping through tariff state and each
controllable appliance, calling `agent.decide(ctx)`, applying the
run/defer decision to update physical state and cost, logging every
decision, and accumulating the aggregate load curve into a
`RunResult`. No orchestration logic implemented yet -- scaffolding
only.
"""
