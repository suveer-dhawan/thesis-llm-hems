"""One household, one agent.

Responsibility (BUILD_SPEC.md §3, §16 P0): CLI entry point that runs
a single household through the simulator loop for one agent and
writes a timestamped results folder. At this scaffolding stage,
`environment/*` has no physics yet, so this script runs a minimal
in-place loop (48 steps, placeholder appliance/household context)
against the agent interface only, to prove the agent contract and
CLI wiring work end-to-end before any real logic is implemented.
Only `--agent stub` is wired up; other agent names are rejected until
milp/noisy_milp/llm agents exist.
"""

from __future__ import annotations

import argparse

from hems.agents.base import Decision, DecisionAgent, DecisionContext
from hems.agents.stub_agent import StubAgent

AGENTS: dict[str, type[DecisionAgent]] = {
    "stub": StubAgent,
}

N_STEPS = 48  # 30-min steps in one day, per BUILD_SPEC.md §4


def build_agent(name: str) -> DecisionAgent:
    if name not in AGENTS:
        raise NotImplementedError(
            f"agent '{name}' is not implemented yet; available: {sorted(AGENTS)}"
        )
    return AGENTS[name]()


def run(agent: DecisionAgent) -> list[Decision]:
    """Minimal placeholder loop -- no tariff/appliance physics yet.

    Mirrors the shape of the real loop in BUILD_SPEC.md §4 closely
    enough to exercise `DecisionContext` -> `agent.decide` ->
    `Decision` end-to-end, without depending on unimplemented
    `environment` modules.
    """
    decisions: list[Decision] = []
    for t in range(N_STEPS):
        ctx = DecisionContext(
            t=t,
            clock=f"{t // 2:02d}:{'00' if t % 2 == 0 else '30'}",
            day_index=0,
            appliance={"type": "placeholder"},
            cost_now=0.0,
            rate=0.0,
            is_cpp=False,
            household={"type": "placeholder"},
        )
        decisions.append(agent.decide(ctx))
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", default="stub", choices=sorted(AGENTS), help="agent to run"
    )
    args = parser.parse_args()

    agent = build_agent(args.agent)
    decisions = run(agent)
    n_run = sum(1 for d in decisions if d.run)
    print(
        f"[run_single] agent={agent.name()} steps={len(decisions)} "
        f"runs={n_run} defers={len(decisions) - n_run}"
    )


if __name__ == "__main__":
    main()
