"""Hybrid satisficing agent (the thesis).

Responsibility (BUILD_SPEC.md §6.3): per appliance per interval,
prompt the LLM for a structured `wtp` (willingness-to-pay) + one-line
`reason` given comfort state and time -- no cost comparison is asked
of the LLM. Python then computes `U = wtp - cost_now` and applies the
deterministic halt rule: run iff `U >= 0`. Every prompt, WTP,
reasoning string, computed cost, and run/defer decision is logged.
No LLM logic implemented yet -- scaffolding only.
"""
