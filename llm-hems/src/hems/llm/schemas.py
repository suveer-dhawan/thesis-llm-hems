"""Pydantic I/O contracts for LLM calls.

Responsibility: structured-output schemas for the LLM satisficing
agent's per-interval, per-appliance call (§6.3) -- the `wtp` +
`reason` response contract that `llm_agent.py` parses and Python then
turns into a run/defer decision. No schemas implemented yet --
scaffolding only.
"""
