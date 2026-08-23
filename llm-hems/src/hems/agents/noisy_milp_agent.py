"""Noisy-MILP heterogeneous strawman.

Responsibility (BUILD_SPEC.md §6.2): same MILP as `milp_agent.py`,
but per household draws seeded random perturbations -- jittered
comfort-band width, setpoint, and/or effective deadline within
tolerance, optionally small perceived-price noise. Exists to test
whether trivial heterogeneity alone reproduces staggering + a bounded
elasticity, i.e. the bar the LLM agent must beat. No logic
implemented yet -- scaffolding only.
"""
