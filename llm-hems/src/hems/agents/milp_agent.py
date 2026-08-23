"""MILP optimal cost-minimiser (hyper-rational baseline).

Responsibility (BUILD_SPEC.md §6.1): PuLP-based (CBC solver) horizon
optimisation minimising total energy cost, subject to shiftable-load
contiguity/deadline constraints and hard HVAC comfort-band
constraints. Solves once per household, then the schedule is fed
through `simulator.py` for apples-to-apples accounting. No MILP
logic implemented yet -- scaffolding only.
"""
