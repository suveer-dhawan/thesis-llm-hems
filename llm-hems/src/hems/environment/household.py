"""Base load + physical state container.

Responsibility (BUILD_SPEC.md §8): holds the SGSC-derived
uncontrollable base load profile for a household plus the physical
state container that appliances and agents read/write each step, and
the interpretable comfort/persona params from `config/household.yaml`
(thermal_sensitivity, price_sensitivity, deadline, occupied_hours).
No logic implemented yet -- scaffolding only.
"""
