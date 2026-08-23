"""Elasticity estimator -- the key metric.

Responsibility (BUILD_SPEC.md §10): short-run own-price arc
elasticity of demand for the CPP-priced window, comparing a baseline
day against a CPP day holding the household fixed. Reports
per-appliance and aggregate; success criterion is falling within
Conway 2020's Australian range (-0.026 to -0.47), calibrated on a
calibration subset and reported on a held-out subset. No estimator
implemented yet -- scaffolding only.
"""
