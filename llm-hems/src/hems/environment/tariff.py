"""TOU + CPP cost calculator.

Responsibility (BUILD_SPEC.md §9): expose `rate_at(t)` returning
`(rate, is_cpp)` for a given 30-min step, driven entirely by
`config/tariff.yaml` (peak/shoulder/off-peak windows and CPP event
schedule). No tariff math is implemented yet — scaffolding only.
"""
