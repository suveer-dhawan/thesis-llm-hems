"""Noisy-MILP heterogeneous strawman.

Responsibility (BUILD_SPEC.md §6.2): same MILP as `milp_agent.py`, but with
seeded per-household perturbations to the comfort band width, setpoint, and
effective washer deadline (and optionally the perceived price) before
solving. Represents genuine household-level heterogeneity in preferences,
not a "belief vs. reality" gap -- the jittered params are this simulated
household's actual comfort/urgency, so the same jittered params are what
`run_single.py` should feed back through `simulator.run_simulation` for
this agent's accounting (see `jittered_hvac_params` / `jittered_washer_params`
below). Exists to test whether trivial heterogeneity alone reproduces
staggering + a bounded elasticity -- the bar the LLM agent must beat, not
just the single clean MILP.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import replace
from pathlib import Path

import yaml

from hems.agents.base import Decision, DecisionContext
from hems.agents.milp_agent import MILPAgent
from hems.environment.appliances import HVACParams, WasherParams
from hems.environment.household import Household
from hems.environment.tariff import Tariff

DEFAULT_EXPERIMENT_CONFIG = "config/experiment.yaml"


class NoisyMILPAgent:
    def __init__(
        self,
        household: Household,
        tariff: Tariff,
        hvac_params: HVACParams,
        washer_params: WasherParams,
        date: dt.date,
        seed: int,
        jitter_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
        initial_indoor_temp_c: float | None = None,
    ):
        with open(jitter_config_path) as f:
            cfg = yaml.safe_load(f)["noisy_milp"]

        # Seeded on (run seed, household_id) so each household in a future
        # population run draws an independent but reproducible perturbation.
        rng = random.Random(f"{seed}:{household.params.household_id}")
        band_jitter = rng.uniform(-cfg["comfort_band_jitter_c"], cfg["comfort_band_jitter_c"])
        setpoint_jitter = rng.uniform(-cfg["setpoint_jitter_c"], cfg["setpoint_jitter_c"])
        deadline_jitter = rng.randint(-cfg["deadline_jitter_steps"], cfg["deadline_jitter_steps"])

        self.jittered_hvac_params = replace(
            hvac_params,
            setpoint_c=hvac_params.setpoint_c + setpoint_jitter,
            tolerance_band_c=max(0.1, hvac_params.tolerance_band_c + band_jitter),
        )
        earliest_feasible_deadline = washer_params.released_step + washer_params.duration_steps
        deadline_step = max(
            earliest_feasible_deadline,
            min(washer_params.deadline_step + deadline_jitter, 47),
        )
        self.jittered_washer_params = replace(washer_params, deadline_step=deadline_step)

        self._milp = MILPAgent(
            household,
            tariff,
            self.jittered_hvac_params,
            self.jittered_washer_params,
            date,
            initial_indoor_temp_c=initial_indoor_temp_c,
            price_noise_std=float(cfg.get("price_noise_std", 0.0)),
            rng=rng,
        )

    def decide(self, ctx: DecisionContext) -> Decision:
        decision = self._milp.decide(ctx)
        return Decision(run=decision.run, reason="noisy-milp schedule")

    def name(self) -> str:
        return "noisy_milp"
