from __future__ import annotations

from typing import Optional

import numpy as np

from baselines.policy_utils import pick_random_valid_action

class RandomPolicy:
    """Random baseline policy compatible with SDNLoadBalancingEnv."""

    # Initialize deterministic random generator when seed is provided.
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    # Stateless policy reset hook to keep interface consistent.
    def reset(self) -> None:
        return None

    # Choose one valid random migration action from the current assignment.
    def select_action(self, obs: np.ndarray, env) -> int:
        _ = obs
        return pick_random_valid_action(
            switch_assignment = env.switch_assignment,
            num_controllers = env.num_controllers,
            rng = self._rng,
        )