from __future__ import annotations

from typing import Optional

import numpy as np

from baselines.policy_utils import pick_random_valid_action


class RandomPolicy:
    """Random baseline policy compatible with SDNLoadBalancingEnv."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def select_action(self, obs: np.ndarray, env) -> int:
        _ = obs
        return pick_random_valid_action(
            switch_assignment=env.switch_assignment,
            num_controllers=env.num_controllers,
            rng=self._rng,
        )
