from __future__ import annotations

from typing import Optional

import numpy as np

# Map a concrete migration decision into the environment's discrete action index.
def encode_action(switch_assignment: np.ndarray, switch_id: int, target_controller: int, num_controllers: int) -> int:
    """Encode (switch_id, target_controller) to discrete action index used by SDNLoadBalancingEnv."""
    current_controller = int(switch_assignment[switch_id])
    if target_controller == current_controller:
        raise ValueError("target_controller must be different from current controller")

    candidates = [c for c in range(num_controllers) if c != current_controller]
    offset = candidates.index(int(target_controller))
    return int(switch_id * (num_controllers - 1) + offset)

# Generate one valid random migration action for exploration/baseline fallback.
def pick_random_valid_action(switch_assignment: np.ndarray, num_controllers: int, rng: Optional[np.random.Generator] = None) -> int:
    """Pick a valid random migration action (always changes controller)."""
    if rng is None:
        rng = np.random.default_rng()

    switch_id = int(rng.integers(0, len(switch_assignment)))
    current_controller = int(switch_assignment[switch_id])
    candidates = [c for c in range(num_controllers) if c != current_controller]
    target_controller = int(candidates[int(rng.integers(0, len(candidates)))])
    return encode_action(switch_assignment, switch_id, target_controller, num_controllers)