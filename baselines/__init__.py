from baselines.least_load import LeastLoadBalancer
from baselines.random_policy import RandomPolicy
from baselines.round_robin import RoundRobinBalancer

def make_baseline(name: str, num_controllers: int, num_switches: int):
    key = name.strip().lower()
    if key == "round_robin":
        return RoundRobinBalancer(num_controllers = num_controllers, num_switches = num_switches)
    if key == "least_load":
        return LeastLoadBalancer(num_controllers = num_controllers, num_switches = num_switches)
    if key == "random":
        return RandomPolicy()
    raise ValueError(f"Unknown baseline: {name}")

__all__ = [
    "RoundRobinBalancer",
    "LeastLoadBalancer",
    "RandomPolicy",
    "make_baseline",
]