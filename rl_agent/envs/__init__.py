from .sdn_env import SDNLoadBalancingEnv

try:
    from .sdn_multiagent_env import SDNMultiAgentEnv
except Exception:
    SDNMultiAgentEnv = None

try:
    from gymnasium.envs.registration import register

    register(
        id="SDNLoadBalancing-v0",
        entry_point="rl_agent.envs.sdn_env:SDNLoadBalancingEnv",
    )
except Exception:
    # Ignore duplicate registration when module is imported multiple times.
    pass

__all__ = [
    "SDNLoadBalancingEnv",
    "SDNMultiAgentEnv",
]

__version__ = "1.0.0"