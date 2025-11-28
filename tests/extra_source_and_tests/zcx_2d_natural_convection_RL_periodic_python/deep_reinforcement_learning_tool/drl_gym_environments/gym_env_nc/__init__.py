from gymnasium.envs.registration import register

register(
    id="NC-v0",
    entry_point="gym_env_nc.envs:NCEnvironment",
    kwargs={'parallel_envs': 0},
    max_episode_steps=500,  # ?
    reward_threshold=500.0,
)

from .marl import parallel_env, raw_env

__all__ = ["parallel_env", "raw_env"]
