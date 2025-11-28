import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CentralizedParallelToGym(gym.Env):
    """
    Wrap a PettingZoo ParallelEnv (multi-agent) as a single-agent Gymnasium Env.

    - Action: shape=(n_seg,), each element is the scalar action for one agent.
    - Observation: reuse any single agent's observation (all agents share the same shape).
    - Reward: aggregate all agents' rewards by mean (default) or sum.

    Notes
    -----
    * This wrapper assumes the parallel env expects one scalar action per agent
      (i.e., agent action_space is Box(shape=(1,), ...)).
    * The wrapper always submits actions for ALL agents at each step, so the
      underlying parallel env should not return empty observations in normal flow.
    """
    metadata = {"render_modes": []}

    def __init__(self, pz_parallel_env):
        super().__init__()
        self.pz = pz_parallel_env
        self.agents = list(self.pz.possible_agents)
        self.n_seg = len(self.agents)

        # Single-agent action space: R^{n_seg} in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_seg,), dtype=np.float32)

        self.observation_space = self.pz.observation_space

    def reset(self, *, seed=None, options=None):
        obs_dict, _infos = self.pz.reset(seed=seed, options=options)

        if not obs_dict:
            # Defensive fallback: zero obs if env returns empty dict
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            # Take the first agent's observation
            obs = next(iter(obs_dict.values()))

        self._last_obs = np.array(obs, copy=True)
        return obs, {}

    def step(self, action):
        """
        Parameters
        ----------
        action : np.ndarray
            Shape (n_seg,). Each element corresponds to one agent's scalar action.

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # Validate and format actions -> dict[str, np.ndarray] for the parallel env
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_seg:
            raise ValueError(
                f"Action must have shape ({self.n_seg},), got {action.shape}"
            )

        act_dict = {
            f"agent_{i}": np.array([float(action[i])], dtype=np.float32)
            for i in range(self.n_seg)
        }

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.pz.step(act_dict)

        # Observation fallback if the env ever returns empty dict (shouldn't happen here)
        if obs_dict:
            obs = next(iter(obs_dict.values()))
            self._last_obs = np.array(obs, copy=True)
        else:
            obs = (
                self._last_obs
                if self._last_obs is not None
                else np.zeros(self.observation_space.shape, dtype=np.float32)
            )

        # Reward aggregation
        reward = float(np.mean(list(rew_dict.values()))) if rew_dict else 0.0
        terminated = bool(any(term_dict.values())) if term_dict else False
        truncated = bool(any(trunc_dict.values())) if trunc_dict else False

        return obs, reward, terminated, truncated, {}
