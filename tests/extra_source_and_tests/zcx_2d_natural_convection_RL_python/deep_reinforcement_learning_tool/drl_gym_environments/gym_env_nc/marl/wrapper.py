import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CentralizedParallelToGym(gym.Env):
    """
    把 PettingZoo ParallelEnv（多智能体）封装成 Gym 单智能体环境：
    - 动作：shape=(n_seg,)，每个元素对应一个 agent 的标量动作
    - 观测：直接用任意一个 agent 的观测（各 agent 相同）
    - 奖励：对所有 agent 奖励取均值（你也可以改成求和）
    """
    metadata = {"render_modes": []}

    def __init__(self, pz_parallel_env):
        super().__init__()
        self.pz = pz_parallel_env
        self.agents = list(self.pz.possible_agents)
        self.n_seg = len(self.agents)

        # 单智能体动作空间 = 10 维向量（每维对应一个 agent 的 1 维动作）
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_seg,), dtype=np.float32)
        # 观测空间直接复用任意一个 agent 的观测空间（它们一致）
        self.observation_space = self.pz.observation_space

    def reset(self, *, seed=None, options=None):
        obs_dict, _infos = self.pz.reset(seed=seed, options=options)
        # 有些并行环境的 reset 也可能返回空，这里都兜住
        if not obs_dict:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = next(iter(obs_dict.values()))
        self._last_obs = np.array(obs, copy=True)
        return obs, {}

    def step(self, action):
        # action: shape = (n_seg,)
        act_dict = {f"agent_{i}": np.array([float(action[i])], dtype=np.float32)
                    for i in range(self.n_seg)}

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.pz.step(act_dict)

        # ---- 观测兜底 ----
        if not obs_dict:
            # 并行环境在“未收齐动作/回合结束后”的一次调用中可能给空观测
            if self._last_obs is not None:
                obs = self._last_obs
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = next(iter(obs_dict.values()))
            self._last_obs = np.array(obs, copy=True)

        # 奖励聚合（按需改成 sum）
        reward = float(np.mean(list(rew_dict.values()))) if rew_dict else 0.0
        terminated = bool(any(term_dict.values())) if term_dict else False
        truncated = bool(any(trunc_dict.values())) if trunc_dict else False

        return obs, reward, terminated, truncated, {}
