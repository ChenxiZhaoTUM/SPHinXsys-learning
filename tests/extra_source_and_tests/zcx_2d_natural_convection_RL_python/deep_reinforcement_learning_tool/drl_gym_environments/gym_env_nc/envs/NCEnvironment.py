import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# TODO: replace with an env var or auto-discovery
sys.path.append(r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_natural_convection_RL_python\lib\Release")
import zcx_2d_natural_convection_RL_python as test_2d


class NCEnvironment(gym.Env):
    """Single-agent environment: action is three bottom-wall temperatures (left/middle/right)."""
    metadata = {}

    def __init__(self, render_mode=None, parallel_envs=0):
        # identifiers
        self.parallel_envs = parallel_envs
        self.episode = 1

        # timing
        self.window_total = 120.0    # CFD time advanced per RL step (seconds)
        self.warmup_time  = 120.0    # baseline warm-up run in reset() before first action
        self.action_time  = 0.0      # absolute simulation clock (seconds)

        # action: absolute temperatures [T_left, T_mid, T_right]
        self.action_low  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.action_high = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        # observation: 8 values (4 flux scalars, 3 KE sums, global KE)
        self.obs_numbers = 8
        obs_low  = np.full(self.obs_numbers, -1e6, dtype=np.float32)
        obs_high = np.full(self.obs_numbers,  1e6, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # internals
        self.nc = None
        self.action_time_steps = 0
        self.total_reward_per_episode = 0.0

    def _read_observation(self):
        """Return 8-dim observation vector."""
        obs = np.zeros(self.obs_numbers, dtype=np.float32)

        # 4 scalar fluxes
        obs[0] = self.nc.getPhiFluxSum()
        obs[1] = self.nc.getLeftPhiFlux()
        obs[2] = self.nc.getMiddlePhiFlux()
        obs[3] = self.nc.getRightPhiFlux()

        # 8×30 probes → KE sums over 3 column groups (0..9, 10..19, 20..29), column-major flattening
        n_rows = 8
        def col_major_idx(row, col): return col * n_rows + row

        def group_energy(col_start, col_end):
            s = 0.0
            for col in range(col_start, col_end):
                for row in range(n_rows):
                    idx = col_major_idx(row, col)
                    vx = self.nc.getLocalVelocity(idx, 0)
                    vy = self.nc.getLocalVelocity(idx, 1)
                    s += vx * vx + vy * vy
            return s

        obs[4] = group_energy(0, 10)
        obs[5] = group_energy(10, 20)
        obs[6] = group_energy(20, 30)

        # global kinetic energy
        obs[7] = self.nc.getGlobalKineticEnergy()
        return obs

    def _compute_reward(self, obs: np.ndarray) -> float:
        """Encourage heat flux, penalize kinetic energy."""
        return float(obs[0] - 0.01 * obs[7])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # create solver; episode index is passed to your C++ object for logging
        self.nc = test_2d.natural_convection_from_sph_cpp(self.parallel_envs, self.episode)

        # warm-up to a baseline state before first action (set to 0.0 if you don't want this)
        self.action_time = float(self.warmup_time)
        self.nc.run_case(self.action_time)

        self.action_time_steps = 0
        self.total_reward_per_episode = 0.0

        obs = self._read_observation().astype(np.float32)
        return obs, {}

    def step(self, action):
        # parse action: [T_left, T_mid, T_right]
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 3:
            raise ValueError(f"Action must have 3 elements [T_left, T_middle, T_right], got shape={action}")
        T_left, T_mid, T_right = map(float, a.tolist())

        self.action_time_steps += 1

        # compute absolute target time for this step
        end_time = self.action_time + self.window_total

        # set temperatures once at window start, then integrate to end_time
        if hasattr(self.nc, "set_down_wall_temperatures"):
            self.nc.set_down_wall_temperatures(T_left, T_mid, T_right)
            self.nc.run_case(end_time)  # absolute time
        else:
            self.nc.run_case_with_temps(end_time, T_left, T_mid, T_right)

        # last-frame observation and reward
        obs = self._read_observation().astype(np.float32)
        reward = self._compute_reward(obs)
        self.total_reward_per_episode += reward

        # advance simulation clock
        self.action_time = end_time

        # logging (optional)
        with open(f'action_env{self.parallel_envs}_epi{self.episode}.txt', 'a') as f:
            f.write(f'action_time: {self.action_time:.6f}  action: [{T_left:.4f}, {T_mid:.4f}, {T_right:.4f}]\n')
        with open(f'reward_env{self.parallel_envs}_epi{self.episode}.txt', 'a') as f:
            f.write(f'action_time: {self.action_time:.6f}  reward: {reward:.6f}\n')

        # one step per episode (change condition if you want multi-step episodes)
        terminated, truncated = True, False
        with open(f'reward_env{self.parallel_envs}.txt', 'a') as file:
            file.write(f'episode: {self.episode}  total_reward: {self.total_reward_per_episode:.6f}\n')
        self.episode += 1

        return obs, float(reward), terminated, truncated, {}

    def render(self):
        return 0

    def _render_frame(self):
        return 0

    def close(self):
        return 0
