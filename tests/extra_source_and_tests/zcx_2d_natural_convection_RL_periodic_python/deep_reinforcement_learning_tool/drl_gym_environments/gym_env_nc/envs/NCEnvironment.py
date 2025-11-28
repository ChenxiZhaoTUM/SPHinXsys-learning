import sys
import os
import glob, re
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# TODO: replace with an env var or auto-discovery
sys.path.append(r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_natural_convection_RL_periodic_python\lib\Release")
import zcx_2d_natural_convection_RL_periodic_python as test_2d


def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _latest_restart_step(restart_dir: str) -> int:
    steps = []
    for f in glob.glob(os.path.join(restart_dir, "*_rst_*.xml")):
        m = re.search(r"_rst_(\d+)\.xml$", f.replace("\\", "/"))
        if m:
            steps.append(int(m.group(1)))
    mx = max(steps) if steps else -1
    return mx if mx > 0 else -1


class NCEnvironment(gym.Env):
    """
    Single-agent environment for natural convection control (SPHinXsys).

    Action: R^n_seg in [-1,1], converted to segment wall temperatures around 2.0
    Observation: time-averaged (last 4 CFD steps) probe fields (u, v, T) on an 8x30 grid (flattened)
    Reward: computed *directly from solver* (not from obs):
            r = 2.67 - [ (1 - beta) * Nu_global + beta * mean(Nu_local_i) ]
    """

    metadata = {}

    def __init__(self, render_mode=None, parallel_envs: int = 0, n_seg: int = 10):
        super().__init__()

        # ----- bookkeeping -----
        self.parallel_envs = int(parallel_envs)
        self.episode = 1

        # ----- control segmentation -----
        self.n_seg = int(n_seg)

        # ----- folders -----
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "../training_process"))
        self.training_root = _mkdir(os.path.join(proj_root, "training_results_single"))
        self.restart_dir = _mkdir(os.path.join(self.training_root, "restart"))
        self.log_dir = _mkdir(os.path.join(self.training_root, f"logs_env_{self.parallel_envs}"))

        # ----- timing -----
        self.step_to_load = 0
        self.warmup_time = 400.0
        self.delta_time = 2.0
        self.sim_time = 0.0

        # ----- episode length -----
        self.max_steps_per_episode = 200
        self.step_count = 0
        self.max_steps_per_episode_eval = 4 * self.max_steps_per_episode
        self.deterministic = False

        # ----- action space -----
        self.action_low = np.full(self.n_seg, -1.0, dtype=np.float32)
        self.action_high = np.full(self.n_seg, 1.0, dtype=np.float32)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        # ----- probe grid & observation space -----
        self.n_rows = 8
        self.n_cols = 30
        self._probe_dim = 3  # u, v, T
        self._obs_len = self.n_rows * self.n_cols * self._probe_dim
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(self._obs_len,), dtype=np.float32)
        self._probe_hist = deque(maxlen=4)  # 4-step moving average

        # ----- reward shaping params (Nu mix) -----
        self.beta = 0.0015  # small local contribution
        self.nu_target = 20.87
        self.reward_scale = 1.0

        # ----- solver handles / runtime -----
        self.nc_base = None
        self.nc = None
        self.total_reward_per_episode = 0.0

    # ------------------------------------------------------------------
    # Helper: produce the per-segment temperature array we send to C++
    # ------------------------------------------------------------------
    def _build_segment_temps(self, action_vec: np.ndarray):
        """map raw actions -> physically safe per-segment temperatures near 2.0"""
        if len(action_vec) != self.n_seg:
            raise ValueError(
                f"Expected action of length {self.n_seg}, got {len(action_vec)}"
            )

        baseline_T, ampl = 2.0, 0.75
        raw = np.asarray(action_vec, dtype=np.float32)
        raw = np.clip(raw, -1.0, 1.0)
        centered = raw - float(np.mean(raw))
        max_abs = float(np.max(np.abs(centered))) if self.n_seg > 0 else 0.0
        scale = (ampl / max_abs) if max_abs > 1e-8 else 0.0
        temps = baseline_T + centered * scale
        return np.clip(temps, 0.0, 4.0).astype(np.float32).tolist()

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _probe_index_col_major(self, row: int, col: int) -> int:
        return col * self.n_rows + row

    def _snapshot_probes(self, sim) -> np.ndarray:
        """instantaneous (u, v, T) on the 8x30 probe grid, flattened"""
        out = np.empty((self.n_rows, self.n_cols, self._probe_dim), dtype=np.float32)
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                idx = self._probe_index_col_major(row, col)
                u = sim.get_local_velocity(idx, 0)
                v = sim.get_local_velocity(idx, 1)
                T = sim.get_local_temperature(idx)
                out[row, col, 0] = u
                out[row, col, 1] = v
                out[row, col, 2] = T
        return out.reshape(-1)

    def _read_observation(self, sim) -> np.ndarray:
        """time-averaged (last 4 CFD steps) probe fields (u,v,T), flattened"""
        snap = self._snapshot_probes(sim)  # 当前瞬时
        self._probe_hist.append(snap)
        # 若历史不足 4 帧（刚 reset），就用已有帧做平均
        hist = list(self._probe_hist)
        obs = np.mean(hist, axis=0).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        """r = nu_target - [(1 - beta) * Nu_global + beta * mean(Nu_local_i)]"""
        gen_Nu = float(self.nc.get_global_heat_flux())
        loc_Nu_vals = [float(self.nc.get_local_phi_flux(i)) for i in range(self.n_seg)]
        loc_Nu = float(np.sum(loc_Nu_vals)) if self.n_seg > 0 else 0.0
        reward_nu = (1.0 - self.beta) * gen_Nu + self.beta * loc_Nu
        reward = (self.nu_target - reward_nu) / self.reward_scale

        # try:
        #     dbg_path = os.path.join(self.log_dir, f"nu_debug_env{self.parallel_envs}_epi{self.episode}.txt")
        #     line = (
        #         f"t={self.sim_time:.3f}, gen_Nu={gen_Nu:.6f}, "
        #         f"loc_Nu_sum={loc_Nu:.6f}\n"
        #     )
        #     print("[NuDbg]", line.strip())
        #     with open(dbg_path, "a", encoding="utf-8") as f:
        #         f.write(line)
        # except Exception as e:
        #     print(f"[NuDbg][warn] logging failed: {e}")

        return reward

    # ------------------------------------------------------------------
    # Gym API: reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Starts a new episode:
        - Create a new CFD solver instance in C++.
        - Set the bottom wall to a uniform baseline temperature (2.0).
        - Advance simulation to warmup_time (baseline / uncontrolled flow).
        - Return the observation after warmup.
        """
        super().reset(seed=seed)
        # ---- Episode start banner ----
        msg = f"[env {self.parallel_envs}] ===== Episode {self.episode} start ====="
        print(msg)
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "episodes.txt"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        action_log = os.path.join(self.log_dir, f"action_env{self.parallel_envs}_epi{self.episode}.txt")
        reward_log = os.path.join(self.log_dir, f"reward_env{self.parallel_envs}_epi{self.episode}.txt")
        open(action_log, "w").close()
        open(reward_log, "w").close()
        open(os.path.join(self.log_dir, f"reward_env{self.parallel_envs}.txt"), "a").close()

        os.chdir(self.training_root)
        if self.episode == 1:
            # baseline solver (starts from t=0)
            self.nc_base = test_2d.natural_convection_from_sph_cpp(self.parallel_envs, self.episode, 0)
            self.nc_base.set_segment_temperatures([2.0] * self.n_seg)
            self.sim_time = float(self.warmup_time)
            self.nc_base.run_case(self.sim_time)
            # switch to training solver from latest restart
        self.step_to_load = _latest_restart_step(self.restart_dir)
        if self.step_to_load < 0:
            raise RuntimeError("No restart files found under training_results/restart. "
                               "Make sure episode 1 finished the warm-up.")
        self.nc = test_2d.natural_convection_from_sph_cpp(self.parallel_envs, self.episode, int(self.step_to_load))
        self.nc.run_case(self.warmup_time)
        self.sim_time = float(self.warmup_time)

        # housekeeping
        self.step_count = 0
        self.total_reward_per_episode = 0.0

        # fill probe history with current snapshot to start averaging
        self._probe_hist.clear()
        snap0 = self._snapshot_probes(self.nc)
        for _ in range(4):
            self._probe_hist.append(snap0.copy())

        obs0 = self._read_observation(self.nc)
        return obs0, {}

    # ------------------------------------------------------------------
    # Gym API: step
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Multistep episode:
        - Take an action vector of length n_seg (segment temperatures).
        - Send these segment temps to C++.
        - Advance CFD by delta_time seconds of sim time.
        - Observe, compute reward (baseline-subtracted), return and terminate.
        """
        # 1) action -> segment temperatures
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != self.n_seg:
            raise ValueError(f"Action must have length {self.n_seg}, got shape={action}")
        seg_temps = self._build_segment_temps(a)
        self.nc.set_segment_temperatures(seg_temps)

        # 2) advance CFD
        end_time = self.sim_time + self.delta_time
        self.nc.run_case(end_time)
        self.step_count += 1
        self.sim_time = end_time

        # 3) observation (time-averaged probes)
        obs = self._read_observation(self.nc)

        # 4) reward (directly from solver Nu)
        reward_now = self._compute_reward()
        self.total_reward_per_episode += reward_now

        # 5) logging
        with open(os.path.join(self.log_dir, f'action_env{self.parallel_envs}_epi{self.episode}.txt'), 'a') as f:
            f.write(f"clock: {self.sim_time:.6f}  raw_action: {a.tolist()}  seg_temps: {seg_temps}\n")

        with open(os.path.join(self.log_dir, f'reward_env{self.parallel_envs}_epi{self.episode}.txt'), 'a') as f:
            f.write(f'clock: {self.sim_time:.6f} | reward: {reward_now:.6f} | temps: {seg_temps}\n')

        # 6) termination
        episode_limit = self.max_steps_per_episode if not self.deterministic else self.max_steps_per_episode_eval
        if self.step_count >= episode_limit:
            terminated, truncated = True, False
            with open(os.path.join(self.log_dir, f'reward_env{self.parallel_envs}.txt'), 'a',
                      encoding='utf-8') as file:
                file.write(f'episode: {self.episode}  total_reward: {self.total_reward_per_episode:.6f}\n')
            self.episode += 1
        else:
            terminated, truncated = False, False

        return obs, float(reward_now), terminated, truncated, {}

    def render(self):
        return 0

    def _render_frame(self):
        return 0

    def close(self):
        return 0
