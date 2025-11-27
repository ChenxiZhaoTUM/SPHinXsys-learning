import sys
import os
import glob, re
import numpy as np
from typing import Dict, List, Tuple
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

# TODO: replace with an env var or auto-discovery
sys.path.append(r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_natural_convection_RL_python\lib\Release")
import zcx_2d_natural_convection_RL_python as test_2d


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


class NCMAEnvParallel(ParallelEnv):
    """
    Multi-agent environment for natural convection control.
    """

    metadata = {"name": "NCMA-v0"}

    def __init__(self, n_seg: int = 10, warmup_time: float = 400.0, delta_time: float = 2.0,
                 parallel_envs: int = 0):
        super().__init__()

        # ----- identifiers / bookkeeping -----
        self.parallel_envs = parallel_envs  # which parallel env this is (for logging)
        self.episode = 1  # episode counter

        # ----- control segmentation -----
        assert n_seg > 0
        self.n_seg = n_seg
        self.agents = [f"agent_{i}" for i in range(n_seg)]
        self.possible_agents = list(self.agents)

        # ----- result folder -----
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # .../bin/drl
        self.training_root = _mkdir(os.path.join(proj_root, "drl_tianshou_training", "training_results"))
        self.restart_dir = _mkdir(os.path.join(self.training_root, "restart"))  # ./restart
        # each env log directory
        self.log_dir = _mkdir(os.path.join(self.training_root, f"logs_env_{self.parallel_envs}"))

        # ----- physical timing -----
        self.step_to_load = 0
        self.warmup_time = float(warmup_time)  # warmup_time: run baseline before first action
        self.delta_time = float(delta_time)  # delta_time: CFD physical time per control step
        # TODO: test how long it will take to stabilize after the temperature is changed, or it does not need steady solution
        # running simulation time cursor (absolute physical time in solver)
        self.sim_time = 0.0

        # ---------------- episode length in terms of control actions ----------------
        self.max_steps_per_episode = 200  # TRAINING length: 200 actions per episode
        self.step_count = 0
        self.max_steps_per_episode_eval = 4 * self.max_steps_per_episode  # for evaluation
        self.deterministic = False  # training mode by default

        # ----- action space -----
        self._act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._pending_actions: Dict[str, float] = {}

        # ----- observation space -----
        # obs dimension = 2*n_seg + 2:
        #   [global flux] +
        #   [n_seg local flux per segment] +
        #   [n_seg local KE per segment group from probes] +
        #   [global KE]
        self.obs_numbers = self.n_seg * 2 + 2
        obs_low = np.full(self.obs_numbers, -1e6, dtype=np.float32)
        obs_high = np.full(self.obs_numbers, 1e6, dtype=np.float32)
        self._obs_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self._last_obs = None  # 缓存观测

        # ----- reward shaping / normalization -----
        # We'll compute a "baseline_reward" in reset() after warmup,
        # and subtract it in step() to stabilize learning.
        # self.baseline_reward = None  # will be set in reset()
        self.beta = 0.0015
        self.base_gen_KE = None  # will be set in reset()
        self.base_loc_KE = None
        self.base_loc_Nu = None
        self.base_gen_Nu = None
        self.Nu_base = None
        self.KE_base = None
        self.reward_scale = 1.0

        # ----- runtime state -----
        self.nc_base = None  # the C++ solver / simulation handle
        self.baseline_obs = None
        self.nc = None  # the C++ solver / simulation handle
        self.total_reward_per_episode = 0.0
        self._done = False

    # ---------- PettingZoo interface ----------
    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    @property
    def observation_spaces(self):
        return {agent: self._obs_space for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self._act_space for agent in self.possible_agents}

    # ------------------------------------------------------------------
    # Helper: produce the per-segment temperature array we send to C++
    # ------------------------------------------------------------------
    def _build_segment_temps(self, actions_dict: Dict[str, np.ndarray]) -> List[float]:
        """把 10 个 agent 的标量动作合并成 10 段温度"""
        baseline_T, ampl = 2.0, 0.75
        # 拉成长度 n_seg 的向量
        raw = np.array([float(np.clip(actions_dict[f"agent_{i}"], -1.0, 1.0)) for i in range(self.n_seg)],
                       dtype=np.float32)
        centered = raw - float(np.mean(raw))
        max_abs = float(np.max(np.abs(centered))) if self.n_seg > 0 else 0.0
        scale = (ampl / max_abs) if max_abs > 1e-8 else 0.0
        temps = baseline_T + centered * scale
        return np.clip(temps, 0.0, 4.0).tolist()

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _read_observation(self, sim):
        """
        Build observation vector of length (n_seg * 2 + 2):

        Layout:
          obs[0]                      = global heat flux (PhiFluxSum)
          obs[1 + i]                  = local heat flux of segment i
                                        for i in [0, n_seg-1]
          obs[1 + n_seg + i]          = mean kinetic energy of probe group i
                                        (computed from 8x30 probe grid, columns sliced
                                         contiguously into n_seg groups; each group's
                                         KE is averaged over its own probe count)
          obs[-1]                     = global kinetic energy
        """
        obs = np.zeros(self.obs_numbers, dtype=np.float32)

        # --- 0. global heat flux ---
        obs[0] = sim.get_global_heat_flux()

        # --- 1. per-segment local heat flux ---
        # obs[1 + i] for i in [0..n_seg-1]
        for i in range(self.n_seg):
            obs[1 + i] = sim.get_local_phi_flux(int(i))

        # --- 2. per-segment mean KE from probes ---
        n_rows = 8  # probe rows
        n_cols = 30  # probe cols (horizontal direction)

        # We'll slice columns into n_seg contiguous chunks.
        # Group g gets cols [col_start, col_end)
        # col_start = floor(g     * n_cols / n_seg)
        # col_end   = floor((g+1) * n_cols / n_seg)
        def col_major_idx(row, col):
            # flatten (row, col) into a single probe index,
            # consistent with how you are reading get_local_velocity
            return col * n_rows + row

        for g in range(self.n_seg):
            col_start = int(np.floor(g * n_cols / self.n_seg))
            col_end = int(np.floor((g + 1) * n_cols / self.n_seg))

            E_sum = 0.0
            count = 0

            for col in range(col_start, col_end):
                for row in range(n_rows):
                    idx = col_major_idx(row, col)
                    vx = sim.get_local_velocity(idx, 0)
                    vy = sim.get_local_velocity(idx, 1)
                    E_sum += vx * vx + vy * vy
                    count += 1

            # average KE density for this group
            if count > 0:
                E_avg = E_sum / count
            else:
                # shouldn't happen unless n_seg > n_cols, but let's be safe
                E_avg = 0.0

            obs[1 + self.n_seg + g] = E_avg

        # --- 3. global kinetic energy ---
        obs[-1] = sim.get_global_kinetic_energy()

        return obs

    def _normalize_obs(self, obs_raw: np.ndarray) -> np.ndarray:
        n = self.n_seg
        obs_norm = obs_raw.copy().astype(np.float32)
        eps = 1e-8

        obs_norm[0] = obs_norm[0] / (self.base_gen_Nu + eps)
        obs_norm[1:1 + n] = obs_norm[1:1 + n] / (self.base_loc_Nu + eps)
        obs_norm[1 + n:1 + 2 * n] = obs_norm[1 + n:1 + 2 * n] / (self.base_loc_KE + eps)
        obs_norm[-1] = obs_norm[-1] / (self.base_gen_KE + eps)
        return obs_norm

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    def _compute_reward_raw(self, obs: np.ndarray) -> float:
        """
        Physics-based reward before baseline subtraction / scaling.

        Following the Rayleigh-Bénard control idea:
            Reward_Nu   = 0.9985 * gen_Nu   + 0.0015 * loc_Nu
            Reward_kinEn = 0.4    * gen_kinEn + 0.6    * loc_kinEn
            raw_reward   = Reward_Nu - Reward_kinEn

        Mapped to our obs:
            gen_Nu    = obs[0]
            loc_Nu    = mean(obs[1 : 1+n_seg])
            loc_kinEn  = mean(obs[1+n_seg : 1+2*n_seg])
            gen_kinEn  = obs[-1]
        """
        n = self.n_seg

        gen_Nu = float(obs[0])
        seg_fluxes = obs[1:1 + n]
        loc_Nu = float(np.mean(seg_fluxes)) if n > 0 else 0.0

        seg_ke = obs[1 + n:1 + 2 * n]
        loc_kinEn = float(np.mean(seg_ke)) if n > 0 else 0.0

        gen_kinEn = float(obs[-1])

        Reward_Nu = ((1 - self.beta) * gen_Nu + self.beta * loc_Nu) / self.Nu_base
        Reward_kinEn = (0.4 * gen_kinEn + 0.6 * loc_kinEn) / self.KE_base

        alpha = 0.5
        offset_reward = 3.0
        # return Reward_Nu - alpha * Reward_kinEn + offset_reward
        return 2.67 - Reward_Nu

    def _compute_reward(self, obs: np.ndarray) -> float:
        """
        Final reward exposed to the RL algorithm.
        We subtract the baseline reward (measured after warmup with T=2),
        then scale by reward_scale.
        """
        raw_now = self._compute_reward_raw(obs)
        shaped = raw_now / self.reward_scale
        return shaped

    # ------------------------------------------------------------------
    # Gym API: reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Starts a new episode:
        - Create a new CFD solver instance in C++.
        - Set the bottom wall to a uniform baseline temperature (2.0).
        - Advance simulation to warmup_time (baseline / uncontrolled flow).
        - Measure baseline_reward from that state (this defines our "zero").
        - Return the observation after warmup.
        """
        # super().reset(seed=seed)

        if self._done and self.episode >= 1:
            self.episode += 1
            self._done = False

        self.agents = list(self.possible_agents)
        self.step_count = 0
        self._pending_actions.clear()

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
            # new solver instance (pass IDs to help with logging on C++ side)
            self.nc_base = test_2d.natural_convection_from_sph_cpp(
                self.parallel_envs, self.episode, 0
            )

            # apply baseline boundary: wall = 2.0 everywhere
            self.nc_base.set_segment_temperatures([2.0] * self.n_seg)

            # run uncontrolled/baseline flow up to warmup_time seconds of sim time
            self.sim_time = float(self.warmup_time)
            self.nc_base.run_case(self.sim_time)

            # observe after warmup
            self.baseline_obs = self._read_observation(self.nc_base).astype(np.float32)

            # if (self.Nu_base is None) or (self.KE_base is None):
            eps = 1e-8
            self.base_gen_Nu = float(self.baseline_obs[0]) + eps
            seg_fluxes_baseline = self.baseline_obs[1:1 + self.n_seg]
            self.base_loc_Nu = float(np.mean(seg_fluxes_baseline)) + eps
            seg_KE_baseline = self.baseline_obs[1 + self.n_seg:1 + 2 * self.n_seg]
            self.base_loc_KE = float(np.mean(seg_KE_baseline)) + eps
            self.base_gen_KE = float(self.baseline_obs[-1]) + eps
            self.Nu_base = (1 - self.beta) * self.base_gen_Nu + self.beta * self.base_loc_Nu
            self.KE_base = 0.4 * self.base_gen_KE + 0.6 * self.base_loc_KE

            # define baseline_reward = how "good" the uncontrolled baseline is
            # if self.baseline_reward is None:
            #     self.baseline_reward = self._compute_reward_raw(self.baseline_obs)

        self.step_to_load = _latest_restart_step(self.restart_dir)
        if self.step_to_load < 0:
            raise RuntimeError("No restart files found under training_results/restart. "
                               "Make sure episode 1 finished the warm-up.")

        self.nc = test_2d.natural_convection_from_sph_cpp(self.parallel_envs, self.episode, int(self.step_to_load))
        self.nc.run_case(self.warmup_time)
        self.sim_time = float(self.warmup_time)

        # housekeeping
        self.total_reward_per_episode = 0.0

        # return obs to RL
        obs0 = self._normalize_obs(self._read_observation(self.nc).astype(np.float32))
        self._last_obs = obs0.copy()
        obs_dict = {agent: obs0 for agent in self.agents}
        infos = {agent: {"episode": self.episode} for agent in self.agents}
        return obs_dict, infos  # observation: Dict[agent, obs],  infos: Dict[agent, dict]

    # ------------------------------------------------------------------
    # Gym API: step
    # ------------------------------------------------------------------
    def step(self, actions: Dict[str, np.ndarray]):
        """
        Multistep episode:
        - Take an action vector of length n_seg (segment temperatures).
        - Send these segment temps to C++.
        - Advance CFD by delta_time seconds of sim time.
        - Observe, compute reward (baseline-subtracted), return and terminate.
        """
        # 回合已结束，强制要求上层 reset
        if self._done:
            return {}, {}, {}, {}, {}

        # 收动作（并允许分批到达）；当收齐 10 个后统一推进 CFD
        self._pending_actions.update({k: np.array(v).reshape(-1)[0] for k, v in actions.items()})

        # 如果没收齐，就保持状态不变（PettingZoo parallel 约定：可在一次调用中就传齐；常见做法是训练端总是一次传齐）
        if len(self._pending_actions) < self.n_seg:
            obs = self._last_obs if self._last_obs is not None else \
                self._normalize_obs(self._read_observation(self.nc).astype(np.float32))
            return (
                {a: obs for a in self.agents},
                {a: 0.0 for a in self.agents},
                {a: False for a in self.agents},
                {a: False for a in self.agents},
                {a: {} for a in self.agents},
            )

        # 2. convert the action vector to per-segment temperatures
        raw = np.array([self._pending_actions[f"agent_{i}"] for i in range(self.n_seg)], dtype=np.float32)
        seg_temps = self._build_segment_temps(self._pending_actions)
        self._pending_actions.clear()

        # 3. tell the solver to apply these temps on the wall
        self.nc.set_segment_temperatures(seg_temps)

        # 4. figure out new target simulation time
        end_time = self.sim_time + self.delta_time

        # 5. advance CFD to end_time
        self.nc.run_case(end_time)
        self.step_count += 1

        # 6. read final observation
        obs = self._read_observation(self.nc).astype(np.float32)

        # 7. compute shaped reward and accumulate episode return
        reward_now = self._compute_reward(obs)
        self.total_reward_per_episode += reward_now

        # 8. advance internal simulation clock
        self.sim_time = end_time

        # 9. optional logging
        with open(os.path.join(self.log_dir, f'action_env{self.parallel_envs}_epi{self.episode}.txt'), 'a') as f:
            f.write(f"clock: {self.sim_time:.6f}  raw_action: {raw.tolist()}  seg_temps: {seg_temps}\n")

        with open(os.path.join(self.log_dir, f'reward_env{self.parallel_envs}_epi{self.episode}.txt'), 'a') as f:
            f.write(f'clock: {self.sim_time:.6f} | reward: {reward_now:.6f} | temps: {seg_temps}\n')

        # 10. check termination condition
        if not self.deterministic:
            episode_limit = self.max_steps_per_episode
        else:
            episode_limit = self.max_steps_per_episode_eval

        terminated = self.step_count >= episode_limit
        truncated = False

        if terminated:
            # 只标记 done；不在这里自增 episode 和清空 agents
            self._done = True
            total_log = os.path.join(self.log_dir, f'reward_env{self.parallel_envs}.txt')
            with open(total_log, 'a', encoding='utf-8') as file:
                file.write(f'episode: {self.episode}  total_reward: {self.total_reward_per_episode:.6f}\n')

        # 11. Gym API return
        obs_norm = self._normalize_obs(obs)
        self._last_obs = obs_norm.copy()
        obs_dict = {agent: obs_norm for agent in self.agents}
        reward_dict = {agent: float(reward_now) for agent in self.agents}
        term_dict = {agent: terminated for agent in self.agents}
        trunc_dict = {agent: truncated for agent in self.agents}
        info_dict = {agent: {"sim_time": self.sim_time, "temps": seg_temps} for agent in self.agents}
        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict

    def render(self):
        return 0

    def _render_frame(self):
        return 0

    def close(self):
        return 0
