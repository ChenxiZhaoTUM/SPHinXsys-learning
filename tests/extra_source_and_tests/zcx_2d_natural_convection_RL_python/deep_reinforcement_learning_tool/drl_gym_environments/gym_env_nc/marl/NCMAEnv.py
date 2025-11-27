import sys
import os
import glob, re
import numpy as np
from typing import Dict, List
from collections import deque
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

    观测：8x30 探针的 (u, v, T)，做最近 4 个 CFD 步的滑动平均，再展平成 1D 向量；
         所有 agent 共享同一份观测（集中式训练/分布式执行的常见做法之一）。

    动作：每个 agent 输出 1 个标量；10 个 agent 的动作合并后统一一次施加到 10 段底板温度，
         然后推进一次 CFD（delta_time）。

    奖励：团队共享奖励（同一数值发给全部 agent）：
           r = nu_target - [(1 - beta) * Nu_global + beta * sum_i Nu_local[i]]
         注意 loc_Nu 使用**求和**（对齐你在单智能体里的设定）。
    """

    metadata = {"name": "NCMA-v0"}

    def __init__(self, n_seg: int = 10, warmup_time: float = 400.0, delta_time: float = 2.0,
                 parallel_envs: int = 0):
        super().__init__()

        # ----- identifiers / bookkeeping -----
        self.parallel_envs = parallel_envs
        self.episode = 1

        # ----- control segmentation -----
        assert n_seg > 0
        self.n_seg = n_seg
        self.agents = [f"agent_{i}" for i in range(n_seg)]
        self.possible_agents = list(self.agents)

        # ----- result folder -----
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "../training_process"))
        self.training_root = _mkdir(os.path.join(proj_root, "training_results"))
        self.restart_dir = _mkdir(os.path.join(self.training_root, "restart"))
        self.log_dir = _mkdir(os.path.join(self.training_root, f"logs_env_{self.parallel_envs}"))

        # ----- physical timing -----
        self.step_to_load = 0
        self.warmup_time = float(warmup_time)
        self.delta_time = float(delta_time)
        self.sim_time = 0.0

        # ----- episode length -----
        self.max_steps_per_episode = 200
        self.step_count = 0
        self.deterministic = False
        self._done = False  # 控制回合结束后的 step 行为

        # ----- action space -----
        # 每个 agent 输出 1 个标量动作
        self._act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._pending_actions: Dict[str, float] = {}

        # ----- observation space (u, v, T on 8x30 probes, 4-step moving average) -----
        self.n_rows = 8
        self.n_cols = 30
        self._probe_dim = 3  # u, v, T
        self._obs_len = self.n_rows * self.n_cols * self._probe_dim
        self._obs_space = spaces.Box(low=-1e6, high=1e6, shape=(self._obs_len,), dtype=np.float32)
        self._probe_hist = deque(maxlen=4)
        self._last_obs = None

        # ----- reward parameters -----
        self.beta = 0.0015
        self.reward_scale = 1.0
        self.nu_target = 20.87

        # ----- runtime state -----
        self.nc_base = None
        self.nc = None
        self.total_reward_per_episode = 0.0

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
    # Helpers: probes and actions
    # ------------------------------------------------------------------
    def _build_segment_temps(self, actions_dict: Dict[str, np.ndarray]) -> List[float]:
        """把 n_seg 个 agent 的标量动作合并成 n_seg 段温度（中心化 + 缩放到±0.75，再平移到 2.0）"""
        baseline_T, ampl = 2.0, 0.75
        raw = np.array([float(np.clip(actions_dict[f"agent_{i}"], -1.0, 1.0)) for i in range(self.n_seg)],
                       dtype=np.float32)
        centered = raw - float(np.mean(raw))
        max_abs = float(np.max(np.abs(centered))) if self.n_seg > 0 else 0.0
        scale = (ampl / max_abs) if max_abs > 1e-8 else 0.0
        temps = baseline_T + centered * scale
        return np.clip(temps, 0.0, 4.0).tolist()

    def _probe_index_col_major(self, row: int, col: int) -> int:
        return col * self.n_rows + row

    def _snapshot_probes(self, sim) -> np.ndarray:
        """瞬时 (u, v, T) on 8x30 probe grid, flattened"""
        out = np.empty((self.n_rows, self.n_cols, self._probe_dim), dtype=np.float32)
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                idx = self._probe_index_col_major(row, col)
                out[row, col, 0] = sim.get_local_velocity(idx, 0)  # u
                out[row, col, 1] = sim.get_local_velocity(idx, 1)  # v
                out[row, col, 2] = sim.get_local_temperature(idx)  # T
        return out.reshape(-1)

    def _read_observation(self, sim) -> np.ndarray:
        """time-averaged (last 4 CFD steps) probe fields (u,v,T), flattened"""
        snap = self._snapshot_probes(sim)
        self._probe_hist.append(snap)
        hist = list(self._probe_hist)
        obs = np.mean(hist, axis=0).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_team_reward(self) -> float:
        """r = nu_target - [(1 - beta) * Nu_global + beta * sum_i Nu_local[i]]"""
        gen_Nu = float(self.nc.get_global_heat_flux())
        loc_vals = [float(self.nc.get_local_phi_flux(i)) for i in range(self.n_seg)]
        loc_sum = float(np.sum(loc_vals)) if self.n_seg > 0 else 0.0
        mix_nu = (1.0 - self.beta) * gen_Nu + self.beta * loc_sum
        reward = (self.nu_target - mix_nu) / self.reward_scale

        # 调试日志（可选）
        # try:
        #     dbg_path = os.path.join(self.log_dir, f"nu_debug_env{self.parallel_envs}_epi{self.episode}.txt")
        #     with open(dbg_path, "a", encoding="utf-8") as f:
        #         f.write(f"t={self.sim_time:.3f}, gen_Nu={gen_Nu:.6f}, loc_sum={loc_sum:.6f}, "
        #                 f"mixNu={mix_nu:.6f}, reward={reward:.6f}\n")
        # except Exception:
        #     pass

        return reward

    # ------------------------------------------------------------------
    # PettingZoo API: reset & step
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # 回合切换标记
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
            # baseline warmup
            self.nc_base = test_2d.natural_convection_from_sph_cpp(self.parallel_envs, self.episode, 0)
            self.nc_base.set_segment_temperatures([2.0] * self.n_seg)
            self.sim_time = float(self.warmup_time)
            self.nc_base.run_case(self.sim_time)

        self.step_to_load = _latest_restart_step(self.restart_dir)
        if self.step_to_load < 0:
            raise RuntimeError("No restart files found under training_results/restart. "
                               "Make sure episode 1 finished the warm-up.")

        self.nc = test_2d.natural_convection_from_sph_cpp(self.parallel_envs, self.episode, int(self.step_to_load))
        self.nc.run_case(self.warmup_time)
        self.sim_time = float(self.warmup_time)

        # 观测历史初始化
        self._probe_hist.clear()
        snap0 = self._snapshot_probes(self.nc)
        for _ in range(4):
            self._probe_hist.append(snap0.copy())
        obs0 = self._read_observation(self.nc).astype(np.float32)
        self._last_obs = obs0.copy()

        self.total_reward_per_episode = 0.0

        obs_dict = {agent: obs0 for agent in self.agents}
        infos = {agent: {"episode": self.episode} for agent in self.agents}
        return obs_dict, infos

    def step(self, actions: Dict[str, np.ndarray]):
        # 回合已结束，强制上层 reset
        if self._done:
            return {}, {}, {}, {}, {}

        # 收动作；未收齐则回传上一观测，奖励=0
        self._pending_actions.update({k: np.array(v).reshape(-1)[0] for k, v in actions.items()})
        if len(self._pending_actions) < self.n_seg:
            obs = self._last_obs if self._last_obs is not None else self._read_observation(self.nc).astype(np.float32)
            return (
                {a: obs for a in self.agents},
                {a: 0.0 for a in self.agents},
                {a: False for a in self.agents},
                {a: False for a in self.agents},
                {a: {} for a in self.agents},
            )

        # 合并动作 -> 温度并施加
        raw = np.array([self._pending_actions[f"agent_{i}"] for i in range(self.n_seg)], dtype=np.float32)
        seg_temps = self._build_segment_temps(self._pending_actions)
        self._pending_actions.clear()
        self.nc.set_segment_temperatures(seg_temps)

        # 2) advance CFD
        end_time = self.sim_time + self.delta_time
        self.nc.run_case(end_time)
        self.step_count += 1

        # 新观测 + 奖励
        obs = self._read_observation(self.nc).astype(np.float32)
        reward_now = self._compute_team_reward()
        self.total_reward_per_episode += reward_now
        self.sim_time = end_time

        # 日志
        with open(os.path.join(self.log_dir, f'action_env{self.parallel_envs}_epi{self.episode}.txt'), 'a') as f:
            f.write(f"clock: {self.sim_time:.6f}  raw_action: {raw.tolist()}  seg_temps: {seg_temps}\n")
        with open(os.path.join(self.log_dir, f'reward_env{self.parallel_envs}_epi{self.episode}.txt'), 'a') as f:
            f.write(f'clock: {self.sim_time:.6f} | reward: {reward_now:.6f} | temps: {seg_temps}\n')

        # 终止判断
        terminated = self.step_count >= self.max_steps_per_episode
        truncated = False
        if terminated:
            self._done = True
            total_log = os.path.join(self.log_dir, f'reward_env{self.parallel_envs}.txt')
            with open(total_log, 'a', encoding='utf-8') as file:
                file.write(f'episode: {self.episode}  total_reward: {self.total_reward_per_episode:.6f}\n')

        # 返回
        self._last_obs = obs.copy()
        obs_dict = {agent: obs for agent in self.agents}
        reward_dict = {agent: float(reward_now) for agent in self.agents}  # 团队共享奖励
        term_dict = {agent: terminated for agent in self.agents}
        trunc_dict = {agent: truncated for agent in self.agents}
        info_dict = {agent: {"sim_time": self.sim_time, "temps": seg_temps} for agent in self.agents}
        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict

    def render(self):
        return 0

    def close(self):
        return 0
