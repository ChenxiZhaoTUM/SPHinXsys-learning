# NCPseudoEnv.py
import os
import glob
import re
from typing import Callable, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from .wrapper import Wrapper, recenter_grid_vignon
except ImportError:
    from wrapper import Wrapper, recenter_grid_vignon


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


class NCPseudoEnv(gym.Env):
    """
    One pseudo-env per segment (inv_id = 0..n_seg-1).
    A group (parallel_envs) of n_seg pseudo-env processes shares ONE CFD via disk sync (Wrapper).

    Design aligned with NCEnvironment:
      - episode 1: baseline warmup barrier
      - each reset: load from latest restart step, publish actuation=0 snapshot
      - each step: actuation = step_count + 1, leader runs CFD once, all envs read same result
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            solver_factory: Callable[[int, int, int], object],
            parallel_envs: int,
            inv_id: int,
            n_seg: int = 10,
            warmup_time: float = 400.0,
            delta_time: float = 2.0,
            max_steps_per_episode: int = 200,
            n_rows: int = 8,
            n_cols: int = 30,
            avg_len: int = 4,
            beta: float = 0.0015,
            nu_target: float = 22.5,
            reward_scale: float = 1.0,
            baseline_temp: float = 2.0,
            training_root: Optional[str] = None,
            restart_dir: Optional[str] = None,
            poll_dt: float = 0.05,
    ):
        super().__init__()

        self.solver_factory = solver_factory
        self.parallel_envs = int(parallel_envs)
        self.inv_id = int(inv_id)
        self.n_seg = int(n_seg)

        self.warmup_time = float(warmup_time)
        self.delta_time = float(delta_time)
        self.max_steps = int(max_steps_per_episode)

        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.avg_len = int(avg_len)

        self.beta = float(beta)
        self.nu_target = float(nu_target)
        self.reward_scale = float(reward_scale)
        self.baseline_temp = float(baseline_temp)

        self.episode_return = 0.0
        self.mean_episode_return = 0.0

        if not (0 <= self.inv_id < self.n_seg):
            raise ValueError(f"inv_id must be in [0, n_seg-1], got {self.inv_id}")
        if self.n_cols % self.n_seg != 0:
            raise ValueError(f"Need n_cols % n_seg == 0. n_cols={self.n_cols}, n_seg={self.n_seg}")

        # ---- default paths (match NCEnvironment style) ----
        if training_root is None:
            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "../training_process"))
            training_root = os.path.join(proj_root, "training_results_multi")
        self.training_root = _mkdir(os.path.abspath(training_root))

        if restart_dir is None:
            restart_dir = os.path.join(self.training_root, "restart")
        self.restart_dir = _mkdir(os.path.abspath(restart_dir))

        # Wrapper (public API only)
        self.wrap = Wrapper(
            sync_root=self.training_root,
            parallel_envs=self.parallel_envs,
            n_seg=self.n_seg,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            avg_len=self.avg_len,
            warmup_time=self.warmup_time,
            delta_time=self.delta_time,
            poll_dt=poll_dt,
        )

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(self.n_rows * self.n_cols * 3,), dtype=np.float32
        )

        # episode state
        self.episode = 1
        self.step_count = 0
        self.sim_time = float(self.warmup_time)
        self.restart_step = -1

        # leader keeps solver handle, followers keep None
        self._sim: Optional[object] = None

        # ---- logging ----
        self.log_root = os.path.join(self.training_root, f"logs_env_{self.parallel_envs}")
        self._step_curve_inited = False

    @property
    def is_leader(self) -> bool:
        return self.inv_id == 0

    # -------- obs / reward from shared result --------
    def _obs_from_result(self, result: Dict[str, np.ndarray]) -> np.ndarray:
        grid = result["grid_time_avg"]  # (Ny, Nx, 3)
        grid_rc = recenter_grid_vignon(grid, seg_index=self.inv_id, n_seg=self.n_seg)
        return grid_rc.reshape(-1).astype(np.float32)

    def _reward_from_result(self, result: Dict[str, np.ndarray]) -> float:
        gen_flux = float(result["gen_flux"][0])
        local_flux = np.asarray(result["local_flux"], dtype=np.float32).reshape(-1)
        loc_i = float(local_flux[self.inv_id])
        mix = (1.0 - self.beta) * gen_flux + self.beta * loc_i * self.n_seg
        return float((self.nu_target - mix) / self.reward_scale)

    def _all_rewards_from_result(self, result: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute rewards for all segments (shape: [n_seg]).
        Leader uses this to log 10 rewards + mean at each actuation.
        """
        gen_flux = float(result["gen_flux"][0])
        local_flux = np.asarray(result["local_flux"], dtype=np.float32).reshape(-1)  # (n_seg,)
        mixes = (1.0 - self.beta) * gen_flux + self.beta * local_flux * float(self.n_seg)
        rewards = (self.nu_target - mixes) / float(self.reward_scale)
        return rewards.astype(np.float32)

    # -------- Gym API --------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # ---- Episode start banner ----
        if self.is_leader:
            print(f"[env {self.parallel_envs}] ===== Episode {self.episode} start =====")

        self.step_count = 0
        self.sim_time = float(self.warmup_time)
        self.episode_return = 0.0
        self.mean_episode_return = 0.0

        # solver 很可能依赖相对路径
        os.chdir(self.training_root)

        # 清理本 episode 的 EP 目录（只让 leader clean）
        self.wrap.prepare_episode(self.episode, is_leader=self.is_leader, clean=self.is_leader)

        # episode 1 baseline barrier（只 leader 跑，其他等 flag）
        self.wrap.ensure_baseline_once(
            solver_factory=self.solver_factory,
            inv_id=self.inv_id,
            episode=self.episode,
            baseline_temp=self.baseline_temp,
        )

        # 只在 reset 扫一次 restart_step，并缓存
        self.restart_step = _latest_restart_step(self.restart_dir)
        if self.restart_step < 0:
            raise RuntimeError(
                f"No restart files found in {self.restart_dir}. "
                f"Make sure baseline/warmup produced restart XMLs."
            )

        # actuation=0：发布/等待初始状态
        if self.is_leader:
            result0, self._sim = self.wrap.publish_initial_state(
                sim=None,
                solver_factory=self.solver_factory,
                episode=self.episode,
                restart_step=self.restart_step,
                sim_time=self.sim_time,
                baseline_temp=self.baseline_temp,
            )
        else:
            result0 = self.wrap.wait_result(self.episode, actuation=0)
            self._sim = None

        obs0 = self._obs_from_result(result0)
        info = {
            "episode": int(self.episode),
            "inv_id": int(self.inv_id),
            "actuation": 0,
            "sim_time": float(self.sim_time),
        }

        # leader: init per-episode step curve file (folder A)
        if self.is_leader:
            self.wrap.init_step_curve_file(self.episode)
            self._step_curve_inited = True

        return obs0, info

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        a = float(np.clip(a, -1.0, 1.0))

        actuation = self.step_count + 1  # 1..max_steps

        # 1) 写入本 pseudo-env 的 action
        self.wrap.merge_action(
            episode=self.episode,
            actuation=actuation,
            inv_id=self.inv_id,
            action_scalar=a,
        )

        # 2) shared step：leader 跑 CFD，followers 等待结果
        result, maybe_sim = self.wrap.step_shared(
            sim=self._sim if self.is_leader else None,
            solver_factory=self.solver_factory,
            inv_id=self.inv_id,
            episode=self.episode,
            actuation=actuation,
            restart_step=self.restart_step,
            sim_time=self.sim_time,
            baseline_temp=self.baseline_temp,
        )
        if self.is_leader:
            self._sim = maybe_sim

        # 3) 解析 result -> obs/reward
        self.sim_time = float(result["sim_time"][0])
        obs = self._obs_from_result(result)
        reward = self._reward_from_result(result)
        self.episode_return += float(reward)

        # leader: write per-actuation 10 rewards + mean reward (folder A)
        if self.is_leader:
            if not self._step_curve_inited:
                self.wrap.init_step_curve_file(self.episode)
                self._step_curve_inited = True

            rewards_all = self._all_rewards_from_result(result)  # (n_seg,)
            mean_r = float(np.mean(rewards_all))
            self.mean_episode_return += mean_r
            self.wrap.append_step_mean_reward(
                episode=self.episode,
                actuation=actuation,
                sim_time=self.sim_time,
                mean_reward=mean_r,
                rewards_vec=rewards_all,
            )

        self.step_count += 1
        terminated = (self.step_count >= self.max_steps)
        truncated = False

        if terminated:
            if self.is_leader:
                # leader: append episode total mean return (folder B)
                self.wrap.append_episode_mean_return(self.episode, self.mean_episode_return)

            # 3) 准备下一个 episode
            self.episode += 1
            self.episode_return = 0.0
            self.mean_episode_return = 0.0
            self._sim = None
            self._step_curve_inited = False

        info = {
            "episode": int(self.episode),
            "inv_id": int(self.inv_id),
            "actuation": int(actuation),
            "sim_time": float(self.sim_time),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self._sim = None
        return
