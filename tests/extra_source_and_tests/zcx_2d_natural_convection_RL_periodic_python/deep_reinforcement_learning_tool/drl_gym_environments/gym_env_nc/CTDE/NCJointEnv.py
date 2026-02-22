import os
import glob
import re
from collections import deque
from typing import Callable, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


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


def actions_to_segment_temps(
        raw_actions: np.ndarray,
        baseline_T: float = 2.0,
        ampl: float = 0.75,
        Tmin: float = 0.0,
        Tmax: float = 4.0,
) -> np.ndarray:
    """Map joint actions in [-1,1] to safe segment temperatures with demean+normalize."""
    a = np.asarray(raw_actions, dtype=np.float32).reshape(-1)
    a = np.clip(a, -1.0, 1.0)
    centered = a - float(np.mean(a))
    K2 = max(1.0, float(np.max(np.abs(centered))))
    temps = baseline_T + ampl * centered / K2
    return np.clip(temps, Tmin, Tmax).astype(np.float32)


class NCJointEnv(gym.Env):
    """
    Joint env (1 CFD group == 1 env):
    - action: joint action (n_seg,)
    - obs: local obs for each agent, shape (n_seg, local_dim) (strict local patch)

    Training reward: ONLY global heat flux (get_global_heat_flux)
    Logging: per-seg diagnostic reward from local_phi_flux(i)

    Solver interface:
    - solver_factory(group_id, episode, restart_step) -> sim
    - sim.set_segment_temperatures(list_len_n_seg)
    - sim.run_case(end_time)
    - sim.get_local_velocity(idx, dim), sim.get_local_temperature(idx)
    - sim.get_global_heat_flux(), sim.get_local_phi_flux(i)
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            solver_factory: Callable[[int, int, int], object],
            group_id: int,
            training_root: str,
            n_seg: int = 10,
            n_rows: int = 8,
            n_cols: int = 30,
            avg_len: int = 4,
            warmup_time: float = 400.0,
            delta_time: float = 2.0,
            max_steps_per_episode: int = 200,
            deterministic_eval: bool = False,
            max_steps_eval_mul: int = 4,
            # reward params (still used for shaping / logging)
            beta: float = 0.0015,  # (kept for compatibility; not used in global reward now)
            nu_target: float = 22.5,
            reward_scale: float = 1.0,
            # action->temperature
            baseline_T: float = 2.0
    ):
        super().__init__()
        self.solver_factory = solver_factory
        self.group_id = int(group_id)

        # ---- persist roots (FIX: training_root must exist as member) ----
        self.training_root = os.path.abspath(training_root)
        self.case_root = _mkdir(os.path.join(self.training_root, f"CFD_n{self.group_id}"))
        for name in ("input", "output", "reload", "restart"):
            _mkdir(os.path.join(self.case_root, name))
        self.restart_dir = os.path.join(self.case_root, "restart")

        # ---- env geometry ----
        self.n_seg = int(n_seg)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.avg_len = int(avg_len)
        self._probe_dim = 3

        if self.n_cols % self.n_seg != 0:
            raise ValueError(f"Need n_cols % n_seg == 0. n_cols={self.n_cols}, n_seg={self.n_seg}")
        self.cols_per_seg = self.n_cols // self.n_seg
        self.local_dim = self.n_rows * self.cols_per_seg * self._probe_dim

        # ---- time & episode ----
        self.warmup_time = float(warmup_time)
        self.delta_time = float(delta_time)
        self.max_steps = int(max_steps_per_episode)

        self.deterministic = bool(deterministic_eval)
        self.max_steps_eval = int(max_steps_eval_mul) * self.max_steps

        # ---- reward params ----
        self.beta = float(beta)  # kept (not used for global reward)
        self.nu_target = float(nu_target)
        self.reward_scale = float(reward_scale)

        # ---- action->temperature params ----
        self.baseline_T = float(baseline_T)

        # ---- spaces ----
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_seg,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(self.n_seg, self.local_dim), dtype=np.float32
        )

        # ---- runtime state ----
        self.episode = 1
        self.step_count = 0
        self.sim_time = 0.0
        self.restart_step = -1
        self.nc_base = None
        self.nc = None
        self._probe_hist = deque(maxlen=self.avg_len)

        # ---- logging ----
        self.log_root = os.path.join(self.training_root, f"logs_env_{self.group_id}")
        self._step_curve_inited = False
        self._episode_return = 0.0

    # ------------------ probe grid & obs ------------------
    def _probe_index_col_major(self, row: int, col: int) -> int:
        return col * self.n_rows + row

    def _snapshot_grid(self, sim) -> np.ndarray:
        out = np.empty((self.n_rows, self.n_cols, self._probe_dim), dtype=np.float32)
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                idx = self._probe_index_col_major(row, col)
                out[row, col, 0] = float(sim.get_local_velocity(idx, 0))
                out[row, col, 1] = float(sim.get_local_velocity(idx, 1))
                out[row, col, 2] = float(sim.get_local_temperature(idx))
        return out

    def _update_moving_average(self, sim) -> np.ndarray:
        snap = self._snapshot_grid(sim).reshape(-1).astype(np.float32)
        self._probe_hist.append(snap)
        # stack once to avoid list(...) overhead in loops
        arr = np.stack(list(self._probe_hist), axis=0)
        return np.mean(arr, axis=0).astype(np.float32)

    def _global_to_local_obs(self, obs_flat: np.ndarray) -> np.ndarray:
        grid = np.asarray(obs_flat, dtype=np.float32).reshape(self.n_rows, self.n_cols, 3)
        locals_ = np.empty((self.n_seg, self.local_dim), dtype=np.float32)
        for i in range(self.n_seg):
            x0 = i * self.cols_per_seg
            x1 = (i + 1) * self.cols_per_seg
            patch = grid[:, x0:x1, :]
            locals_[i] = patch.reshape(-1)
        return locals_

    # ------------------ rewards ------------------
    def _global_reward_from_genNu(self, gen_Nu: float) -> float:
        # ONLY global heat flux
        return float((self.nu_target - float(gen_Nu)) / float(self.reward_scale))

    def _local_reward_vec(self, local_phi: np.ndarray) -> np.ndarray:
        # diagnostic only (not used in training target)
        r_local = (self.nu_target - local_phi.astype(np.float32) * float(self.n_seg)) / float(self.reward_scale)
        return r_local.astype(np.float32)

    # ------------------ baseline / restart ------------------
    def _ensure_baseline_once(self) -> None:
        os.chdir(self.case_root)
        step = _latest_restart_step(self.restart_dir)
        if step > 0:
            return
        sim = self.solver_factory(self.group_id, 1, 0)
        sim.set_segment_temperatures([self.baseline_T] * self.n_seg)
        sim.run_case(float(self.warmup_time))
        self.nc_base = None  # release handle

    # ------------------ logging helpers ------------------
    # ---- folder A: per-episode step curve ----
    def _step_curve_dir(self) -> str:
        return os.path.join(self.log_root, "mean_reward_curve")

    def _step_curve_file(self, episode: int) -> str:
        return os.path.join(self._step_curve_dir(), f"episode_{int(episode):06d}.txt")

    # ---- folder B: per-episode total mean return ----
    def _episode_curve_dir(self) -> str:
        return os.path.join(self.log_root, "mean_reward_by_episode")

    def _episode_curve_file(self) -> str:
        return os.path.join(self._episode_curve_dir(), f"reward_env_{int(self.group_id)}.txt")

    def init_step_curve_file(self, episode: int) -> str:
        os.makedirs(self._step_curve_dir(), exist_ok=True)
        f = self._step_curve_file(int(episode))
        with open(f, "w", encoding="utf-8") as fp:
            header = ["actuation", "sim_time", "mean_reward"]
            header += [f"reward_seg{i}" for i in range(self.n_seg)]
            fp.write(",".join(header) + "\n")
        self._step_curve_inited = True
        return f

    def append_step_reward(self, episode: int, actuation: int, sim_time: float,
                           step_reward: float, rewards_vec: np.ndarray) -> None:
        os.makedirs(self._step_curve_dir(), exist_ok=True)
        f = self._step_curve_file(int(episode))
        rv = np.asarray(rewards_vec, dtype=np.float32).reshape(-1)
        if rv.size != self.n_seg:
            raise ValueError(f"rewards_vec size mismatch: got {rv.size}, expected {self.n_seg}")

        row = [str(int(actuation)),
               f"{float(sim_time):.6f}",
               f"{float(step_reward):.6f}"] + [f"{float(x):.6f}" for x in rv.tolist()]

        with open(f, "a", encoding="utf-8", buffering=1) as fp:
            fp.write(",".join(row) + "\n")

    def append_episode_return(self, episode: int, episode_return: float) -> None:
        os.makedirs(self._episode_curve_dir(), exist_ok=True)
        f = self._episode_curve_file()
        with open(f, "a", encoding="utf-8", buffering=1) as fp:
            fp.write(f"episode: {int(episode)}  mean_return: {float(episode_return):.6f}\n")

    # ------------------ gym API ------------------
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        os.chdir(self.case_root)

        print(f"[env {self.group_id}] ===== Episode {self.episode} start =====")

        self._ensure_baseline_once()
        self.restart_step = _latest_restart_step(self.restart_dir)
        if self.restart_step < 0:
            raise RuntimeError(f"No restart files found in {self.restart_dir}")

        self.nc = self.solver_factory(self.group_id, self.episode, int(self.restart_step))
        self.nc.run_case(float(self.warmup_time))
        self.sim_time = float(self.warmup_time)

        # reset episode-level state
        self.step_count = 0
        self._episode_return = 0.0
        self._step_curve_inited = False

        # moving average init
        self._probe_hist.clear()
        snap0 = self._snapshot_grid(self.nc).reshape(-1).astype(np.float32)
        for _ in range(self.avg_len):
            self._probe_hist.append(snap0.copy())

        global_obs0 = self._update_moving_average(self.nc)
        local_obs0 = self._global_to_local_obs(global_obs0)

        # init per-episode file
        self.init_step_curve_file(self.episode)

        info = {"episode": self.episode, "group_id": self.group_id, "sim_time": self.sim_time}
        return local_obs0, info

    def step(self, action: np.ndarray):
        os.chdir(self.case_root)

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != self.n_seg:
            raise ValueError(f"Action must have length {self.n_seg}, got {a.shape}")

        temps = actions_to_segment_temps(
            a,
            baseline_T=self.baseline_T
        )
        self.nc.set_segment_temperatures(temps.tolist())

        end_time = float(self.sim_time + self.delta_time)
        self.nc.run_case(end_time)
        self.sim_time = end_time
        self.step_count += 1

        # obs
        global_obs = self._update_moving_average(self.nc)
        obs = self._global_to_local_obs(global_obs)

        # compute flux
        gen_Nu = float(self.nc.get_global_heat_flux())
        local_phi_flux = np.asarray([float(self.nc.get_local_phi_flux(i)) for i in range(self.n_seg)], dtype=np.float32)

        # training reward: global only
        reward = self._global_reward_from_genNu(gen_Nu)
        self._episode_return += float(reward)

        # logging: per-seg diagnostic reward
        rvec_local = self._local_reward_vec(local_phi_flux)

        if not self._step_curve_inited:
            self.init_step_curve_file(self.episode)

        self.append_step_reward(
            episode=self.episode,
            actuation=int(self.step_count),
            sim_time=float(self.sim_time),
            step_reward=float(reward),
            rewards_vec=rvec_local,
        )

        episode_limit = self.max_steps if not self.deterministic else self.max_steps_eval
        terminated = (self.step_count >= episode_limit)
        truncated = False

        if terminated:
            # IMPORTANT: log under the episode that just finished
            finished_ep = int(self.episode)
            self.append_episode_return(finished_ep, self._episode_return)

            self.episode += 1
            self.nc = None
            self._step_curve_inited = False

        info: Dict[str, object] = {
            "episode": self.episode,
            "group_id": self.group_id,
            "sim_time": self.sim_time,
            "gen_Nu": gen_Nu,
            "local_phi_flux": local_phi_flux,  # (n_seg,)
            "temps": temps,  # (n_seg,)
        }

        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.nc = None
        self.nc_base = None
        return
