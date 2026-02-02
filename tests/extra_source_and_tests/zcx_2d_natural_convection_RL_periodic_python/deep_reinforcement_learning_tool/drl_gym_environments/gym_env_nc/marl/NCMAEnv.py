import sys
import os
import glob, re
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

# ---- Bind to the SPHinXsys C++ interface ----
sys.path.append(r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_natural_convection_RL_periodic_python\lib"
                r"\Release")
import zcx_2d_natural_convection_RL_periodic_python as test_2d


def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _latest_restart_step(restart_dir: str) -> int:
    """Return the largest step index found in '<name>_rst_<step>.xml' files, or -1 if none."""
    steps: List[int] = []
    for f in glob.glob(os.path.join(restart_dir, "*_rst_*.xml")):
        m = re.search(r"_rst_(\d+)\.xml$", f.replace("\\", "/"))
        if m:
            steps.append(int(m.group(1)))
    mx = max(steps) if steps else -1
    return mx if mx > 0 else -1


class NCMAEnvParallel(ParallelEnv):
    """
    Parallel multi-agent environment for natural-convection control (PettingZoo 'parallel' API).

    - Observation:
        We sample an 8x30 probe grid of (u, v, T). A 4-frame moving average is kept.
        Each agent i receives a "recentered" view

    - Action:
        Each agent outputs a single scalar in [-1, 1]. All agents' actions are collected; once we
        have one action per segment, we map them to segment temperatures with mean-centering and
        amplitude limiting, then advance the CFD by `delta_time`.

    - Reward (per agent):
        r_i = nu_target - [ (1 - beta) * Nu_global + beta * Nu_local(i) * local_scale ]
        where `local_scale` is a tunable gain to emphasize locality (kept from your version = 10).
    """

    metadata = {"name": "NCMA-v0"}

    # ------------------------ Initialization ------------------------
    def __init__(self, n_seg: int = 10, warmup_time: float = 400.0, delta_time: float = 2.0,
                 parallel_envs: int = 0):
        super().__init__()

        # ----- segmentation / agents -----
        assert n_seg > 0
        self.n_seg = int(n_seg)
        self.possible_agents = [f"agent_{i}" for i in range(self.n_seg)]
        self.agents = list(self.possible_agents)

        # ----- directories -----
        # keep all runtime artifacts under drl/training_process
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "../training_process"))
        self.training_root = _mkdir(os.path.join(proj_root, "training_results_multi"))
        self.restart_dir = _mkdir(os.path.join(self.training_root, "restart"))
        self.parallel_envs = parallel_envs
        self.log_dir = _mkdir(os.path.join(self.training_root, f"logs_env_{self.parallel_envs}"))

        # ----- time control -----
        self.warmup_time = float(warmup_time)
        self.delta_time = float(delta_time)
        self.sim_time = 0.0
        self.episode = 1
        self.max_steps_per_episode = 200
        self.step_count = 0
        self.deterministic = False  # (kept for parity with single-agent env)

        # ----- probe grid & observation space -----
        self.n_rows = 8
        self.n_cols = 30
        self._probe_dim = 3  # (u, v, T)
        self._avg_len = 4  # moving-average window length
        self._probe_hist: deque[np.ndarray] = deque(maxlen=self._avg_len)

        # unified per-segment column width to keep obs length fixed
        self._cols_per_seg_max = int(np.ceil(self.n_cols / self.n_seg))
        self._seg_bounds = [(int(np.floor(s * self.n_cols / self.n_seg)),
                             int(np.floor((s + 1) * self.n_cols / self.n_seg)))
                            for s in range(self.n_seg)]
        self.hor_inv_probes = self.n_cols // self.n_seg  # columns per segment
        self._obs_len = self.n_rows * self.n_cols * self._probe_dim

        # ----- spaces -----
        self._act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._obs_space = spaces.Box(low=-1e6, high=1e6, shape=(self._obs_len,), dtype=np.float32)
        self._last_obs = None

        # ----- reward parameters -----
        self.beta = 0.0015
        self.reward_scale = 1.0
        self.nu_target = 22.5

        # ----- runtime handles -----
        self.nc_base = None  # warmup/baseline solver
        self.nc = None  # training solver
        self.total_reward_per_episode = 0.0

        if self.n_cols % self.n_seg != 0:
            raise ValueError("Need n_cols % n_seg == 0 for Vignon-style block recenter.")

    # ------------------------ PettingZoo spaces ------------------------
    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    @property
    def observation_spaces(self):
        return {agent: self._obs_space for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self._act_space for agent in self.possible_agents}

    # ------------------------ Segmenting / recenter helpers ------------------------
    def _seg_col_range(self, seg: int) -> Tuple[int, int]:
        """
        Return [start, end) column indices for segment `seg` using floor-based equal slicing.
        """
        start = int(np.floor(seg * self.n_cols / self.n_seg))
        end = int(np.floor((seg + 1) * self.n_cols / self.n_seg))
        return start, end

    def _snapshot_grid(self, sim) -> np.ndarray:
        """
        Instantaneous probe grid (n_rows, n_cols, 3) with (u, v, T).
        The C++ API uses column-major probe indexing: idx = col * n_rows + row.
        """
        grid = np.empty((self.n_rows, self.n_cols, self._probe_dim), dtype=np.float32)
        for c in range(self.n_cols):
            for r in range(self.n_rows):
                idx = c * self.n_rows + r
                grid[r, c, 0] = sim.get_local_velocity(idx, 0)
                grid[r, c, 1] = sim.get_local_velocity(idx, 1)
                grid[r, c, 2] = sim.get_local_temperature(idx)
        return grid

    def _time_avg_grid(self) -> np.ndarray:
        """
        Moving-average over the last `_avg_len` frames. Returns (n_rows, n_cols, 3).
        """
        hist = list(self._probe_hist)
        return np.mean(hist, axis=0).astype(np.float32)

    def _recenter_obs(self, grid_time_avg: np.ndarray, i: int) -> np.ndarray:
        mid_seg = self.n_seg // 2
        shift_seg = (i - mid_seg) % self.n_seg
        shift_cols = shift_seg * self.hor_inv_probes
        shifted = np.roll(grid_time_avg, shift=-shift_cols, axis=1)
        # flatten to 1D observation vector
        return shifted.reshape(-1).astype(np.float32)

    # ------------------------ Reward ------------------------
    def _compute_rewards(self) -> Dict[str, float]:
        """
        Per-agent reward using its own local Nusselt number plus the global term:
            r_i = nu_target - [ (1 - beta) * Nu_global + beta * local_scale * Nu_local(i) ]
        Notes:
            - `local_scale` is a tunable gain (kept as 10.0 from your previous version).
        """
        gen_Nu = float(self.nc.get_global_heat_flux())
        rewards = {}
        for i, agent in enumerate(self.agents):
            loc_Nu_i = float(self.nc.get_local_phi_flux(int(i)))
            mix = (1.0 - self.beta) * gen_Nu + self.beta * loc_Nu_i * self.n_seg
            r = (self.nu_target - mix) / self.reward_scale
            rewards[agent] = float(r)
        return rewards

    # ------------------------ Reset / Step ------------------------
    def reset(self, seed=None, options=None):
        """
        Create (or reuse) the CFD solver, run warm-up (episode 1), then load from the latest
        restart and return the initial observations for all agents.
        """
        # mark all agents as active for a new episode
        self.agents = list(self.possible_agents)
        self.step_count = 0

        # banner + logs
        msg = f"[env {self.parallel_envs}] ===== Episode {self.episode} start ====="
        print(msg)
        _mkdir(self.log_dir)
        with open(os.path.join(self.log_dir, "episodes.txt"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        # warm-up on episode 1 to generate restart state
        os.chdir(self.training_root)
        if self.episode == 1:
            self.nc_base = test_2d.natural_convection_from_sph_cpp(self.parallel_envs, self.episode, 0)
            self.nc_base.set_segment_temperatures([2.0] * self.n_seg)
            self.sim_time = float(self.warmup_time)
            self.nc_base.run_case(self.sim_time)

        step_to_load = _latest_restart_step(self.restart_dir)
        if step_to_load < 0:
            raise RuntimeError("No restart files found under training_process/restart.")
        # create the training solver from the latest restart and advance to warm-up time
        self.nc = test_2d.natural_convection_from_sph_cpp(self.parallel_envs, self.episode, int(step_to_load))
        self.nc.run_case(self.warmup_time)
        self.sim_time = float(self.warmup_time)

        self.total_reward_per_episode = 0.0

        # initialize the 4-frame moving average with the current snapshot
        self._probe_hist.clear()
        snap0 = self._snapshot_grid(self.nc)
        for _ in range(self._avg_len):
            self._probe_hist.append(snap0.copy())

        grid_time_avg = self._time_avg_grid()
        obs_dict = {agent: self._recenter_obs(grid_time_avg, i) for i, agent in enumerate(self.agents)}
        infos = {agent: {"episode": self.episode} for agent in self.agents}
        self._last_obs = obs_dict
        return obs_dict, infos

    def _build_segment_temps(self, actions_dict: Dict[str, np.ndarray]) -> List[float]:
        """
        Map per-agent scalar actions to segment temperatures.
        - Mean-center to avoid global heating drift.
        - Limit amplitude to keep temperatures in a safe band around baseline.
        """
        baseline_T, ampl = 2.0, 0.75
        raw = np.empty(self.n_seg, dtype=np.float32)
        for i in range(self.n_seg):
            val = float(np.asarray(actions_dict[f"agent_{i}"]).reshape(-1)[0])
            raw[i] = float(np.clip(val, -1.0, 1.0))

        centered = raw - float(np.mean(raw))
        K2 = max(1.0, float(np.max(np.abs(centered))))
        temps = baseline_T + ampl * centered / K2
        return np.clip(temps, 0.0, 4.0).tolist()

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Collect actions (may arrive in parts); once we have all n_seg actions:
        - Convert actions to segment temperatures
        - Advance CFD by `delta_time`
        - Produce recentered observations, per-agent rewards, and termination flags.
        """
        # 1) collect actions
        if set(actions.keys()) != set(self.agents):
            raise ValueError("ParallelEnv.step requires a full action dict for all active agents.")

        # 2) build temperatures and advance CFD
        seg_temps = self._build_segment_temps(actions)

        self.nc.set_segment_temperatures(seg_temps)
        end_time = self.sim_time + self.delta_time
        self.nc.run_case(end_time)
        self.step_count += 1
        self.sim_time = end_time

        # 3) observations (moving average)
        snap = self._snapshot_grid(self.nc)
        self._probe_hist.append(snap)
        grid_time_avg = self._time_avg_grid()
        obs_dict = {agent: self._recenter_obs(grid_time_avg, i) for i, agent in enumerate(self.agents)}
        self._last_obs = obs_dict

        # 4) rewards
        reward_dict = self._compute_rewards()
        self.total_reward_per_episode += float(np.mean(list(reward_dict.values())))

        # 5) logging (actions + rewards)
        with open(os.path.join(self.log_dir, f"action_env{self.parallel_envs}_epi{self.episode}.txt"), "a") as f:
            f.write(f"clock: {self.sim_time:.6f}  seg_temps: {seg_temps}\n")

        if reward_dict:
            mean_rew = float(np.mean(list(reward_dict.values())))
            rew_items = ", ".join(f"{k}:{v:.6f}" for k, v in reward_dict.items())
        else:
            mean_rew = 0.0
            rew_items = "(empty)"

        with open(os.path.join(self.log_dir, f"reward_env{self.parallel_envs}_epi{self.episode}.txt"), "a") as f:
            f.write(f"clock: {self.sim_time:.6f} | mean_reward: {mean_rew:.6f} | per_agent: {rew_items}\n")

        # 6) termination check
        if self.step_count >= self.max_steps_per_episode:
            term = {a: True for a in self.agents}
            trunc = {a: False for a in self.agents}
            with open(os.path.join(self.log_dir, f"reward_env{self.parallel_envs}.txt"), "a", encoding="utf-8") as f:
                f.write(f"episode: {self.episode}  total_reward: {self.total_reward_per_episode:.6f}\n")
            self.episode += 1
            self.agents = []
        else:
            term = {a: False for a in self.agents}
            trunc = {a: False for a in self.agents}

        info_dict = {a: {"sim_time": self.sim_time, "temps": seg_temps} for a in self.agents}
        return obs_dict, reward_dict, term, trunc, info_dict

    # ------------------------ Render / Close ------------------------
    def render(self):
        return 0

    def _render_frame(self):
        return 0

    def close(self):
        return 0


def _debug_print_blocks(arr_1d, n_rows, n_cols, n_seg, probe_dim, title=""):
    """Print per-segment mean for a chosen channel (0) to see where the 'hot' block is."""
    grid = arr_1d.reshape(n_rows, n_cols, probe_dim)
    ch0 = grid[:, :, 0]  # look at channel 0
    cols_per_seg = n_cols // n_seg
    seg_means = []
    for s in range(n_seg):
        x0 = s * cols_per_seg
        x1 = (s + 1) * cols_per_seg
        seg_means.append(float(np.mean(ch0[:, x0:x1])))
    print(title, "seg_means(ch0):", ["{:.1f}".format(v) for v in seg_means])


def test_recenter_unit(env: "NCMAEnvParallel"):
    """
    Unit-test recenter without running CFD:
    - Build synthetic grid where segment i has value 1, others 0 (on channel 0 only).
    - Recenter for agent i should move that block to mid_seg.
    """
    Ny, Nx, C = env.n_rows, env.n_cols, env._probe_dim
    n_seg = env.n_seg
    cols_per_seg = env.hor_inv_probes
    mid_seg = n_seg // 2

    print(f"[TEST] Ny={Ny}, Nx={Nx}, C={C}, n_seg={n_seg}, cols_per_seg={cols_per_seg}, mid_seg={mid_seg}")

    ok = True
    for i in range(n_seg):
        grid = np.zeros((Ny, Nx, C), dtype=np.float32)

        # put a clear marker only in segment i, channel 0
        x0 = i * cols_per_seg
        x1 = (i + 1) * cols_per_seg
        grid[:, x0:x1, 0] = 1.0

        rec = env._recenter_obs(grid, i).reshape(Ny, Nx, C)

        # expected: the '1.0' block is now at mid_seg
        mx0 = mid_seg * cols_per_seg
        mx1 = (mid_seg + 1) * cols_per_seg

        # check mean inside mid block should be ~1, outside should be ~0
        inside = float(np.mean(rec[:, mx0:mx1, 0]))
        outside_left = float(np.mean(rec[:, :mx0, 0])) if mx0 > 0 else 0.0
        outside_right = float(np.mean(rec[:, mx1:, 0])) if mx1 < Nx else 0.0

        cond = (inside > 0.99) and (outside_left < 1e-3) and (outside_right < 1e-3)
        if not cond:
            ok = False
            print(f"[FAIL] agent={i}: inside={inside:.3f}, outL={outside_left:.3e}, outR={outside_right:.3e}")
            # print per-seg means for diagnosis
            _debug_print_blocks(grid.reshape(-1), Ny, Nx, n_seg, C, title=f"orig(i={i})")
            _debug_print_blocks(rec.reshape(-1), Ny, Nx, n_seg, C, title=f"recenter(i={i})")
            # hint about direction
            print("Hint: if the hot block lands in the wrong place, try flipping roll sign: "
                  "`np.roll(..., shift=+shift_cols, axis=1)` instead of `-shift_cols`.")
            break

    if ok:
        print("[PASS] recenter unit-test passed for all agents.")


if __name__ == "__main__":
    # Build env object only for testing recenter math (no CFD needed).
    # We won't call env.reset() or env.step() here.
    env = NCMAEnvParallel(n_seg=10, parallel_envs=0)

    # run the unit test
    test_recenter_unit(env)
