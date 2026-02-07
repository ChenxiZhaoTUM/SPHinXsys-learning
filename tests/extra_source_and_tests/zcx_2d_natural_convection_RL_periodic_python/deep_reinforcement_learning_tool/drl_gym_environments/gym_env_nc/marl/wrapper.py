import os
import time
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import glob


# ------------------------------
#  Utilities: action->temperatures, recenter
# ------------------------------
def actions_to_segment_temps(
        raw_actions: np.ndarray,
        baseline_T: float = 2.0,
        ampl: float = 0.75,
        Tmin: float = 0.0,
        Tmax: float = 4.0,
) -> np.ndarray:
    """
    Map joint actions in [-1, 1] to physically safe segment temperatures.
    This matches the "demean + normalize by max abs" strategy
    """
    a = np.asarray(raw_actions, dtype=np.float32).reshape(-1)
    a = np.clip(a, -1.0, 1.0)
    centered = a - float(np.mean(a))
    K2 = max(1.0, float(np.max(np.abs(centered))))
    temps = baseline_T + ampl * centered / K2
    return np.clip(temps, Tmin, Tmax).astype(np.float32)


def recenter_grid_vignon(
        grid: np.ndarray,
        seg_index: int,
        n_seg: int,
) -> np.ndarray:
    """
    grid: (Ny, Nx, 3)
    shift columns (axis=1) so that segment `seg_index` is moved to the middle segment.

    Requirement: Nx % n_seg == 0
    """
    if grid.ndim != 3:
        raise ValueError(f"grid must be (Ny, Nx, 3), got {grid.shape}")
    Ny, Nx, C = grid.shape
    if C != 3:
        raise ValueError(f"grid last dim must be 3 (u,v,T), got {C}")
    if Nx % n_seg != 0:
        raise ValueError(f"Need Nx % n_seg == 0 for block recenter. Nx={Nx}, n_seg={n_seg}")

    cols_per_seg = Nx // n_seg
    center_seg = (n_seg - n_seg // 2) - 1
    shift_cols = (center_seg - seg_index) * cols_per_seg
    return np.roll(grid, shift=shift_cols, axis=1)


# ------------------------------
# Windows-friendly filesystem helpers
# ------------------------------
def _rmtree_retry(path: str, retry: int = 30, sleep: float = 0.1) -> None:
    """Best-effort recursive delete with retries (useful for Windows transient locks)."""
    for _ in range(retry):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(sleep)
    if os.path.isdir(path):
        shutil.rmtree(path)


def _safe_remove(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except FileNotFoundError:
        pass


# ------------------------------
#  File-sync protocol
# ------------------------------
@dataclass
class SyncPaths:
    root: str
    parallel_envs: int
    n_seg: int

    def cfd_root(self) -> str:
        return os.path.join(self.root, f"CFD_n{self.parallel_envs}")

    def ep_dir(self, episode: int) -> str:
        return os.path.join(self.cfd_root(), f"EP_{episode}")

    def act_dir(self, episode: int, actuation: int) -> str:
        return os.path.join(self.ep_dir(episode), f"Actions_actuation{actuation}")

    def action_file(self, episode: int, actuation: int, inv_id: int) -> str:
        return os.path.join(self.act_dir(episode, actuation), f"seg_{inv_id}.npy")

    def result_file(self, episode: int, actuation: int) -> str:
        return os.path.join(self.ep_dir(episode), f"Results_actuation{actuation}.npz")

    def done_flag(self, episode: int, actuation: int) -> str:
        return os.path.join(self.ep_dir(episode), f"is_finished_Actuation{actuation}.flag")

    def baseline_flag(self) -> str:
        return os.path.join(self.cfd_root(), "baseline_done.flag")

    def episode_ready_flag(self, episode: int) -> str:
        return os.path.join(self.ep_dir(episode), "_episode_ready.flag")

    def log_dir(self) -> str:
        return os.path.join(self.root, f"logs_env_{self.parallel_envs}")

    def agent_episode_reward_file(self, episode: int, inv_id: int) -> str:
        # 每个 agent 一个临时文件
        return os.path.join(self.log_dir(), f"ep_{episode:06d}_agent_{inv_id:03d}.npy")

    # ---- folder A: per-episode step curve ----
    def step_curve_dir(self) -> str:
        return os.path.join(self.log_dir(), "mean_reward_curve")

    def step_curve_file(self, episode: int) -> str:
        return os.path.join(self.step_curve_dir(), f"episode_{episode:06d}.txt")

    # ---- folder B: per-episode total mean return ----
    def episode_curve_dir(self) -> str:
        return os.path.join(self.log_dir(), "mean_reward_by_episode")

    def episode_curve_file(self) -> str:
        return os.path.join(self.episode_curve_dir(), f"reward_env_{self.parallel_envs}.txt")


# ------------------------------
# Wrapper
# ------------------------------
class Wrapper:
    """
    - Each pseudo-env (inv_id in [0..n_seg-1]) writes its scalar action.
    - Leader inv_id==0 waits for all actions, runs ONE CFD step, writes result.
     - Followers wait for result file and read it.

    Public API
    - prepare_episode(...)
    - ensure_baseline_once(...)
    - merge_action(...)
    - publish_initial_state(...)
    - step_shared(...)
    - wait_result(...)
    - append_mean_reward(...)
    """

    def __init__(
            self,
            sync_root: str,
            parallel_envs: int,
            n_seg: int,
            n_rows: int,
            n_cols: int,
            avg_len: int = 4,
            warmup_time: float = 400.0,
            delta_time: float = 2.0,
            poll_dt: float = 0.001,
    ):
        self.paths = SyncPaths(sync_root, parallel_envs, n_seg)
        self.n_seg = int(n_seg)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.avg_len = int(avg_len)
        self.warmup_time = float(warmup_time)
        self.delta_time = float(delta_time)
        self.poll_dt = float(poll_dt)

        # Leader-side moving average history (followers do not use this)
        self._probe_hist: List[np.ndarray] = []

        os.makedirs(self.paths.cfd_root(), exist_ok=True)

        if self.n_cols % self.n_seg != 0:
            raise ValueError(f"Need n_cols % n_seg == 0. n_cols={self.n_cols}, n_seg={self.n_seg}")

    # --------------------------
    # Public: episode directory mgmt
    # --------------------------
    def prepare_episode(self, episode: int, is_leader: bool, clean: bool = True) -> None:
        """
        只允许 leader 创建/清理 EP 目录；follower 等待 leader 写好 ready flag。
        这一步必须是“屏障(barrier)”，否则 Windows 目录竞争必炸。
        """
        epd = self.paths.ep_dir(episode)
        ready = self.paths.episode_ready_flag(episode)

        if is_leader:
            # If epd is accidentally created as a file, remove it.
            if os.path.isfile(epd):
                _safe_remove(epd)

            if clean and os.path.isdir(epd):
                _rmtree_retry(epd)

            os.makedirs(epd, exist_ok=True)

            # Remove stale flags/results from previous runs.
            _safe_remove(ready)
            for f in glob.glob(os.path.join(epd, "is_finished_Actuation*.flag")):
                _safe_remove(f)
            for f in glob.glob(os.path.join(epd, "Results_actuation*.npz")):
                _safe_remove(f)
            for d in glob.glob(os.path.join(epd, "Actions_actuation*")):
                if os.path.isdir(d):
                    _rmtree_retry(d)

            # write ready flag, release follower
            open(ready, "w").close()

        else:
            # follower：only wait, do not make mkdir/rmtree
            while not os.path.isfile(ready):
                time.sleep(self.poll_dt)

    # --------------------------
    # Public: merge action
    # --------------------------
    def merge_action(self, episode: int, actuation: int, inv_id: int, action_scalar: float) -> None:
        epd = self.paths.ep_dir(episode)
        os.makedirs(epd, exist_ok=True)

        actd = self.paths.act_dir(episode, actuation)
        os.makedirs(actd, exist_ok=True)

        f = self.paths.action_file(episode, actuation, inv_id)
        tmp = f + ".tmp.npy"
        np.save(tmp, np.array([float(action_scalar)], dtype=np.float32))
        os.replace(tmp, f)  # atomic-ish on Windows

    # --------------------------
    # Public: wait result
    # --------------------------
    def wait_result(self, episode: int, actuation: int) -> Dict[str, np.ndarray]:
        """
        Block until result is ready, then return the npz contents as a dict.
        """
        ep = int(episode)
        ac = int(actuation)
        flag = self.paths.done_flag(ep, ac)
        res = self.paths.result_file(ep, ac)

        while True:
            if not os.path.isfile(flag):
                time.sleep(self.poll_dt)
                continue
            if not os.path.isfile(res):
                time.sleep(self.poll_dt)
                continue

            try:
                data = np.load(res)
                return {k: data[k] for k in data.files}
            except (FileNotFoundError, PermissionError, OSError, ValueError):
                time.sleep(self.poll_dt)

    # --------------------------
    # Internal: leader waits all actions
    # --------------------------
    def _leader_wait_all_actions(self, episode: int, actuation: int) -> np.ndarray:
        ep = int(episode)
        ac = int(actuation)

        while True:
            ok = True
            for inv_id in range(self.n_seg):
                if not os.path.isfile(self.paths.action_file(ep, ac, inv_id)):
                    ok = False
                    break
            if ok:
                break
            time.sleep(self.poll_dt)

        raw = np.empty(self.n_seg, dtype=np.float32)
        for inv_id in range(self.n_seg):
            raw[inv_id] = float(np.load(self.paths.action_file(ep, ac, inv_id))[0])
        return np.clip(raw, -1.0, 1.0)

    # --------------------------
    # Internal: leader writes result
    # --------------------------
    def _leader_write_result(
            self,
            episode: int,
            actuation: int,
            sim_time: float,
            raw_actions: np.ndarray,
            seg_temps: np.ndarray,
            grid_time_avg: np.ndarray,
            gen_flux: float,
            local_flux: np.ndarray,
    ) -> None:
        ep = int(episode)
        ac = int(actuation)
        res = self.paths.result_file(ep, ac)
        os.makedirs(os.path.dirname(res), exist_ok=True)

        tmp_base = res + ".tmp"
        tmp = tmp_base if tmp_base.endswith(".npz") else (tmp_base + ".npz")

        np.savez(
            tmp,
            sim_time=np.array([sim_time], dtype=np.float32),
            raw_actions=raw_actions.astype(np.float32),
            seg_temps=seg_temps.astype(np.float32),
            grid_time_avg=grid_time_avg.astype(np.float32),  # (Ny, Nx, 3)
            gen_flux=np.array([gen_flux], dtype=np.float32),
            local_flux=local_flux.astype(np.float32),  # (n_seg,)
        )
        os.replace(tmp, res)
        open(self.paths.done_flag(ep, ac), "w").close()

    # --------------------------
    # Internal: leader snapshot + moving average
    # --------------------------
    def _snapshot_grid(self, sim) -> np.ndarray:
        grid = np.empty((self.n_rows, self.n_cols, 3), dtype=np.float32)
        for c in range(self.n_cols):
            for r in range(self.n_rows):
                idx = c * self.n_rows + r
                grid[r, c, 0] = sim.get_local_velocity(idx, 0)
                grid[r, c, 1] = sim.get_local_velocity(idx, 1)
                grid[r, c, 2] = sim.get_local_temperature(idx)
        return grid

    def _init_moving_average(self, snap0: np.ndarray) -> None:
        self._probe_hist = [snap0.copy() for _ in range(self.avg_len)]

    def _update_moving_average(self, snap: np.ndarray) -> np.ndarray:
        self._probe_hist.append(snap.astype(np.float32))
        if len(self._probe_hist) > self.avg_len:
            self._probe_hist = self._probe_hist[-self.avg_len:]
        return np.mean(self._probe_hist, axis=0).astype(np.float32)

    # --------------------------
    # Public: baseline barrier
    # --------------------------
    def ensure_baseline_once(
            self,
            solver_factory: Callable[[int, int, int], object],
            inv_id: int,
            episode: int,
            baseline_temp: float = 2.0,
    ) -> None:
        """
        Only leader (inv_id==0) performs warmup in episode==1; others wait baseline_done.flag.
        """
        ep = int(episode)
        if ep != 1:
            return

        flag = self.paths.baseline_flag()
        if int(inv_id) == 0:
            if not os.path.isfile(flag):
                os.makedirs(self.paths.cfd_root(), exist_ok=True)
                sim = solver_factory(self.paths.parallel_envs, ep, 0)
                sim.set_segment_temperatures([float(baseline_temp)] * self.n_seg)
                sim.run_case(float(self.warmup_time))
                open(flag, "w").close()
        else:
            while not os.path.isfile(flag):
                time.sleep(self.poll_dt)

    # --------------------------
    # Public: publish initial state (actuation=0)
    # --------------------------
    def publish_initial_state(
            self,
            sim: Optional[object],
            solver_factory: Callable[[int, int, int], object],
            episode: int,
            restart_step: int,
            sim_time: float,
            baseline_temp: float = 2.0,
    ) -> Tuple[Dict[str, np.ndarray], object]:
        """
        Leader-only: create/load sim, advance to warmup_time, initialize moving-average,
        then write actuation=0 result (a "reset snapshot").
        """
        ep = int(episode)

        if sim is None:
            sim = solver_factory(self.paths.parallel_envs, ep, int(restart_step))
            sim.run_case(float(self.warmup_time))

        # uniform baseline temps for actuation 0
        seg_temps = np.full((self.n_seg,), float(baseline_temp), dtype=np.float32)
        sim.set_segment_temperatures(seg_temps.tolist())

        snap0 = self._snapshot_grid(sim)
        self._init_moving_average(snap0)
        grid_time_avg = np.mean(self._probe_hist, axis=0).astype(np.float32)

        gen_flux = float(sim.get_global_heat_flux())
        local_flux = np.array([float(sim.get_local_phi_flux(i)) for i in range(self.n_seg)], dtype=np.float32)

        raw_actions = np.zeros((self.n_seg,), dtype=np.float32)
        self._leader_write_result(
            episode=ep,
            actuation=0,
            sim_time=float(sim_time),
            raw_actions=raw_actions,
            seg_temps=seg_temps,
            grid_time_avg=grid_time_avg,
            gen_flux=gen_flux,
            local_flux=local_flux,
        )

        out = {
            "sim_time": np.array([float(sim_time)], dtype=np.float32),
            "raw_actions": raw_actions,
            "seg_temps": seg_temps,
            "grid_time_avg": grid_time_avg,
            "gen_flux": np.array([gen_flux], dtype=np.float32),
            "local_flux": local_flux,
        }
        return out, sim

    # --------------------------
    # Public: shared CFD step (actuation>=1)
    # --------------------------
    def step_shared(
            self,
            sim: Optional[object],
            solver_factory: Callable[[int, int, int], object],
            inv_id: int,
            episode: int,
            actuation: int,
            restart_step: int,
            sim_time: float,
            baseline_temp: float = 2.0,
    ) -> Tuple[Dict[str, np.ndarray], Optional[object]]:
        """
        - Leader (inv_id==0): wait all actions -> run CFD -> write result
        - Follower: wait result -> read result
        """
        ep = int(episode)
        ac = int(actuation)
        inv = int(inv_id)

        if inv == 0:
            raw_actions = self._leader_wait_all_actions(ep, ac)

            if sim is None:
                sim = solver_factory(self.paths.parallel_envs, ep, int(restart_step))
                sim.run_case(float(self.warmup_time))
                snap0 = self._snapshot_grid(sim)
                self._init_moving_average(snap0)

            seg_temps = actions_to_segment_temps(raw_actions, baseline_T=float(baseline_temp))
            sim.set_segment_temperatures(seg_temps.tolist())

            end_time = float(sim_time + self.delta_time)
            sim.run_case(end_time)

            snap = self._snapshot_grid(sim)
            grid_time_avg = self._update_moving_average(snap)

            gen_flux = float(sim.get_global_heat_flux())
            local_flux = np.array([float(sim.get_local_phi_flux(i)) for i in range(self.n_seg)], dtype=np.float32)

            self._leader_write_result(
                episode=ep,
                actuation=ac,
                sim_time=end_time,
                raw_actions=raw_actions,
                seg_temps=seg_temps,
                grid_time_avg=grid_time_avg,
                gen_flux=gen_flux,
                local_flux=local_flux,
            )

            out = {
                "sim_time": np.array([end_time], dtype=np.float32),
                "raw_actions": raw_actions.astype(np.float32),
                "seg_temps": seg_temps.astype(np.float32),
                "grid_time_avg": grid_time_avg.astype(np.float32),
                "gen_flux": np.array([gen_flux], dtype=np.float32),
                "local_flux": local_flux.astype(np.float32),
            }
            return out, sim

        # follower
        out = self.wait_result(ep, ac)
        return out, None

    # --------------------------
    # reward logging (leader only)
    # --------------------------
    def init_step_curve_file(self, episode: int) -> str:
        """Create/overwrite the per-episode step curve file (folder A)."""
        os.makedirs(self.paths.step_curve_dir(), exist_ok=True)
        f = self.paths.step_curve_file(int(episode))

        with open(f, "w", encoding="utf-8") as fp:
            header = ["actuation", "sim_time", "mean_reward"]
            header += [f"reward_seg{i}" for i in range(self.n_seg)]
            fp.write(",".join(header) + "\n")
            return f

    def append_step_mean_reward(
            self,
            episode: int,
            actuation: int,
            sim_time: float,
            mean_reward: float,
            rewards_vec: np.ndarray,
    ) -> None:
        """
        Append one actuation record (folder A):
        actuation,sim_time,mean_reward,reward_seg0,...,reward_seg{n_seg-1}
        """
        os.makedirs(self.paths.step_curve_dir(), exist_ok=True)
        f = self.paths.step_curve_file(int(episode))
        rv = np.asarray(rewards_vec, dtype=np.float32).reshape(-1)

        if rv.size != self.n_seg:
            raise ValueError(f"rewards_vec size mismatch: got {rv.size}, expected {self.n_seg}")

        row = [str(int(actuation)),
               f"{float(sim_time):.6f}",
               f"{float(mean_reward):.6f}",
               ] + [f"{float(x):.6f}" for x in rv.tolist()]

        with open(f, "a", encoding="utf-8", buffering=1) as fp:
            fp.write(",".join(row) + "\n")

    def append_episode_mean_return(self, episode: int, mean_return: float) -> None:
        """Append per-episode total mean return (folder B)."""
        os.makedirs(self.paths.episode_curve_dir(), exist_ok=True)
        f = self.paths.episode_curve_file()
        with open(f, "a", encoding="utf-8", buffering=1) as fp:
            fp.write(f"episode: {int(episode)}  mean_return: {float(mean_return):.6f}\n")
