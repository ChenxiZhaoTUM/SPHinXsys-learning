import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# TODO: replace with an env var or auto-discovery
sys.path.append(r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_natural_convection_RL_python\lib\Release")
import zcx_2d_natural_convection_RL_python as test_2d


class NCEnvironment(gym.Env):
    """
    Single-agent environment for natural convection control.

    Each action controls n_seg heater segments on the bottom wall.

    Action (per episode step):
        a[0:n_seg]: target temperature (or temperature command) for each segment.

    Observation (returned by reset() and step()):
        obs is length (2 * n_seg + 2), laid out as:
            obs[0]                    = global heat flux (PhiFluxSum)
            obs[1 + i]                = local heat flux of segment i
                                        for i in [0 .. n_seg-1]
            obs[1 + n_seg + i]        = mean kinetic energy of probe block i
                                        computed from the 8x30 probe grid,
                                        columns split contiguously into n_seg groups,
                                        then averaged to remove group-size bias
            obs[-1]                   = global kinetic energy

    Reward (shaped):
        We want to enhance heat transfer and suppress convection intensity.
        Following the style of the Rayleigh-Bénard MARL code:

            Reward_Nus   = 0.9985 * gen_Nus   + 0.0015 * loc_Nus
            Reward_kinEn = 0.4    * gen_kinEn + 0.6    * loc_kinEn

        where
            gen_Nus   = global heat flux
            loc_Nus   = mean per-segment heat flux
            gen_kinEn = global kinetic energy
            loc_kinEn = mean per-segment local KE

        raw_reward = Reward_Nus - Reward_kinEn

        final_reward = (raw_reward - baseline_reward) / reward_scale

        baseline_reward is measured after reset(), when the wall is held at
        uniform temperature 2.0 for warmup_time seconds (the "baseline" case).

    Episode structure:
        - This environment is multistep episodic.
          On reset(), we create a fresh CFD simulation, apply a baseline
          boundary temperature of 2.0 on the lower wall, run for warmup_time,
          measure baseline_reward, and return the observation.
        - On step(action), we:
            - pass the segment temperatures to the solver,
            - advance by delta_time in simulation time,
            - read obs and compute reward,
            - terminate the episode.
        - So each reset/step pair is a full "episode".
    """

    metadata = {}

    def __init__(self, render_mode=None, parallel_envs=0, n_seg=3):
        super().__init__()

        # ----- identifiers / bookkeeping -----
        self.parallel_envs = parallel_envs  # which parallel env this is (for logging)
        self.episode = 1  # episode counter

        # ----- control segmentation -----
        # number of bottom-wall control segments
        self.n_seg = int(n_seg)

        # ----- physical timing -----
        # warmup_time: run baseline before first action
        # delta_time: CFD physical time per control step
        self.warmup_time = 120.0
        self.delta_time = 5.0
        # TODO: test how long it will take to stabilize after the temperature is changed, or it does not need steady solution
        # running simulation time cursor (absolute physical time in solver)
        self.sim_time = 0.0

        # ---------------- episode length in terms of control actions ----------------
        self.max_steps_per_episode = 10  # TRAINING length: 10 actions per episode
        self.step_count = 0
        self.max_steps_per_episode_eval = 4 * self.max_steps_per_episode  # for evaluation
        self.deterministic = False  # training mode by default

        # ----- action space -----
        # The agent directly outputs n_seg scalar commands.
        self.action_low = np.full(self.n_seg, -1.0, dtype=np.float32)
        self.action_high = np.full(self.n_seg, 1.0, dtype=np.float32)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        # ----- observation space -----
        # obs dimension = 2*n_seg + 2:
        #   [global flux] +
        #   [n_seg local flux per segment] +
        #   [n_seg local KE per segment group from probes] +
        #   [global KE]
        self.obs_numbers = self.n_seg * 2 + 2
        obs_low = np.full(self.obs_numbers, -1e6, dtype=np.float32)
        obs_high = np.full(self.obs_numbers, 1e6, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # ----- reward shaping / normalization -----
        # We'll compute a "baseline_reward" in reset() after warmup,
        # and subtract it in step() to stabilize learning.
        self.baseline_reward = None  # will be set in reset()
        self.reward_scale = 1.0

        # ----- runtime state -----
        self.nc = None  # the C++ solver / simulation handle
        self.total_reward_per_episode = 0.0

    # ------------------------------------------------------------------
    # Helper: produce the per-segment temperature array we send to C++
    # ------------------------------------------------------------------
    def _build_segment_temps(self, action_vec: np.ndarray):
        """
        Map raw agent actions -> safe per-segment wall temperatures.

        We do three things:
        1. Interpret the action as relative commands per segment (not direct temperature).
        2. Normalize them so they don't bias the whole wall too hot or too cold.
        3. Scale+shift them around a physical baseline temperature (2.0),
           with a limited amplitude 'ampl'.

        This mimics the spirit of Tfunc.apply_T:
        - Keep wall temps near baseline (2.0).
        - Enforce max contrast between segments.
        - Prevent runaway heating.
        """

        if len(action_vec) != self.n_seg:
            raise ValueError(
                f"Expected action of length {self.n_seg}, got {len(action_vec)}"
            )

        # ---- hyperparameters you can tune ----
        baseline_T = 2.0
        ampl = 0.75  # max +/- amplitude around baseline (like 2.0 ± 0.75)

        # 1) get raw command per segment as float32
        raw = np.asarray(action_vec, dtype=np.float32)
        raw = np.clip(raw, -1.0, 1.0)

        # 2) mean-center them so "all segments high" doesn't just globally heat up
        # This is similar to subtracting Mean in the original Tfunc.
        mean_raw = float(np.mean(raw))
        centered = raw - mean_raw  # now average(centered) == 0

        # 3) compute how large the centered variations are vs. our allowed amplitude
        max_abs = float(np.max(np.abs(centered))) if self.n_seg > 0 else 0.0
        if max_abs < 1e-8:
            scale = 0.0
        else:
            # We want max deviation -> 'ampl' at most
            scale = ampl / max_abs  # if max_abs > 1, this scales down the pattern

        scaled = centered * scale  # each segment now in [-ampl, +ampl] approximately

        # 4) shift back around baseline_T
        temps = baseline_T + scaled  # final segment temperatures

        # 5) final physical safety clip (just in case)
        T_min = 0.0
        T_max = 4.0
        temps = np.clip(temps, T_min, T_max)

        # 6) return as python list for pybind11 -> std::vector<double>
        return temps.astype(np.float32).tolist()

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _read_observation(self):
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
        obs[0] = self.nc.get_global_heat_flux()

        # --- 1. per-segment local heat flux ---
        # obs[1 + i] for i in [0..n_seg-1]
        for i in range(self.n_seg):
            obs[1 + i] = self.nc.get_local_phi_flux(int(i))

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
                    vx = self.nc.get_local_velocity(idx, 0)
                    vy = self.nc.get_local_velocity(idx, 1)
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
        obs[-1] = self.nc.get_global_kinetic_energy()

        return obs

    def _normalize_obs(self, obs_raw: np.ndarray) -> np.ndarray:
        """
        Hook for observation normalization.
        Currently passthrough.
        In future you can:
          - divide flux terms by baseline flux
          - divide KE terms by baseline KE
          - clip or log-scale
        """
        return obs_raw

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    def _compute_reward_raw(self, obs: np.ndarray) -> float:
        """
        Physics-based reward before baseline subtraction / scaling.

        Following the Rayleigh-Bénard control idea:
            Reward_Nus   = 0.9985 * gen_Nus   + 0.0015 * loc_Nus
            Reward_kinEn = 0.4    * gen_kinEn + 0.6    * loc_kinEn
            raw_reward   = Reward_Nus - Reward_kinEn

        Mapped to our obs:
            gen_Nus    = obs[0]
            loc_Nus    = mean(obs[1 : 1+n_seg])
            loc_kinEn  = mean(obs[1+n_seg : 1+2*n_seg])
            gen_kinEn  = obs[-1]
        """
        n = self.n_seg

        gen_Nus = float(obs[0])
        seg_fluxes = obs[1:1 + n]
        loc_Nus = float(np.mean(seg_fluxes)) if n > 0 else 0.0

        seg_ke = obs[1 + n:1 + 2 * n]
        loc_kinEn = float(np.mean(seg_ke)) if n > 0 else 0.0

        gen_kinEn = float(obs[-1])

        Reward_Nus = 0.9985 * gen_Nus + 0.0015 * loc_Nus
        Reward_kinEn = 0.4 * gen_kinEn + 0.6 * loc_kinEn

        return Reward_Nus - Reward_kinEn

    def _compute_reward(self, obs: np.ndarray) -> float:
        """
        Final reward exposed to the RL algorithm.
        We subtract the baseline reward (measured after warmup with T=2),
        then scale by reward_scale.
        """
        raw_now = self._compute_reward_raw(obs)
        shaped = (raw_now - self.baseline_reward) / self.reward_scale
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
        super().reset(seed=seed)

        # new solver instance (pass IDs to help with logging on C++ side)
        self.nc = test_2d.natural_convection_from_sph_cpp(
            self.parallel_envs, self.episode
        )

        # apply baseline boundary: wall = 2.0 everywhere
        # TODO: provide one of these in your C++ bindings.
        if hasattr(self.nc, "set_segment_temperatures"):
            baseline_vec = [2.0] * self.n_seg
            self.nc.set_segment_temperatures(baseline_vec)
        else:
            raise RuntimeError(
                "C++ solver missing set_segment_temperatures(...) needed for baseline."
            )

        # run uncontrolled/baseline flow up to warmup_time seconds of sim time
        self.sim_time = float(self.warmup_time)
        self.nc.run_case(self.sim_time)

        # observe after warmup
        baseline_obs = self._read_observation().astype(np.float32)

        # define baseline_reward = how "good" the uncontrolled baseline is
        if self.baseline_reward is None:
            self.baseline_reward = self._compute_reward_raw(baseline_obs)

        # housekeeping
        self.step_count = 0
        self.total_reward_per_episode = 0.0

        # return obs to RL
        return baseline_obs, {}

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
        # 1. validate action
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != self.n_seg:
            raise ValueError(
                f"Action must have length {self.n_seg}, got shape={action}"
            )

        # 2. convert the action vector to per-segment temperatures
        seg_temps = self._build_segment_temps(a)  # list[float] of length n_seg

        # 3. tell the solver to apply these temps on the wall
        # TODO: implement this in C++ and expose in pybind11:
        #   void set_segment_temperatures(const std::vector<Real>& temps);
        if hasattr(self.nc, "set_segment_temperatures"):
            self.nc.set_segment_temperatures(seg_temps)
        else:
            raise RuntimeError(
                "C++ solver has no set_segment_temperatures(...) binding."
            )

        # 4. figure out new target simulation time
        end_time = self.sim_time + self.delta_time

        # 5. advance CFD to end_time
        self.nc.run_case(end_time)
        self.step_count += 1

        # 6. read final observation
        obs = self._read_observation().astype(np.float32)

        # 7. compute shaped reward and accumulate episode return
        reward_now = self._compute_reward(obs)
        self.total_reward_per_episode += reward_now

        # 8. advance internal simulation clock
        self.sim_time = end_time

        # 9. optional logging
        with open(f'action_env{self.parallel_envs}_epi{self.episode}.txt', 'a') as f:
            f.write(
                f"clock: {self.sim_time:.6f}  raw_action: {a.tolist()}  seg_temps: {seg_temps}\n"
            )
        with open(f'reward_env{self.parallel_envs}_epi{self.episode}.txt', 'a') as f:
            f.write(
                f'clock: {self.sim_time:.6f}  reward: {reward_now:.6f}\n'
            )

        # 10. check termination condition
        if not self.deterministic:
            episode_limit = self.max_steps_per_episode
        else:
            episode_limit = self.max_steps_per_episode_eval

        if self.step_count >= episode_limit:
            terminated = True
            truncated = False

            # log total episode reward
            with open(f'reward_env{self.parallel_envs}.txt', 'a') as file:
                file.write(
                    f'episode: {self.episode}  total_reward: '
                    f'{self.total_reward_per_episode:.6f}\n'
                )

            # increment episode counter for next reset
            self.episode += 1
        else:
            terminated = False
            truncated = False

        # 11. Gym API return
        return obs, float(reward_now), terminated, truncated, {}

    def render(self):
        return 0

    def _render_frame(self):
        return 0

    def close(self):
        return 0
