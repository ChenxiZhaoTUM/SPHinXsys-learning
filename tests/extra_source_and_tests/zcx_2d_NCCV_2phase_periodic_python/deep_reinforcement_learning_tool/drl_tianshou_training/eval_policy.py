#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a trained shared SAC policy for Vignon-style invariant MARL (pseudo-envs).

- Loads policy.pth
- Creates only test group (parallel_envs=999) with n_seg pseudo-env processes
- Runs rollout for eval_steps steps
- Switches control off after switch_off_step (action=0)
- Reads gen_flux / gen_ke from Results_actuation*.npz in sync/CFD_n999/EP_xxxxxx
"""

import os
import sys
import glob
import argparse
import datetime
from functools import partial
import numpy as np
import torch
import torch.nn as nn

from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import SACPolicy
from tianshou.data import Batch


# ---------------------------
# Env factory (top-level, Windows spawn safe)
# ---------------------------
def make_env(
    *,
    sph_lib_path: str,
    training_root: str,
    sync_root: str,
    restart_dir: str,
    parallel_envs: int,
    inv_id: int,
    n_seg: int,
    n_rows: int,
    n_cols: int,
    warmup_time: float,
    delta_time: float,
    avg_len: int,
    max_steps: int,
    beta: float,
    baseline_temp: float,
):
    if sph_lib_path not in sys.path:
        sys.path.insert(0, sph_lib_path)

    import zcx_2d_NCCV_2phase_periodic_python as test_2d

    # ensure gym env code importable
    this_dir = os.path.dirname(os.path.abspath(__file__))
    drl_dir = os.path.dirname(this_dir)
    gym_root = os.path.join(drl_dir, "drl_gym_environments")
    for p in (this_dir, gym_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    from gym_env_nc.marl.NCPseudoEnv import NCPseudoEnv

    return NCPseudoEnv(
        solver_factory=test_2d.natural_convection_from_sph_cpp,
        training_root=training_root,
        sync_root=sync_root,
        restart_dir=restart_dir,
        parallel_envs=parallel_envs,
        inv_id=inv_id,
        n_seg=n_seg,
        n_rows=n_rows,
        n_cols=n_cols,
        warmup_time=warmup_time,
        delta_time=delta_time,
        avg_len=avg_len,
        # IMPORTANT: allow one long episode so we can do control-on then control-off
        max_steps_per_episode=max_steps,
        beta=beta,
        baseline_temp=baseline_temp,
    )


def load_flux_ke_trace(ep_dir: str, max_actuation: int):
    """Read Results_actuation{a}.npz for a=0..max_actuation (best-effort)."""
    times, fluxes, kes = [], [], []
    for a in range(0, max_actuation + 1):
        f = os.path.join(ep_dir, f"Results_actuation{a}.npz")
        if not os.path.isfile(f):
            if a > 0:
                break
            continue
        d = np.load(f)
        t = float(np.asarray(d["sim_time"]).reshape(-1)[0])
        fl = float(np.asarray(d["gen_flux"]).reshape(-1)[0])
        ke = float(np.asarray(d["gen_ke"]).reshape(-1)[0])
        times.append(t)
        fluxes.append(fl)
        kes.append(ke)
    return times, fluxes, kes


def build_policy_from_spaces(
    obs_space, act_space, device,
    hidden_sizes, actor_lr, critic_lr,
    gamma, tau, alpha, auto_alpha, alpha_lr,
):
    obs_shape = obs_space.shape
    act_shape = act_space.shape
    max_action = float(act_space.high[0])

    net_a = Net(obs_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh, device=device)
    actor = ActorProb(
        net_a, act_shape, device=device, max_action=max_action,
        conditioned_sigma=True
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

    net_c1 = Net(obs_shape, act_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh,
                 concat=True, device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)

    net_c2 = Net(obs_shape, act_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh,
                 concat=True, device=device)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    alpha_val = float(alpha)
    if auto_alpha:
        prod = np.prod(act_shape)
        target_entropy = -float(prod) if np.isscalar(prod) else -float(prod.item())
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha_val = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha_val,
        estimation_step=1,
        action_space=act_space,
    )
    return policy


def get_args():
    p = argparse.ArgumentParser()

    default_policy_dir = r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_NCCV_2phase_periodic_python\training_process\training_results_pseudo\logs\NC-v0-marl\sac\0\260304-194903"
    default_policy_path = os.path.join(default_policy_dir, "policy.pth")

    p.add_argument("--policy-path", type=str, default=default_policy_path)
    p.add_argument(
        "--run-dir", type=str,
        default=r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_NCCV_2phase_periodic_python\training_process\training_results_pseudo",
    )
    p.add_argument(
        "--sph-lib-path", type=str,
        default=r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_NCCV_2phase_periodic_python\lib\Release",
    )

    # env config (should match training)
    p.add_argument("--n-seg", type=int, default=10)
    p.add_argument("--n-rows", type=int, default=8)
    p.add_argument("--n-cols", type=int, default=30)
    p.add_argument("--warmup-time", type=float, default=400.0)
    p.add_argument("--delta-time", type=float, default=2.0)
    p.add_argument("--avg-len", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.0015)
    p.add_argument("--baseline-temp", type=float, default=2.0)

    # eval control (YOUR REQUEST)
    p.add_argument("--joint-episodes", type=int, default=1)
    p.add_argument(
        "--switch-off-step", type=int, default=200,
        help="Run control for first 200 steps, then force action=0 afterwards."
    )
    p.add_argument(
        "--eval-steps", type=int, default=400,
        help="Total steps to run per joint episode. Should be > switch-off-step."
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # network/policy (must match training)
    p.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--auto-alpha", action="store_true", default=True)
    p.add_argument("--alpha-lr", type=float, default=5e-4)

    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    if args.eval_steps <= args.switch_off_step:
        raise ValueError("eval_steps must be > switch_off_step so we can observe control removed phase.")

    policy_path = os.path.abspath(args.policy_path)
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(policy_path)

    run_dir = os.path.abspath(args.run_dir)
    sync_root = os.path.join(run_dir, "sync")
    restart_dir_t = os.path.join(sync_root, "CFD_n999", "restart")
    if not os.path.isdir(restart_dir_t) or len(glob.glob(os.path.join(restart_dir_t, "*_rst_*.xml"))) == 0:
        raise RuntimeError(
            f"Test restart_dir not ready: {restart_dir_t}\n"
            f"Need restart XMLs. Make sure this run_dir is the one that produced your policy."
        )

    # build test envs: n_seg pseudo-envs in group 999
    test_env_fns = []
    for inv_id in range(args.n_seg):
        test_env_fns.append(
            partial(
                make_env,
                sph_lib_path=args.sph_lib_path,
                training_root=run_dir,
                sync_root=sync_root,
                restart_dir=restart_dir_t,
                parallel_envs=999,
                inv_id=inv_id,
                n_seg=args.n_seg,
                n_rows=args.n_rows,
                n_cols=args.n_cols,
                warmup_time=args.warmup_time,
                delta_time=args.delta_time,
                avg_len=args.avg_len,
                # IMPORTANT: long eval episode
                max_steps=args.eval_steps,
                beta=args.beta,
                baseline_temp=args.baseline_temp,
            )
        )
    envs = SubprocVectorEnv(test_env_fns)

    # spaces
    obs_space = envs.observation_space
    act_space = envs.action_space
    if isinstance(obs_space, (list, tuple)):
        obs_space = obs_space[0]
    if isinstance(act_space, (list, tuple)):
        act_space = act_space[0]

    # build policy and load weights
    policy = build_policy_from_spaces(
        obs_space, act_space, args.device,
        hidden_sizes=args.hidden_sizes,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        gamma=args.gamma, tau=args.tau,
        alpha=args.alpha, auto_alpha=args.auto_alpha, alpha_lr=args.alpha_lr
    )
    state = torch.load(policy_path, map_location=args.device)
    policy.load_state_dict(state)
    policy.eval()
    print("[eval] loaded policy:", policy_path)
    print("[eval] run_dir:", run_dir)
    print(f"[eval] switch_off_step={args.switch_off_step}, eval_steps={args.eval_steps} (dt={args.delta_time})")

    # output CSV
    ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    out_csv = os.path.join(run_dir, "logs", f"eval_flux_trace_{ts}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", encoding="utf-8") as fcsv:
        fcsv.write("joint_episode,actuation,sim_time,gen_flux,gen_ke,control_on\n")

        for j in range(args.joint_episodes):
            obs, info = envs.reset()

            # try to infer EP number from leader info; fallback to j+1
            ep_num = j + 1
            try:
                if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict):
                    ep_num = int(info[0].get("episode", ep_num))
            except Exception:
                pass

            # rollout fixed eval_steps steps
            for t in range(args.eval_steps):
                with torch.no_grad():
                    obs_arr = np.asarray(obs, dtype=np.float32)
                    n_env = obs_arr.shape[0]
                    batch = Batch(obs=obs_arr, info=[{} for _ in range(n_env)])  # provide batch.info
                    out = policy(batch)
                    act = out.act
                    if isinstance(act, torch.Tensor):
                        act = act.detach().cpu().numpy()

                # control removed after switch_off_step (200): force action=0
                if t >= args.switch_off_step:
                    act = np.zeros_like(act, dtype=np.float32)

                obs, rew, terminated, truncated, info = envs.step(act)

                # if any env truly terminates, stop early
                if np.any(terminated):
                    break

            # read flux trace from result files in EP_{ep_num}
            ep_dir = os.path.join(sync_root, "CFD_n999", f"EP_{ep_num}")
            times, fluxes, kes = load_flux_ke_trace(ep_dir, args.eval_steps)

            arr_f = np.asarray(fluxes, dtype=np.float32)
            if arr_f.size == 0:
                print(f"[warn] no flux samples found in {ep_dir}")
                continue

            print(f"[eval joint-ep {j}] samples={len(arr_f)} mean_flux={arr_f.mean():.6f} std_flux={arr_f.std():.6f}")

            # control-on/off means (exclude actuation 0 snapshot if present)
            # fluxes list corresponds to actuation indices in file names (starts at 0 if exists)
            # Use actuation 1..switch for control-on; switch+1..end for control-off
            if len(arr_f) > args.switch_off_step + 1:
                on_seg = arr_f[1: args.switch_off_step + 1]   # actuation 1..switch
                off_seg = arr_f[args.switch_off_step + 1:]   # after switch
                if on_seg.size > 0 and off_seg.size > 0:
                    print(f"    control-on  mean={on_seg.mean():.6f} | control-off mean={off_seg.mean():.6f}")

            # write to CSV
            for a, (tm, fl, ke) in enumerate(zip(times, fluxes, kes)):
                # control_on: 1 for actuation 1..switch_off_step, else 0
                control_on = 1 if (1 <= a <= args.switch_off_step) else 0
                fcsv.write(f"{j},{a},{tm:.6f},{fl:.8f},{ke:.8f},{control_on}\n")

    print("[eval] saved:", out_csv)
    envs.close()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
