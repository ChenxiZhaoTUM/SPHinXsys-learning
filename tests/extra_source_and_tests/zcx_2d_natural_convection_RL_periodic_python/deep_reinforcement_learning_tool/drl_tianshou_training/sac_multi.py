#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAC training for Vignon-style invariant multi-agent control.

- One pseudo-env per segment (inv_id = 0..n_seg-1)
- n_seg pseudo-envs share ONE CFD simulation via Wrapper (disk sync)
- ONE shared SAC policy (parameter sharing)
"""

import os
import sys
import argparse
import datetime
from functools import partial
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger

import shutil
import stat


# -------------------------------------------------
# Env factory (MUST be top-level for Windows spawn)
# -------------------------------------------------
def make_env(
        *,
        sph_lib_path: str,
        training_root: str,
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
        nu_target: float,
        reward_scale: float,
        baseline_temp: float,
):
    import sys
    import os

    # ---- SPHinXsys binding ----
    if sph_lib_path not in sys.path:
        sys.path.insert(0, sph_lib_path)

    import zcx_2d_natural_convection_RL_periodic_python as test_2d

    # ---- make sure env code is importable ----
    this_dir = os.path.dirname(os.path.abspath(__file__))
    drl_dir = os.path.dirname(this_dir)
    gym_root = os.path.join(drl_dir, "drl_gym_environments")

    for p in (this_dir, gym_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    # ---- import env (NO monkey patch) ----
    from gym_env_nc.marl.NCPseudoEnv import NCPseudoEnv

    return NCPseudoEnv(
        solver_factory=test_2d.natural_convection_from_sph_cpp,
        training_root=training_root,
        restart_dir=restart_dir,
        parallel_envs=parallel_envs,
        inv_id=inv_id,
        n_seg=n_seg,
        n_rows=n_rows,
        n_cols=n_cols,
        warmup_time=warmup_time,
        delta_time=delta_time,
        avg_len=avg_len,
        max_steps_per_episode=max_steps,
        beta=beta,
        nu_target=nu_target,
        reward_scale=reward_scale,
        baseline_temp=baseline_temp,
    )


def _rmtree_onerror(func, path, exc_info):
    # Windows: remove read-only / locked attributes best-effort
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def rebuild_training_root(training_root: str) -> str:
    """
    ALWAYS delete training_root entirely, then recreate:
      - training_root/
      - training_root/restart/
      - training_root/input, output, reload
    Return absolute training_root.
    """
    training_root = os.path.abspath(training_root)

    # SAFETY GUARD: only allow clearing a dir named training_results_pseudo
    base = os.path.basename(training_root.rstrip("\\/"))
    if base != "training_results_pseudo":
        raise RuntimeError(
            f"Refuse to clear unexpected directory:\n  {training_root}\n"
            f"Expected basename == 'training_results_pseudo'."
        )

    # best-effort retry to avoid transient locks
    if os.path.isdir(training_root):
        for _ in range(5):
            try:
                shutil.rmtree(training_root, onerror=_rmtree_onerror)
                break
            except Exception:
                time.sleep(0.2)
        if os.path.isdir(training_root):
            shutil.rmtree(training_root, onerror=_rmtree_onerror)

    # recreate skeleton
    os.makedirs(training_root, exist_ok=True)
    os.makedirs(os.path.join(training_root, "restart"), exist_ok=True)
    for name in ("input", "output", "reload"):
        os.makedirs(os.path.join(training_root, name), exist_ok=True)

    return training_root


# -------------------------------------------------
# Args
# -------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--n-seg", type=int, default=10)
    p.add_argument("--n-rows", type=int, default=8)
    p.add_argument("--n-cols", type=int, default=30)
    p.add_argument("--groups", type=int, default=1)

    p.add_argument("--warmup-time", type=float, default=400.0)
    p.add_argument("--delta-time", type=float, default=2.0)
    p.add_argument("--avg-len", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=200)

    p.add_argument("--beta", type=float, default=0.0015)
    p.add_argument("--nu-target", type=float, default=22.5)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--baseline-temp", type=float, default=2.0)

    p.add_argument("--training-root", type=str, default=None)
    p.add_argument("--restart-dir", type=str, default=None)
    p.add_argument(
        "--sph-lib-path",
        type=str,
        default=r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_natural_convection_RL_periodic_python\lib\Release",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--auto-alpha", default=True, action="store_true")
    p.add_argument("--alpha-lr", type=float, default=5e-4)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--step-per-epoch", type=int, default=2048)
    p.add_argument("--step-per-collect", type=int, default=512)
    p.add_argument("--update-per-step", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--logdir", type=str, default="log")

    return p.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    # ---- paths ----
    if args.training_root is None:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "../training_process"))
        args.training_root = os.path.join(proj_root, "training_results_pseudo")
    args.training_root = os.path.abspath(args.training_root)

    # >>> 每次运行都全删并重建（关键点：必须在 spawn 子进程前做） <<<
    args.training_root = rebuild_training_root(args.training_root)

    # restart_dir 强制设为 training_root/restart（你也可以保留 argparse，但这里按“全删重建”固定）
    args.restart_dir = os.path.join(args.training_root, "restart")
    os.makedirs(args.restart_dir, exist_ok=True)

    print(f"[main] training_root rebuilt: {args.training_root}")
    print(f"[main] restart_dir: {args.restart_dir}")

    # ---- build envs ----
    env_fns = []
    for g in range(args.groups):
        for inv_id in range(args.n_seg):
            env_fns.append(
                partial(
                    make_env,
                    sph_lib_path=args.sph_lib_path,
                    training_root=args.training_root,
                    restart_dir=args.restart_dir,
                    parallel_envs=g,
                    inv_id=inv_id,
                    n_seg=args.n_seg,
                    n_rows=args.n_rows,
                    n_cols=args.n_cols,
                    warmup_time=args.warmup_time,
                    delta_time=args.delta_time,
                    avg_len=args.avg_len,
                    max_steps=args.max_steps,
                    beta=args.beta,
                    nu_target=args.nu_target,
                    reward_scale=args.reward_scale,
                    baseline_temp=args.baseline_temp,
                )
            )

    train_envs = SubprocVectorEnv(env_fns)

    # ---- spaces (SAFE) ----
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space

    # tianshou VectorEnv 可能返回 list[Space]
    if isinstance(obs_space, (list, tuple)):
        obs_space = obs_space[0]
    if isinstance(act_space, (list, tuple)):
        act_space = act_space[0]

    obs_shape = obs_space.shape
    act_shape = act_space.shape
    max_action = float(act_space.high[0])

    # ---- networks ----
    net_a = Net(obs_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(net_a, act_shape, max_action=max_action, device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(obs_shape, act_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    net_c2 = Net(obs_shape, act_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(act_shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        action_space=act_space,
    )

    # ---- collector ----
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    collector = Collector(policy, train_envs, buffer, exploration_noise=False)

    # ---- logging ----
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, f"sac_seg{args.n_seg}", now)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # ---- train ----
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=collector,
        test_collector=None,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=1,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        logger=logger,
    ).run()

    print(result)


if __name__ == "__main__":
    main()
