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
import shutil
import stat

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger


# -------------------------------------------------
# Env factory (MUST be top-level for Windows spawn)
# -------------------------------------------------
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
    nu_target: float,
    reward_scale: float,
    baseline_temp: float,
):
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
        max_steps_per_episode=max_steps,
        beta=beta,
        nu_target=nu_target,
        reward_scale=reward_scale,
        baseline_temp=baseline_temp,
    )


def _rmtree_onerror(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def rebuild_run_dir(run_dir: str, run_name: str, groups: int, test_gid: int = 999):
    """
      run_dir/
        sync/CFD_n{gid}/input|output|reload|restart
        logs/
    """
    run_dir = os.path.abspath(run_dir)

    base = os.path.basename(run_dir.rstrip("\\/"))
    if base != run_name:
        raise RuntimeError(
            f"Refuse to clear unexpected directory:\n  {run_dir}\n"
            f"Expected basename == '{run_name}'."
        )

    if os.path.isdir(run_dir):
        for _ in range(5):
            try:
                shutil.rmtree(run_dir, onerror=_rmtree_onerror)
                break
            except Exception:
                time.sleep(0.2)
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir, onerror=_rmtree_onerror)

    # root
    os.makedirs(run_dir, exist_ok=True)

    # sync/logs
    sync_root = os.path.join(run_dir, "sync")
    logs_root = os.path.join(run_dir, "logs")
    os.makedirs(sync_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    gids = list(range(groups)) + [test_gid]
    for gid in gids:
        gdir = os.path.join(sync_root, f"CFD_n{gid}")
        for sub in ("input", "output", "reload", "restart"):
            os.makedirs(os.path.join(gdir, sub), exist_ok=True)

    return run_dir, sync_root, logs_root


# -------------------------------------------------
# Args
# -------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--task", type=str, default="NC-v0-marl")

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

    p.add_argument("--run-name", type=str, default="training_results_pseudo")
    p.add_argument("--run-root", type=str, default=None)
    p.add_argument(
        "--sph-lib-path",
        type=str,
        default=r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_natural_convection_RL_periodic_python\lib\Release",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # ---- align with single-agent semantics ----
    p.add_argument("--episodes-per-epoch", type=int, default=20)  # 20 episodes per epoch
    p.add_argument("--test-num", type=int, default=1)             # joint episodes (we convert internally)
    p.add_argument("--buffer-size", type=int, default=100_000)

    p.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)

    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--auto-alpha", default=False, action="store_true")
    p.add_argument("--alpha-lr", type=float, default=5e-4)

    p.add_argument("--epoch", type=int, default=30)
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

    # ---- fixed run_dir under training_process/ ----
    if args.run_root is None:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "training_process"))
        args.run_root = proj_root

    run_dir = os.path.join(os.path.abspath(args.run_root), args.run_name)

    run_dir, sync_root, logs_root = rebuild_run_dir(
        run_dir=run_dir,
        run_name=args.run_name,
        groups=args.groups,
        test_gid=999,
    )

    print(f"[main] run_dir rebuilt: {run_dir}")
    print(f"[main] sync_root: {sync_root}")
    print(f"[main] logs_root: {logs_root}")

    args.training_root = run_dir
    args.sync_root = sync_root
    args.logs_root = logs_root

    # ---- build TRAIN envs: groups * n_seg ----
    train_env_fns = []
    for g in range(args.groups):
        restart_dir_g = os.path.join(args.sync_root, f"CFD_n{g}", "restart")
        for inv_id in range(args.n_seg):
            train_env_fns.append(
                partial(
                    make_env,
                    sph_lib_path=args.sph_lib_path,
                    training_root=args.training_root,
                    sync_root=args.sync_root,
                    restart_dir=restart_dir_g,
                    parallel_envs=g,          # train groups: 0..groups-1
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
    train_envs = SubprocVectorEnv(train_env_fns)

    # ---- build TEST envs: one isolated group parallel_envs=999 ----
    restart_dir_t = os.path.join(args.sync_root, "CFD_n999", "restart")
    test_env_fns = []
    for inv_id in range(args.n_seg):
        test_env_fns.append(
            partial(
                make_env,
                sph_lib_path=args.sph_lib_path,
                training_root=args.training_root,
                sync_root=args.sync_root,
                restart_dir=restart_dir_t,
                parallel_envs=999,          # isolated test CFD_n999
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
    test_envs = SubprocVectorEnv(test_env_fns)

    # ---- spaces (SAFE) ----
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    if isinstance(obs_space, (list, tuple)):
        obs_space = obs_space[0]
    if isinstance(act_space, (list, tuple)):
        act_space = act_space[0]

    obs_shape = obs_space.shape
    act_shape = act_space.shape
    max_action = float(act_space.high[0])

    # align with your single script behavior
    args.auto_alpha = True

    # ---- networks (align single: 2x512 + tanh + Adam) ----
    net_a = Net(obs_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, device=args.device)
    actor = ActorProb(
        net_a, act_shape, device=args.device,
        max_action=max_action, conditioned_sigma=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(obs_shape, act_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh,
                 concat=True, device=args.device)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    net_c2 = Net(obs_shape, act_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh,
                 concat=True, device=args.device)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    alpha = float(args.alpha)
    if args.auto_alpha:
        # use act_shape (NOT train_envs.action_space.shape which can be list)
        prod = np.prod(act_shape)
        target_entropy = -float(prod) if np.isscalar(prod) else -float(prod.item())
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        estimation_step=1,
        action_space=act_space,
    )

    # ---- replay buffer + collectors ----
    num_envs = len(train_envs)          # = groups * n_seg
    steps_per_episode = int(args.max_steps)

    buffer = VectorReplayBuffer(args.buffer_size, buffer_num=num_envs)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=False)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # ---- random prefill (align single: 15 episodes, here convert to env-episodes) ----
    # 15 joint episodes ~= 15 * num_envs env-episodes
    train_collector.collect(n_episode=15 * num_envs, random=True)

    # ---- logging (align single style) ----
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_path = os.path.join(args.logs_root, os.path.join(args.task, "sac", str(args.seed), now))
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy_):
        torch.save(policy_.state_dict(), os.path.join(log_path, "policy.pth"))

    # ---- align "20 episodes per epoch" with vector env semantics ----
    # one joint episode -> steps_per_episode * num_envs transitions
    step_per_epoch = steps_per_episode * int(args.episodes_per_epoch) * num_envs
    step_per_collect = steps_per_episode * 2 * num_envs
    args.update_per_step = 1.0 / args.n_seg

    # test_num is joint episodes; Collector counts env-episodes, so multiply by n_seg (test_envs length)
    episode_per_test = int(args.test_num) * len(test_envs)  # 1个epoch测试一次

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        episode_per_test=episode_per_test,
        max_epoch=args.epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    print(result)


if __name__ == "__main__":
    main()
