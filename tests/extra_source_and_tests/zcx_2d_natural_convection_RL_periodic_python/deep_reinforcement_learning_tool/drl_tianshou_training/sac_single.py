#!/usr/bin/env python3
import gymnasium as gym
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../bin/drl/drl_tianshou_training
DRL_DIR = os.path.dirname(CURRENT_DIR)  # .../bin/drl
GYM_ROOT = os.path.join(DRL_DIR, "drl_gym_environments")  # .../bin/drl/drl_gym_environments
# put drl_gym_environments into sys.path
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

import gym_env_nc

# for test
# env = gym.make("NC-v0", n_seg=4)
# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)
# obs, _ = env.reset()
# print("reset obs shape:", obs.shape, "first few:", obs[:5])


import argparse
import datetime
import os
import pprint

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.env import SubprocVectorEnv
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

import shutil
import stat
import time


def _rmtree_onerror(func, path, exc_info):
    # Windows: remove read-only attributes best-effort
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
    if base != "training_results_single":
        raise RuntimeError(
            f"Refuse to clear unexpected directory:\n  {training_root}\n"
            f"Expected basename == 'training_results_single'."
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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-seg", type=int, default=10,
                        help="number of heater segments along the bottom wall")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)

    parser.add_argument("--task", type=str, default="NC-v0")  # Environment ID
    parser.add_argument("--episodes-per-epoch", type=int, default=20)
    parser.add_argument("--buffer-size", type=int, default=10e4)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=5e-4)
    parser.add_argument("--start-timesteps", type=int, default=0)  # Replace the step number with episode pre-sampling
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    # parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument("--watch", default=False, action="store_true", help="Watch the play of pre-trained policy only")
    return parser.parse_args()


def training_sac(args=get_args()):
    # ---- CLEAR + REBUILD training_results_single BEFORE env spawn ----
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "../training_process"))
    training_root = os.path.join(proj_root, "training_results_single")
    training_root = rebuild_training_root(training_root)
    print(f"[single] training_root rebuilt: {training_root}")
    print(f"[single] restart_dir: {os.path.join(training_root, 'restart')}")

    """Main function for setting up and training a SAC agent in the NC environment."""
    restart_dir = os.path.join(training_root, "restart")
    train_env = gym.make("NC-v0", parallel_envs=0, n_seg=args.n_seg,
                         training_root=training_root, restart_dir=restart_dir)
    test_env = gym.make("NC-v0", parallel_envs=999, n_seg=args.n_seg,
                        training_root=training_root, restart_dir=restart_dir)

    # Create vectorized environments for parallel training
    # envs = SubprocVectorEnv([
    #     lambda i=i: gym.make(args.task, parallel_envs=i)  # Create environments with different parallel_envs IDs
    #     for i in range(args.training_num)])

    # Retrieve environment observation and action space details
    args.state_shape = train_env.observation_space.shape or train_env.observation_space.n
    args.action_shape = train_env.action_space.shape or train_env.action_space.n
    args.max_action = train_env.action_space.high[0]
    args.auto_alpha = True

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define the actor and critic neural networks for SAC
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, device=args.device)
    actor = ActorProb(net_a, args.action_shape, device=args.device, max_action=args.max_action,
                      conditioned_sigma=True).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(args.state_shape, args.action_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, concat=True, device=args.device)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    net_c2 = Net(args.state_shape, args.action_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, concat=True, device=args.device)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    alpha = float(args.alpha)
    # Optionally, use automatic tuning of the temperature parameter (alpha)
    if args.auto_alpha:
        prod = np.prod(train_env.action_space.shape)
        target_entropy = -float(prod) if np.isscalar(prod) else -float(prod.item())
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    # Initialize the SAC policy
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
        estimation_step=args.n_step,
        action_space=train_env.action_space,
    )

    # load a previous or trained policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    if args.watch:
        # Watch the play of the trained policy
        policy.eval()
        test_collector = Collector(policy, train_env)
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num)
        print(f"Final reward: {result['rews'].mean()}, length: {result['lens'].mean()}")
        return

    # Setup replay buffer and collector for training
    buffer = VectorReplayBuffer(args.buffer_size, len(train_env)) if args.training_num > 1 else ReplayBuffer(
        args.buffer_size)
    train_collector = Collector(policy, train_env, buffer, exploration_noise=False)
    test_collector = Collector(policy, test_env, exploration_noise=False)
    # train_collector.collect(n_step=args.start_timesteps, random=True)
    train_collector.collect(n_episode=15, random=True)

    # Setup logging
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, os.path.join(args.task, "sac", str(args.seed), now))
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer) if args.logger == "tensorboard" else None  # Only Tensorboard logging is used

    # Save functions for best policy and checkpoints
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    max_steps = getattr(train_env.unwrapped, "max_steps_per_episode", 200)
    steps_per_episode = int(max_steps)
    step_per_collect = steps_per_episode * int(args.episodes_per_epoch)
    step_per_epoch = step_per_collect  # one collect batch per epoch

    # Training loop using OffpolicyTrainer
    if not args.watch:
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            episode_per_test=args.test_num,
            max_epoch=args.epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
        ).run()
        pprint.pprint(result)


if __name__ == "__main__":
    training_sac()
