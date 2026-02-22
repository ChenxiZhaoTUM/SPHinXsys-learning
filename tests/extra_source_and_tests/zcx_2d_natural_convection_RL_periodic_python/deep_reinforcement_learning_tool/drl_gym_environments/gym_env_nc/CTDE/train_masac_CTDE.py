#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import stat
import shutil
import argparse
import datetime
from typing import List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from NCJointEnv import NCJointEnv
from masac_CTDE import MASACCTDE, MASACConfig, JointReplayBuffer


# -----------------------------
# Utilities: filesystem
# -----------------------------
def _rmtree_onerror(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def rebuild_run_dir(run_dir: str, expected_basename: str) -> str:
    """Clear & recreate run_dir safely (basename guard)."""
    run_dir = os.path.abspath(run_dir)
    base = os.path.basename(run_dir.rstrip("\\/"))
    if base != expected_basename:
        raise RuntimeError(
            f"Refuse to clear unexpected directory:\n  {run_dir}\n"
            f"Expected basename == '{expected_basename}'."
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

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def find_training_project(start_dir: str, max_up: int = 10) -> str:
    """Infer a default run_root by walking up to find training_process/."""
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        tp = os.path.join(cur, "training_process")
        if os.path.isdir(tp):
            return tp
        if os.path.basename(cur).lower() == "zcx_2d_natural_convection_rl_periodic_python":
            return os.path.join(cur, "training_process")
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return os.path.join(os.path.abspath(start_dir), "training_process")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Env factory
# -----------------------------
def make_env(
        *,
        sph_lib_path: str,
        training_root: str,
        group_id: int,
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
) -> NCJointEnv:
    if sph_lib_path and sph_lib_path not in sys.path:
        sys.path.insert(0, sph_lib_path)

    import zcx_2d_natural_convection_RL_periodic_python as test_2d
    solver_factory = test_2d.natural_convection_from_sph_cpp

    return NCJointEnv(
        solver_factory=solver_factory,
        group_id=int(group_id),
        training_root=training_root,
        n_seg=int(n_seg),
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        avg_len=int(avg_len),
        warmup_time=float(warmup_time),
        delta_time=float(delta_time),
        max_steps_per_episode=int(max_steps),
        beta=float(beta),
        nu_target=float(nu_target),
        reward_scale=float(reward_scale),
        baseline_T=float(baseline_temp),
    )


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(env: NCJointEnv, agent: MASACCTDE, episodes: int = 1) -> float:
    # set eval mode (规范做法；即使无 dropout/bn 也建议)
    for a in agent.actors:
        a.eval()
    agent.q1.eval()
    agent.q2.eval()

    rets = []
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            act = agent.select_action(obs, deterministic=True)  # (n_seg,)
            obs, r, terminated, truncated, _ = env.step(act)
            done = bool(terminated or truncated)
            ep_ret += float(r)
        rets.append(ep_ret)

    # back to train mode
    for a in agent.actors:
        a.train()
    agent.q1.train()
    agent.q2.train()

    return float(np.mean(rets))


# -----------------------------
# Args
# -----------------------------
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--task", type=str, default="NC-CTDE-MASAC")
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

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    p.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--alpha-lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--auto-alpha", action="store_true", default=True)
    p.add_argument("--init-alpha", type=float, default=0.25)

    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--start-steps", type=int, default=2000)  # ?
    p.add_argument("--updates-per-step", type=int, default=1)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--episodes-per-epoch", type=int, default=10)  # per group
    p.add_argument("--test-episodes", type=int, default=1)

    p.add_argument(
        "--sph-lib-path",
        type=str,
        default=r"D:\SPHinXsys_build\tests\test_python_interface\zcx_2d_natural_convection_RL_periodic_python\lib\Release",
    )

    p.add_argument("--run-root", type=str, default=None)
    p.add_argument("--run-name", type=str, default="training_results_masac_CTDE")

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    torch.set_num_threads(1)

    # ---- decide run_root ----
    if args.run_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.run_root = find_training_project(script_dir)

    run_root = os.path.abspath(args.run_root)
    os.makedirs(run_root, exist_ok=True)

    run_dir = os.path.join(run_root, args.run_name)
    run_dir = rebuild_run_dir(run_dir, expected_basename=args.run_name)
    print(f"[main] run_dir rebuilt: {run_dir}")

    # ---- build train envs ----
    train_envs: List[NCJointEnv] = []
    for gid in range(args.groups):
        env = make_env(
            sph_lib_path=args.sph_lib_path,
            training_root=run_dir,
            group_id=gid,
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
        train_envs.append(env)

    # ---- test env (isolated group 999) ----
    test_env = make_env(
        sph_lib_path=args.sph_lib_path,
        training_root=run_dir,
        group_id=999,
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

    # ---- infer local obs_dim ----
    obs0, _ = train_envs[0].reset()
    if obs0.shape[0] != args.n_seg:
        raise RuntimeError(f"Env obs should be (n_seg, local_dim). Got {obs0.shape}")
    obs_dim = int(obs0.shape[1])

    # ---- MASAC CTDE ----
    cfg = MASACConfig(
        n_agents=args.n_seg,
        obs_dim=obs_dim,
        act_dim=1,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        auto_alpha=bool(args.auto_alpha),
        init_alpha=float(args.init_alpha),
        target_entropy=-float(args.n_seg),
    )
    agent = MASACCTDE(cfg=cfg, hidden=list(args.hidden_sizes), device=args.device)

    # ---- replay buffer ----
    buf = JointReplayBuffer(args.buffer_size, n_agents=args.n_seg, obs_dim=obs_dim)

    # ---- logging dirs ----
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tb_dir = os.path.join(run_dir, "tb", args.task, str(args.seed), now)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    writer.add_text("args", str(vars(args)))
    print(f"[main] TensorBoard: {tb_dir}")

    best_test = -1e18

    # ---- best model save dir (actors + critics only) ----
    best_dir = os.path.join(run_dir, "best_models", args.task, str(args.seed), now)
    os.makedirs(best_dir, exist_ok=True)
    print(f"[main] best_models: {best_dir}")

    def _atomic_torch_save(obj, path: str):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_best_models(best_value: float, epoch: int):
        # policy (10 actors)
        actor_payload = {
            "actors": [a.state_dict() for a in agent.actors],
            "n_agents": args.n_seg,
            "obs_dim": obs_dim,
            "hidden_sizes": list(args.hidden_sizes),
            "best_test_return": float(best_value),
            "best_epoch": int(epoch),
        }
        _atomic_torch_save(actor_payload, os.path.join(best_dir, "best_actor.pth"))

        # critics (q1,q2)
        critic_payload = {
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "n_agents": args.n_seg,
            "obs_dim": obs_dim,
            "hidden_sizes": list(args.hidden_sizes),
            "gamma": args.gamma,
            "tau": args.tau,
            "best_test_return": float(best_value),
            "best_epoch": int(epoch),
        }
        _atomic_torch_save(critic_payload, os.path.join(best_dir, "best_critic.pth"))

        print(f"[save] best updated @ epoch {epoch}: test_ret={best_value:.6f}")

    try:
        # -----------------------------
        # Prefill (NOTE: env 会写 reward 曲线；如不想污染，建议 start_steps=0 或改 env 加 enable_logging)
        # -----------------------------
        print("[main] Prefill replay with random actions ...")
        target_prefill = min(args.start_steps, args.buffer_size)
        while buf.size < target_prefill:
            for env in train_envs:
                obs, _ = env.reset()
                done = False
                while not done and buf.size < target_prefill:
                    act = np.random.uniform(-1.0, 1.0, size=(args.n_seg,)).astype(np.float32)
                    nxt, r, terminated, truncated, _ = env.step(act)
                    done = bool(terminated or truncated)
                    buf.add(obs, act, r, nxt, done)
                    obs = nxt
        print(f"[main] Prefill done. buffer size={buf.size}")

        # -----------------------------
        # Training loop
        # -----------------------------
        for epoch in range(1, args.epochs + 1):
            losses_q = []
            losses_actor = []
            alphas = []
            train_returns = []

            for _ in range(int(args.episodes_per_epoch)):
                for env in train_envs:
                    obs, _ = env.reset()
                    done = False
                    ep_ret = 0.0

                    while not done:
                        act = agent.select_action(obs, deterministic=False)
                        nxt, r, terminated, truncated, _ = env.step(act)
                        done = bool(terminated or truncated)

                        buf.add(obs, act, r, nxt, done)
                        obs = nxt
                        ep_ret += float(r)

                        for _u in range(int(args.updates_per_step)):
                            batch = buf.sample(args.batch_size)
                            stats = agent.update(batch)
                            losses_q.append(0.5 * (stats["q1_loss"] + stats["q2_loss"]))
                            losses_actor.append(stats["actor_loss_mean"])
                            alphas.append(stats["alpha"])

                    train_returns.append(ep_ret)

            test_ret = evaluate(test_env, agent, episodes=args.test_episodes)

            if losses_q:
                writer.add_scalar("train/q_loss", float(np.mean(losses_q)), epoch)
                writer.add_scalar("train/actor_loss", float(np.mean(losses_actor)), epoch)
                writer.add_scalar("train/alpha", float(np.mean(alphas)), epoch)
            if train_returns:
                writer.add_scalar("train/return_mean", float(np.mean(train_returns)), epoch)
            writer.add_scalar("test/return", float(test_ret), epoch)

            if test_ret > best_test:
                best_test = test_ret
                save_best_models(best_test, epoch)
            writer.add_scalar("test/best_return", float(best_test), epoch)

            print(
                f"[epoch {epoch:03d}] train_ret={np.mean(train_returns):.6f}  "
                f"test_ret={test_ret:.6f}  best={best_test:.6f}  "
                f"alpha={np.mean(alphas) if alphas else 0.0:.4f}"
            )

    finally:
        writer.flush()
        writer.close()
        for env in train_envs:
            try:
                env.close()
            except Exception:
                pass
        try:
            test_env.close()
        except Exception:
            pass

    print(f"[main] Done. best_test={best_test:.6f}")
    print(f"[main] Outputs in: {run_dir}")
    print(f"[main] TensorBoard in: {tb_dir}")
    print(f"[main] Best models in: {best_dir}")


if __name__ == "__main__":
    main()
