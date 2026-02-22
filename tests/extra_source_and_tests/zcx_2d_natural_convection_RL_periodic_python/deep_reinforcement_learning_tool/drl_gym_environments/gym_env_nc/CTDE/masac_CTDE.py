# masac_CTDE.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: List[int], out_dim: int, activation=nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    """
    Independent SAC actor:
    - input: local obs (B, obs_dim)
    - output: action in (-1,1) with tanh-squashed Gaussian
    """
    def __init__(
        self,
        obs_dim: int,
        hidden: List[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.net = mlp(obs_dim, hidden, out_dim=hidden[-1] if hidden else obs_dim)
        feat_dim = hidden[-1] if hidden else obs_dim
        self.mu = nn.Linear(feat_dim, 1)
        self.log_std = nn.Linear(feat_dim, 1)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (a, logp, pre_tanh_u)
        a: (B,1) in (-1,1)
        logp: (B,1)
        """
        x = self.net(obs)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)

        # log_prob with tanh correction
        # gaussian log prob
        logp_u = (-0.5 * (((u - mu) / (std + 1e-8)) ** 2) - torch.log(std + 1e-8) - 0.5 * np.log(2 * np.pi))
        logp_u = logp_u.sum(dim=-1, keepdim=True)

        # tanh correction: log|det(d tanh(u)/du)| = sum log(1 - tanh(u)^2)
        logp = logp_u - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return a, logp, u

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        x = self.net(obs)
        mu = self.mu(x)
        if deterministic:
            a = torch.tanh(mu)
            return a.cpu().numpy()
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        return a.cpu().numpy()


class CentralizedCritic(nn.Module):
    """Q(o_joint, a_joint) -> scalar"""
    def __init__(self, joint_obs_dim: int, joint_act_dim: int, hidden: List[int]):
        super().__init__()
        self.q = mlp(joint_obs_dim + joint_act_dim, hidden, out_dim=1)

    def forward(self, joint_obs: torch.Tensor, joint_act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([joint_obs, joint_act], dim=-1)
        return self.q(x)


@dataclass
class MASACConfig:
    n_agents: int
    obs_dim: int              # local obs dim per agent
    act_dim: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 5e-4
    auto_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy: Optional[float] = None  # if None -> -(n_agents*act_dim)


class MASACCTDE:
    """
    Cooperative CTDE MASAC:
    - 10 independent actors: pi_i(a_i|o_i)
    - 1 centralized twin critic: Q(o_1..o_n, a_1..a_n)
    - shared reward
    """

    def __init__(self, cfg: MASACConfig, hidden: List[int], device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.actors = nn.ModuleList([
            TanhGaussianActor(cfg.obs_dim, hidden=hidden).to(self.device)
            for _ in range(cfg.n_agents)
        ])

        joint_obs_dim = cfg.n_agents * cfg.obs_dim
        joint_act_dim = cfg.n_agents * cfg.act_dim

        self.q1 = CentralizedCritic(joint_obs_dim, joint_act_dim, hidden=hidden).to(self.device)
        self.q2 = CentralizedCritic(joint_obs_dim, joint_act_dim, hidden=hidden).to(self.device)
        self.q1_t = CentralizedCritic(joint_obs_dim, joint_act_dim, hidden=hidden).to(self.device)
        self.q2_t = CentralizedCritic(joint_obs_dim, joint_act_dim, hidden=hidden).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        # optimizers (independent actors => separate optimizers is cleanest)
        self.actor_opt = [torch.optim.Adam(a.parameters(), lr=cfg.actor_lr) for a in self.actors]
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        # alpha (shared joint entropy coefficient)
        if cfg.auto_alpha:
            if cfg.target_entropy is None:
                cfg.target_entropy = -float(cfg.n_agents * cfg.act_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(cfg.init_alpha), device=self.device)
            self.alpha_opt = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _soft_update(self, net_t: nn.Module, net: nn.Module) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p_t, p in zip(net_t.parameters(), net.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    @torch.no_grad()
    def select_action(self, obs_local: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        obs_local: (B, n_agents, obs_dim) or (n_agents, obs_dim)
        return: (B, n_agents) or (n_agents,)
        """
        x = torch.as_tensor(obs_local, dtype=torch.float32, device=self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1,n,obs_dim)
        B, n, d = x.shape
        assert n == self.cfg.n_agents and d == self.cfg.obs_dim

        actions = []
        for i in range(n):
            a_i = self.actors[i].act(x[:, i, :], deterministic=deterministic)  # (B,1)
            actions.append(a_i)
        a = np.concatenate(actions, axis=1).astype(np.float32)  # (B,n)
        if obs_local.ndim == 2:
            return a[0]
        return a

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        batch keys:
          obs:      (B,n,obs_dim)
          act:      (B,n)
          rew:      (B,1)
          next_obs: (B,n,obs_dim)
          done:     (B,1)  float32 0/1
        """
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        B, n, d = obs.shape
        joint_obs = obs.reshape(B, n * d)
        joint_act = act.reshape(B, n * self.cfg.act_dim)
        joint_nxt = nxt.reshape(B, n * d)

        # -------- critic target --------
        with torch.no_grad():
            next_actions = []
            next_logps = []
            for i in range(n):
                a_i, logp_i, _ = self.actors[i](nxt[:, i, :])  # (B,1)
                next_actions.append(a_i)
                next_logps.append(logp_i)
            a_nxt = torch.cat(next_actions, dim=1)            # (B,n)
            logp_sum = torch.sum(torch.cat(next_logps, dim=1), dim=1, keepdim=True)  # (B,1)

            q1_t = self.q1_t(joint_nxt, a_nxt)
            q2_t = self.q2_t(joint_nxt, a_nxt)
            q_t = torch.min(q1_t, q2_t) - self.alpha.detach() * logp_sum

            y = rew + self.cfg.gamma * (1.0 - done) * q_t

        # -------- critic update --------
        q1 = self.q1(joint_obs, joint_act)
        q2 = self.q2(joint_obs, joint_act)
        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        # -------- actor update (independent params) --------
        actor_losses = []
        entropies = []

        # pre-sample other agents' actions with no_grad to reduce compute variance per-actor
        with torch.no_grad():
            other_actions_ng = []
            for j in range(n):
                a_j, _, _ = self.actors[j](obs[:, j, :])
                other_actions_ng.append(a_j)  # list of (B,1)

        for i in range(n):
            # a_i with grad
            a_i, logp_i, _ = self.actors[i](obs[:, i, :])  # (B,1)

            # build joint action where only i has gradient
            acts = []
            for j in range(n):
                if j == i:
                    acts.append(a_i)
                else:
                    acts.append(other_actions_ng[j].detach())
            a_joint = torch.cat(acts, dim=1)  # (B,n)

            q_pi = torch.min(self.q1(joint_obs, a_joint), self.q2(joint_obs, a_joint))
            loss_i = (self.alpha * logp_i - q_pi).mean()

            self.actor_opt[i].zero_grad(set_to_none=True)
            loss_i.backward()
            self.actor_opt[i].step()

            actor_losses.append(loss_i.item())
            entropies.append((-logp_i).mean().item())

        # -------- alpha update (joint entropy) --------
        alpha_loss_val = 0.0
        alpha_val = float(self.alpha.detach().cpu().item())

        if self.cfg.auto_alpha:
            with torch.no_grad():
                # use current policy entropy estimate (joint)
                logps = []
                for i in range(n):
                    _, logp_i, _ = self.actors[i](obs[:, i, :])
                    logps.append(logp_i)
                logp_sum = torch.sum(torch.cat(logps, dim=1), dim=1, keepdim=True)  # (B,1)

            # want: E[-log pi] ~= target_entropy  => E[log pi + target_entropy] ~= 0
            alpha_loss = -(self.log_alpha * (logp_sum + float(self.cfg.target_entropy))).mean()

            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            alpha_loss_val = float(alpha_loss.detach().cpu().item())
            alpha_val = float(self.alpha.detach().cpu().item())

        # -------- soft update targets --------
        self._soft_update(self.q1_t, self.q1)
        self._soft_update(self.q2_t, self.q2)

        return {
            "q1_loss": float(q1_loss.detach().cpu().item()),
            "q2_loss": float(q2_loss.detach().cpu().item()),
            "actor_loss_mean": float(np.mean(actor_losses)),
            "entropy_mean": float(np.mean(entropies)),
            "alpha": alpha_val,
            "alpha_loss": alpha_loss_val,
        }


class JointReplayBuffer:
    """Simple numpy ring buffer for joint transitions."""
    def __init__(self, capacity: int, n_agents: int, obs_dim: int):
        self.capacity = int(capacity)
        self.n = int(n_agents)
        self.d = int(obs_dim)

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, n_agents), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.nxt = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, nxt, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i, 0] = float(rew)
        self.nxt[i] = nxt
        self.done[i, 0] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.nxt[idx],
            "done": self.done[idx],
        }
