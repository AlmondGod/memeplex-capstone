"""Latent-Aligned Independent PPO (LA-IPPO) — Method I.

Each agent has:
  - An encoder  f_θ(o) → latent m   (shared backbone)
  - A policy head  g_ψ(m) → action distribution
  - A value head   V_φ(m) → scalar value

Training alternates between:
  1. Standard PPO updates (every step)
  2. Representation distillation (every k steps):
     L_distill = E_{o~D_i}[ || f_θi(o) - stopgrad(f_θj(o)) ||^2 ]

Designed to be API-compatible with AgileRL's MADDPG / IPPO so the same
training-loop structure works for both.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """MLP encoder: observation → latent vector."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dims: list[int] = (64, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyHead(nn.Module):
    """Gaussian policy head: latent → (mean, log_std) → action distribution."""

    LOG_STD_MIN = -2.0
    LOG_STD_MAX = 0.5

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, m: torch.Tensor):
        h = self.fc(m)
        mean = self.mean(h)
        # Clamp log_std to prevent unbounded entropy growth
        clamped_log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = clamped_log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ValueHead(nn.Module):
    """Value function head: latent → scalar."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.net(m).squeeze(-1)


class AgentNetwork(nn.Module):
    """Full per-agent network: encoder + policy + value."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64,
                 hidden_dims: list[int] = (64, 64)):
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim, hidden_dims)
        self.policy = PolicyHead(latent_dim, action_dim)
        self.value = ValueHead(latent_dim)

    def forward(self, obs: torch.Tensor):
        m = self.encoder(obs)
        dist = self.policy(m)
        val = self.value(m)
        return dist, val, m

    def get_latent(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)


# ---------------------------------------------------------------------------
# Observation buffer for distillation
# ---------------------------------------------------------------------------

class ObservationBuffer:
    """Fixed-size FIFO buffer of observations for distillation sampling."""

    def __init__(self, capacity: int = 512):
        self.buffer: deque[np.ndarray] = deque(maxlen=capacity)

    def add(self, obs: np.ndarray):
        """Add observation(s). Accepts (obs_dim,) or (batch, obs_dim)."""
        if obs.ndim == 1:
            self.buffer.append(obs)
        else:
            for o in obs:
                self.buffer.append(o)

    def sample(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return np.stack([self.buffer[i] for i in idxs])

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Rollout storage (on-policy)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one epoch of rollout data for PPO."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE advantages and discounted returns."""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones, dtype=float)
        values = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns


# ---------------------------------------------------------------------------
# LA-IPPO algorithm
# ---------------------------------------------------------------------------

class LatentAlignedIPPO:
    """Latent-Aligned Independent PPO (Method I).

    Compatible with AgileRL-style training loops.  Key interface methods:
      - get_action(obs, infos=None)
      - learn(experiences)  — PPO update from collected rollouts
      - test(env, ...)      — evaluation rollout
      - save_checkpoint / load (class method)
    """

    def __init__(
        self,
        observation_spaces: list[spaces.Space],
        action_spaces: list[spaces.Space],
        agent_ids: list[str],
        # PPO hyperparameters
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        # Architecture
        latent_dim: int = 64,
        hidden_dims: list[int] = (64, 64),
        # Distillation
        distill_interval: int = 10,
        distill_weight: float = 0.1,
        distill_lr: float = 1e-3,
        distill_batch_size: int = 64,
        obs_buffer_size: int = 512,
        # Annealing
        distill_anneal: bool = False,
        distill_anneal_end: float = 0.0,
        distill_anneal_steps: int = 100,
        # LR annealing
        lr_anneal: bool = False,
        lr_anneal_total_steps: int = 200,
        # Misc
        device: str = "cpu",
    ):
        self.agent_ids = list(agent_ids)
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)
        self.distill_interval = distill_interval
        self.distill_weight_init = distill_weight
        self.distill_weight = distill_weight
        self.distill_batch_size = distill_batch_size
        self.obs_buffer_size = obs_buffer_size
        self.lr = lr
        self.distill_lr = distill_lr

        # Annealing config: linearly decay distill_weight over distill_anneal_steps
        # learn steps (counted in distillation events, not PPO updates)
        self.distill_anneal = distill_anneal
        self.distill_anneal_end = distill_anneal_end
        self.distill_anneal_steps = distill_anneal_steps
        self.distill_step_count = 0  # counts how many distill steps have occurred

        # LR annealing: linearly decay lr from initial to 0 over total learn steps
        self.lr_anneal = lr_anneal
        self.lr_anneal_total_steps = lr_anneal_total_steps

        # Per-agent networks, optimizers, rollout buffers, obs buffers
        # NOTE: A single optimizer per agent covers ALL parameters (encoder +
        # policy + value).  Distillation loss back-propagates only through the
        # encoder, so policy/value params receive zero gradient from distillation.
        # Using one optimizer avoids the pathology of two Adam instances
        # maintaining separate momentum/variance states on the same weights.
        self.networks: dict[str, AgentNetwork] = {}
        self.optimizers: dict[str, torch.optim.Adam] = {}
        self.rollouts: dict[str, RolloutBuffer] = {}
        self.obs_buffers: dict[str, ObservationBuffer] = {}

        for idx, aid in enumerate(self.agent_ids):
            obs_dim = observation_spaces[idx].shape[0]
            act_dim = action_spaces[idx].shape[0]

            net = AgentNetwork(obs_dim, act_dim, latent_dim, self.hidden_dims).to(self.device)
            self.networks[aid] = net
            self.optimizers[aid] = torch.optim.Adam(net.parameters(), lr=lr)
            self.rollouts[aid] = RolloutBuffer()
            self.obs_buffers[aid] = ObservationBuffer(obs_buffer_size)

        # Bookkeeping (AgileRL compatibility)
        self.scores: list[float] = []
        self.fitness: list[float] = []
        self.steps: list[int] = [0]
        self.learn_counter = 0
        self._training = True

    # ------------------------------------------------------------------
    # AgileRL-compatible interface
    # ------------------------------------------------------------------

    def set_training_mode(self, mode: bool):
        self._training = mode
        for net in self.networks.values():
            net.train(mode)

    def get_action(
        self,
        obs: dict[str, np.ndarray],
        infos: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return actions for all agents.

        Returns (actions_dict, raw_actions_dict) for interface compatibility
        with MADDPG. For PPO we also internally store log_probs and values
        in the rollout buffer.
        """
        actions = {}
        raw_actions = {}

        for aid in self.agent_ids:
            o = obs[aid]
            o_t = torch.as_tensor(o, dtype=torch.float32, device=self.device)
            if o_t.dim() == 1:
                o_t = o_t.unsqueeze(0)

            net = self.networks[aid]
            with torch.no_grad():
                dist, val, _ = net(o_t)

            if self._training:
                a = dist.sample()
                log_prob = dist.log_prob(a).sum(-1)
            else:
                a = dist.mean
                log_prob = dist.log_prob(a).sum(-1)

            a_np = a.cpu().numpy()
            # Clamp to action space [0, 1] for MPE
            a_clamped = np.clip(a_np, 0.0, 1.0)

            actions[aid] = a_clamped
            raw_actions[aid] = a_np

            # Store for PPO if training
            if self._training:
                # Store per-env-vector items
                self.rollouts[aid]._last_log_prob = log_prob.cpu().numpy()
                self.rollouts[aid]._last_value = val.cpu().numpy()
                # Store obs in distill buffer
                if o.ndim == 1:
                    self.obs_buffers[aid].add(o)
                else:
                    self.obs_buffers[aid].add(o)

        return actions, raw_actions

    def store_transition(
        self,
        obs: dict[str, np.ndarray],
        raw_actions: dict[str, np.ndarray],
        rewards: dict[str, np.ndarray],
        dones: dict[str, np.ndarray],
    ):
        """Store one transition per agent into rollout buffers.

        This should be called by the training loop after get_action() and
        env.step().  We store per-env-vector averages for simplicity with
        vectorised envs.
        """
        for aid in self.agent_ids:
            rb = self.rollouts[aid]
            r = rewards.get(aid, 0.0)
            d = dones.get(aid, False)

            # Handle vectorised (batch) transitions
            if isinstance(r, np.ndarray) and r.ndim > 0:
                for i in range(len(r)):
                    r_i = float(np.nan_to_num(r[i], nan=0.0))
                    d_i = bool(np.nan_to_num(d[i], nan=1.0))
                    o_i = obs[aid][i] if obs[aid].ndim > 1 else obs[aid]
                    a_i = raw_actions[aid][i] if raw_actions[aid].ndim > 1 else raw_actions[aid]
                    lp_i = float(rb._last_log_prob[i]) if rb._last_log_prob.ndim > 0 else float(rb._last_log_prob)
                    v_i = float(rb._last_value[i]) if rb._last_value.ndim > 0 else float(rb._last_value)
                    rb.add(o_i, a_i, lp_i, r_i, d_i, v_i)
            else:
                rb.add(
                    obs[aid],
                    raw_actions[aid],
                    float(rb._last_log_prob),
                    float(np.nan_to_num(r, nan=0.0)),
                    bool(np.nan_to_num(d, nan=1.0)),
                    float(rb._last_value),
                )

    def learn(self, experiences=None) -> dict[str, float]:
        """Run one PPO update on collected rollouts, plus distillation.

        The `experiences` argument exists for API compatibility but is
        ignored — we use the internal rollout buffers instead.
        """
        self.learn_counter += 1
        total_losses = {}

        # Linear LR annealing: lr → 0 over total learn steps
        if self.lr_anneal:
            frac = max(1.0 - self.learn_counter / max(self.lr_anneal_total_steps, 1), 0.0)
            new_lr = self.lr * frac
            for opt in self.optimizers.values():
                for pg in opt.param_groups:
                    pg["lr"] = new_lr

        for aid in self.agent_ids:
            rb = self.rollouts[aid]
            if len(rb.obs) < self.batch_size:
                continue

            net = self.networks[aid]
            opt = self.optimizers[aid]

            # Compute GAE
            with torch.no_grad():
                last_obs = torch.as_tensor(
                    rb.obs[-1], dtype=torch.float32, device=self.device
                )
                if last_obs.dim() == 1:
                    last_obs = last_obs.unsqueeze(0)
                _, last_val, _ = net(last_obs)
                last_val = last_val.mean().item()

            advantages, returns = rb.compute_gae(last_val, self.gamma, self.gae_lambda)

            # Convert to tensors
            obs_t = torch.as_tensor(np.array(rb.obs), dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(np.array(rb.actions), dtype=torch.float32, device=self.device)
            old_lp_t = torch.as_tensor(np.array(rb.log_probs), dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

            # Normalise advantages
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            n = len(rb.obs)
            epoch_loss = 0.0
            for _ in range(self.update_epochs):
                idxs = np.random.permutation(n)
                for start in range(0, n, self.batch_size):
                    end = start + self.batch_size
                    mb = idxs[start:end]

                    dist, val, _ = net(obs_t[mb])
                    new_lp = dist.log_prob(act_t[mb]).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()

                    ratio = (new_lp - old_lp_t[mb]).exp()
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * adv_t[mb]

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(val, ret_t[mb])
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                    opt.step()
                    epoch_loss += loss.item()

            total_losses[aid] = epoch_loss
            rb.clear()

        # Distillation step
        if self.learn_counter % self.distill_interval == 0:
            self._distill_step()

        return total_losses

    def _distill_step(self):
        """Pairwise representation distillation across all agents.

        Uses the main PPO optimizer (single Adam per agent) to avoid the
        pathology of two optimizers maintaining separate momentum states on
        the same encoder parameters.  Only encoder params receive non-zero
        gradients from the distillation loss, so policy/value weights are
        unaffected.
        """
        # Anneal distill weight linearly: init → end over anneal_steps
        self.distill_step_count += 1
        if self.distill_anneal:
            progress = min(self.distill_step_count / max(self.distill_anneal_steps, 1), 1.0)
            self.distill_weight = (
                self.distill_weight_init
                + (self.distill_anneal_end - self.distill_weight_init) * progress
            )

        if self.distill_weight < 1e-8:
            return  # effectively zero — skip computation

        for i, aid_i in enumerate(self.agent_ids):
            for j, aid_j in enumerate(self.agent_ids):
                if i == j:
                    continue
                buf = self.obs_buffers[aid_i]
                if len(buf) < self.distill_batch_size:
                    continue

                obs_np = buf.sample(self.distill_batch_size)
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

                # Agent i encoder output
                latent_i = self.networks[aid_i].get_latent(obs_t)
                # Agent j encoder output (stop gradient)
                with torch.no_grad():
                    latent_j = self.networks[aid_j].get_latent(obs_t)

                loss = F.mse_loss(latent_i, latent_j) * self.distill_weight

                self.optimizers[aid_i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.networks[aid_i].encoder.parameters(), self.max_grad_norm
                )
                self.optimizers[aid_i].step()

    def test(
        self,
        env,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 3,
        sum_scores: bool = True,
    ) -> float:
        """Evaluate the agent on the environment (AgileRL-compatible)."""
        self.set_training_mode(False)
        total_scores = []

        for _ in range(loop):
            obs, info = env.reset()
            score = np.zeros(env.num_envs) if hasattr(env, "num_envs") else np.array([0.0])
            done_all = False
            step_count = 0

            while not done_all:
                actions, _ = self.get_action(obs)
                obs, reward, termination, truncation, info = env.step(actions)
                step_count += 1

                r_arr = np.array(list(reward.values())).transpose()
                r_arr = np.where(np.isnan(r_arr), 0, r_arr)
                score += np.sum(r_arr, axis=-1) if r_arr.ndim > 1 else np.sum(r_arr)

                dones = {}
                for aid in self.agent_ids:
                    t = termination.get(aid, True)
                    tr = truncation.get(aid, False)
                    t = np.where(np.isnan(t), True, t).astype(bool)
                    tr = np.where(np.isnan(tr), False, tr).astype(bool)
                    dones[aid] = t | tr

                if hasattr(env, "num_envs"):
                    done_all = all(
                        all(dones[aid]) for aid in self.agent_ids
                    )
                else:
                    done_all = all(dones[aid] for aid in self.agent_ids)

                if max_steps and step_count >= max_steps:
                    break

            total_scores.extend(score.tolist() if hasattr(score, "tolist") else [score])

        mean_score = float(np.mean(total_scores))
        self.fitness.append(mean_score)
        self.set_training_mode(True)
        return mean_score

    def reset_action_noise(self, indices=None):
        """No-op — PPO uses stochastic sampling, not additive noise."""
        pass

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        state = {
            "agent_ids": self.agent_ids,
            "networks": {aid: net.state_dict() for aid, net in self.networks.items()},
            "optimizers": {aid: opt.state_dict() for aid, opt in self.optimizers.items()},
            "config": {
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_coef": self.clip_coef,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
                "update_epochs": self.update_epochs,
                "batch_size": self.batch_size,
                "latent_dim": self.latent_dim,
                "hidden_dims": self.hidden_dims,
                "distill_interval": self.distill_interval,
                "distill_weight": self.distill_weight_init,
                "distill_lr": self.distill_lr,
                "distill_batch_size": self.distill_batch_size,
                "obs_buffer_size": self.obs_buffer_size,
                "distill_anneal": self.distill_anneal,
                "distill_anneal_end": self.distill_anneal_end,
                "distill_anneal_steps": self.distill_anneal_steps,
                "lr_anneal": self.lr_anneal,
                "lr_anneal_total_steps": self.lr_anneal_total_steps,
            },
            "scores": self.scores,
            "fitness": self.fitness,
            "steps": self.steps,
            "learn_counter": self.learn_counter,
            "distill_step_count": self.distill_step_count,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, observation_spaces, action_spaces, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        agent = cls(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=checkpoint["agent_ids"],
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_coef=cfg["clip_coef"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            update_epochs=cfg["update_epochs"],
            batch_size=cfg["batch_size"],
            latent_dim=cfg["latent_dim"],
            hidden_dims=cfg["hidden_dims"],
            distill_interval=cfg["distill_interval"],
            distill_weight=cfg["distill_weight"],
            distill_lr=cfg["distill_lr"],
            distill_batch_size=cfg["distill_batch_size"],
            obs_buffer_size=cfg["obs_buffer_size"],
            distill_anneal=cfg.get("distill_anneal", False),
            distill_anneal_end=cfg.get("distill_anneal_end", 0.0),
            distill_anneal_steps=cfg.get("distill_anneal_steps", 100),
            lr_anneal=cfg.get("lr_anneal", False),
            lr_anneal_total_steps=cfg.get("lr_anneal_total_steps", 200),
            device=device,
        )
        for aid in agent.agent_ids:
            agent.networks[aid].load_state_dict(checkpoint["networks"][aid])
            agent.optimizers[aid].load_state_dict(checkpoint["optimizers"][aid])
        agent.scores = checkpoint.get("scores", [])
        agent.fitness = checkpoint.get("fitness", [])
        agent.steps = checkpoint.get("steps", [0])
        agent.learn_counter = checkpoint.get("learn_counter", 0)
        agent.distill_step_count = checkpoint.get("distill_step_count", 0)
        return agent
