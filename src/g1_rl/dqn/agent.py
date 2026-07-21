from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from g1_rl.dqn.q_network import QNetwork
from g1_rl.dqn.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.95
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_capacity: int = 50_000
    epsilon_start: float = 1.00
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    target_update_interval: int = 250
    warmup_transitions: int = 500
    gradient_clip_norm: float = 10.0
    seed: int = 42


class DQNAgent:
    """Student-written Deep Q-Network agent."""

    def __init__(
        self,
        observation_dim: int = 4,
        action_dim: int = 3,
        device: torch.device | None = None,
        config: DQNConfig | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")

        self._random = random.Random(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.online_network = QNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.target_network = QNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_capacity,
            seed=self.config.seed,
        )

        self.epsilon = self.config.epsilon_start
        self.optimization_steps = 0

        self.sync_target_network()

    def select_action(
        self,
        observation: np.ndarray,
        greedy: bool = False,
    ) -> int:
        """Choose an action using epsilon-greedy exploration."""

        explore = (
            not greedy
            and self._random.random() < self.epsilon
        )

        if explore:
            return self._random.randrange(self.action_dim)

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(observation_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )

    def optimize_model(self) -> float | None:
        """Perform one mini-batch Bellman optimization step."""

        required_samples = max(
            self.config.batch_size,
            self.config.warmup_transitions,
        )

        if len(self.replay_buffer) < required_samples:
            return None

        batch = self.replay_buffer.sample(
            batch_size=self.config.batch_size,
            device=self.device,
        )

        selected_q_values = self.online_network(
            batch.states
        ).gather(1, batch.actions)

        with torch.no_grad():
            next_q_values = self.target_network(
                batch.next_states
            ).max(dim=1, keepdim=True).values

            # A true termination ends the task and prevents bootstrapping.
            # A truncation represents the environment time limit, so we
            # continue bootstrapping from its next state.
            bootstrap_mask = (~batch.terminated).to(
                dtype=torch.float32
            )

            bellman_targets = (
                batch.rewards
                + self.config.gamma
                * bootstrap_mask
                * next_q_values
            )

        loss = nn.functional.smooth_l1_loss(
            selected_q_values,
            bellman_targets,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.config.gradient_clip_norm,
        )

        self.optimizer.step()
        self.optimization_steps += 1

        if (
            self.optimization_steps
            % self.config.target_update_interval
            == 0
        ):
            self.sync_target_network()

        return float(loss.item())

    def decay_epsilon(self) -> float:
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        return self.epsilon

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
        self.target_network.eval()

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "optimization_steps": self.optimization_steps,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "config": asdict(self.config),
                "agent_random_state": self._random.getstate(),
                "torch_random_state": torch.get_rng_state(),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )

        self.online_network.load_state_dict(
            checkpoint["online_network"]
        )
        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = float(checkpoint["epsilon"])
        self.optimization_steps = int(
            checkpoint["optimization_steps"]
        )

        if "agent_random_state" in checkpoint:
            self._random.setstate(
                checkpoint["agent_random_state"]
            )

        if "torch_random_state" in checkpoint:
            torch.set_rng_state(
                checkpoint["torch_random_state"]
            )

        self.target_network.eval()
