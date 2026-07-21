from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity experience replay memory for DQN."""

    def __init__(self, capacity: int = 50_000, seed: int = 42) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.capacity = capacity
        self._memory: Deque[Transition] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        transition = Transition(
            state=np.asarray(state, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32).copy(),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        self._memory.append(transition)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if batch_size > len(self._memory):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from "
                f"a buffer containing {len(self._memory)}."
            )

        transitions = self._random.sample(list(self._memory), batch_size)

        states = torch.as_tensor(
            np.stack([transition.state for transition in transitions]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.as_tensor(
            [transition.action for transition in transitions],
            dtype=torch.int64,
            device=device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.stack([transition.next_state for transition in transitions]),
            dtype=torch.float32,
            device=device,
        )
        terminated = torch.as_tensor(
            [transition.terminated for transition in transitions],
            dtype=torch.bool,
            device=device,
        ).unsqueeze(1)
        truncated = torch.as_tensor(
            [transition.truncated for transition in transitions],
            dtype=torch.bool,
            device=device,
        ).unsqueeze(1)

        return ReplayBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
        )

    def __len__(self) -> int:
        return len(self._memory)
