from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Maps a 4-value observation to one Q-value per discrete action."""

    def __init__(
        self,
        observation_dim: int = 4,
        action_dim: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        linear_layers = [
            layer for layer in self.network if isinstance(layer, nn.Linear)
        ]

        for layer in linear_layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.to(dtype=torch.float32)
        return self.network(observation)
