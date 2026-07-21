from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from g1_rl import G1ElbowTargetEnv
from g1_rl.dqn import DQNAgent, DQNConfig


EPSILON_DECAYS = {
    "config_a": 0.995,
    "config_b": 0.985,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN on the Unitree G1 elbow environment."
    )

    parser.add_argument(
        "--config",
        choices=sorted(EPSILON_DECAYS),
        default="config_a",
        help="Exploration-decay configuration.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by Python, NumPy, PyTorch, and Gymnasium.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print progress every N episodes.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short integration test instead of full training.",
    )

    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU-only and reproducible execution.
    torch.set_num_threads(1)


def create_agent_config(
    epsilon_decay: float,
    seed: int,
    smoke_test: bool,
) -> DQNConfig:
    if smoke_test:
        return DQNConfig(
            gamma=0.95,
            learning_rate=0.001,
            batch_size=32,
            replay_capacity=2_000,
            epsilon_start=1.00,
            epsilon_min=0.05,
            epsilon_decay=epsilon_decay,
            target_update_interval=25,
            warmup_transitions=64,
            gradient_clip_norm=10.0,
            seed=seed,
        )

    return DQNConfig(
        gamma=0.95,
        learning_rate=0.001,
        batch_size=64,
        replay_capacity=50_000,
        epsilon_start=1.00,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
        target_update_interval=250,
        warmup_transitions=500,
        gradient_clip_norm=10.0,
        seed=seed,
    )


def main() -> None:
    args = parse_arguments()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    if args.log_every <= 0:
        raise ValueError("--log-every must be greater than zero.")

    set_global_seeds(args.seed)

    device = torch.device("cpu")
    epsilon_decay = EPSILON_DECAYS[args.config]

    run_name = (
        f"smoke_{args.config}"
        if args.smoke_test
        else args.config
    )

    number_of_episodes = (
        12 if args.smoke_test else args.episodes
    )

    result_directory = Path("results") / run_name
    result_directory.mkdir(parents=True, exist_ok=True)

    model_directory = Path("models")
    model_directory.mkdir(parents=True, exist_ok=True)

    metrics_path = result_directory / "training_metrics.csv"
    summary_path = result_directory / "training_summary.json"

    final_checkpoint_path = (
        model_directory / f"{run_name}_final.pt"
    )
    best_checkpoint_path = (
        model_directory / f"{run_name}_best.pt"
    )

    config = create_agent_config(
        epsilon_decay=epsilon_decay,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

    env = G1ElbowTargetEnv(
        render_mode=None,
        goal_angle=None,
        goal_range=(-0.8, 0.8),
    )

    env.action_space.seed(args.seed)

    observation_dim = int(
        np.prod(env.observation_space.shape)
    )
    action_dim = int(env.action_space.n)

    agent = DQNAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=device,
        config=config,
    )

    reward_window: deque[float] = deque(maxlen=20)
    success_window: deque[float] = deque(maxlen=50)

    best_score = (-1.0, float("-inf"))
    training_start = time.perf_counter()

    fieldnames = [
        "episode",
        "goal_angle",
        "cumulative_reward",
        "success",
        "episode_length",
        "final_absolute_error",
        "epsilon",
        "mean_loss",
        "optimization_steps",
        "replay_buffer_size",
        "elapsed_seconds",
    ]

    print("=== DQN TRAINING ===")
    print(f"Run:             {run_name}")
    print(f"Device:          {device}")
    print(f"Episodes:        {number_of_episodes}")
    print(f"Epsilon decay:   {config.epsilon_decay}")
    print(f"Goal range:      [-0.8, +0.8] rad")
    print(f"Smoke test:      {args.smoke_test}")
    print()

    try:
        with metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for episode in range(1, number_of_episodes + 1):
                episode_seed = args.seed + episode - 1

                observation, reset_info = env.reset(
                    seed=episode_seed,
                )

                goal_angle = float(
                    reset_info["goal_angle"]
                )
                epsilon_used = agent.epsilon

                cumulative_reward = 0.0
                episode_losses: list[float] = []

                terminated = False
                truncated = False
                info = reset_info

                while not (terminated or truncated):
                    action = agent.select_action(
                        observation=observation,
                        greedy=False,
                    )

                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    agent.store_transition(
                        state=observation,
                        action=action,
                        reward=reward,
                        next_state=next_observation,
                        terminated=terminated,
                        truncated=truncated,
                    )

                    loss = agent.optimize_model()

                    if loss is not None:
                        episode_losses.append(loss)

                    observation = next_observation
                    cumulative_reward += float(reward)

                success = bool(
                    info.get("is_success", False)
                )
                episode_length = int(
                    info["episode_step"]
                )
                final_absolute_error = float(
                    info["absolute_error"]
                )

                mean_loss = (
                    float(np.mean(episode_losses))
                    if episode_losses
                    else None
                )

                elapsed_seconds = (
                    time.perf_counter() - training_start
                )

                reward_window.append(cumulative_reward)
                success_window.append(float(success))

                writer.writerow(
                    {
                        "episode": episode,
                        "goal_angle": f"{goal_angle:.8f}",
                        "cumulative_reward": (
                            f"{cumulative_reward:.8f}"
                        ),
                        "success": int(success),
                        "episode_length": episode_length,
                        "final_absolute_error": (
                            f"{final_absolute_error:.8f}"
                        ),
                        "epsilon": f"{epsilon_used:.8f}",
                        "mean_loss": (
                            ""
                            if mean_loss is None
                            else f"{mean_loss:.8f}"
                        ),
                        "optimization_steps": (
                            agent.optimization_steps
                        ),
                        "replay_buffer_size": (
                            len(agent.replay_buffer)
                        ),
                        "elapsed_seconds": (
                            f"{elapsed_seconds:.4f}"
                        ),
                    }
                )
                metrics_file.flush()

                agent.decay_epsilon()

                if len(success_window) == success_window.maxlen:
                    current_score = (
                        float(np.mean(success_window)),
                        float(np.mean(reward_window)),
                    )

                    if current_score > best_score:
                        best_score = current_score
                        agent.save_checkpoint(
                            best_checkpoint_path
                        )

                if (
                    episode == 1
                    or episode % args.log_every == 0
                    or episode == number_of_episodes
                ):
                    rolling_reward = float(
                        np.mean(reward_window)
                    )
                    rolling_success = float(
                        np.mean(success_window)
                    )

                    print(
                        f"episode={episode:4d} | "
                        f"goal={goal_angle:+.3f} | "
                        f"reward={cumulative_reward:+.3f} | "
                        f"success={int(success)} | "
                        f"steps={episode_length:3d} | "
                        f"error={final_absolute_error:.4f} | "
                        f"epsilon={epsilon_used:.4f} | "
                        f"reward20={rolling_reward:+.3f} | "
                        f"success50={rolling_success:.1%}"
                    )

        agent.save_checkpoint(final_checkpoint_path)

        if not best_checkpoint_path.exists():
            agent.save_checkpoint(best_checkpoint_path)

        total_training_seconds = (
            time.perf_counter() - training_start
        )

        summary = {
            "run_name": run_name,
            "configuration": args.config,
            "smoke_test": args.smoke_test,
            "episodes": number_of_episodes,
            "seed": args.seed,
            "device": str(device),
            "goal_range": [-0.8, 0.8],
            "dqn_config": asdict(config),
            "training_seconds": total_training_seconds,
            "final_epsilon": agent.epsilon,
            "final_20_mean_reward": float(
                np.mean(reward_window)
            ),
            "final_50_success_rate": float(
                np.mean(success_window)
            ),
            "optimization_steps": (
                agent.optimization_steps
            ),
            "replay_buffer_size": (
                len(agent.replay_buffer)
            ),
            "metrics_file": str(metrics_path),
            "final_checkpoint": str(
                final_checkpoint_path
            ),
            "best_checkpoint": str(
                best_checkpoint_path
            ),
        }

        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print()
        print("=== TRAINING COMPLETE ===")
        print(
            f"Training time:     "
            f"{total_training_seconds:.2f} seconds"
        )
        print(
            f"Final epsilon:     {agent.epsilon:.4f}"
        )
        print(
            f"Final reward mean: "
            f"{np.mean(reward_window):.4f}"
        )
        print(
            f"Final success rate:"
            f" {np.mean(success_window):.1%}"
        )
        print(f"Metrics:           {metrics_path}")
        print(f"Summary:           {summary_path}")
        print(
            f"Final checkpoint:  {final_checkpoint_path}"
        )
        print(
            f"Best checkpoint:   {best_checkpoint_path}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
