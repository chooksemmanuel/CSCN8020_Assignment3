from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from g1_rl import G1ElbowTargetEnv
from g1_rl.dqn import DQNAgent, DQNConfig


ACTION_NAMES = {
    G1ElbowTargetEnv.ACTION_DECREASE: "DECREASE",
    G1ElbowTargetEnv.ACTION_HOLD: "HOLD",
    G1ElbowTargetEnv.ACTION_INCREASE: "INCREASE",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a trained DQN controlling the Unitree G1 elbow."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/selected_dqn.pt"),
        help="Saved PyTorch DQN checkpoint.",
    )
    parser.add_argument(
        "--goals",
        type=float,
        nargs="+",
        default=[-0.8, 0.8],
        help="One or more elbow target angles.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.08,
        help="Extra delay after each Gymnasium step.",
    )

    return parser.parse_args()


def load_agent(
    checkpoint_path: Path,
    env: G1ElbowTargetEnv,
) -> DQNAgent:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = DQNConfig(
        **checkpoint.get("config", {})
    )

    agent = DQNAgent(
        observation_dim=int(
            np.prod(env.observation_space.shape)
        ),
        action_dim=int(env.action_space.n),
        device=torch.device("cpu"),
        config=config,
    )

    agent.load_checkpoint(checkpoint_path)

    agent.epsilon = 0.0
    agent.online_network.eval()

    return agent


def calculate_q_values(
    agent: DQNAgent,
    observation: np.ndarray,
) -> torch.Tensor:
    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():
        return agent.online_network(
            observation_tensor
        ).squeeze(0)


def countdown(seconds: int) -> None:
    print()
    print("Starting in:")

    for remaining in range(seconds, 0, -1):
        print(remaining)
        time.sleep(1.0)

    print("GO")
    print()


def run_episode(
    env: G1ElbowTargetEnv,
    agent: DQNAgent,
    goal_angle: float,
    seed: int,
    countdown_seconds: int,
    step_delay: float,
) -> None:
    observation, info = env.reset(
        seed=seed,
        options={"goal_angle": goal_angle},
    )

    env.render()

    countdown(countdown_seconds)

    cumulative_reward = 0.0
    terminated = False
    truncated = False

    print("=== TRAINED DQN EPISODE ===")
    print(f"Checkpoint epsilon: {agent.epsilon:.1f}")
    print(f"Target goal:        {goal_angle:+.4f} rad")
    print()

    while not (terminated or truncated):
        q_values = calculate_q_values(
            agent,
            observation,
        )

        action = int(
            q_values.argmax().item()
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        cumulative_reward += float(reward)

        print(
            f"step={info['episode_step']:3d} | "
            f"action={action} "
            f"({ACTION_NAMES[action]:8s}) | "
            f"Q=["
            f"{q_values[0].item():+.3f}, "
            f"{q_values[1].item():+.3f}, "
            f"{q_values[2].item():+.3f}] | "
            f"angle={info['elbow_angle']:+.4f} | "
            f"target={info['controller_target']:+.4f} | "
            f"goal={info['goal_angle']:+.4f} | "
            f"error={info['angle_error']:+.4f} | "
            f"streak={info['success_streak']:2d} | "
            f"reward={reward:+.4f}"
        )

        if step_delay > 0:
            time.sleep(step_delay)

    print()
    print("=== EPISODE RESULT ===")
    print(f"Goal angle:        {info['goal_angle']:+.4f}")
    print(f"Final angle:       {info['elbow_angle']:+.4f}")
    print(f"Final error:       {info['angle_error']:+.4f}")
    print(f"Episode steps:     {info['episode_step']}")
    print(f"Cumulative reward: {cumulative_reward:.4f}")
    print(f"Terminated:        {terminated}")
    print(f"Truncated:         {truncated}")
    print(f"Success:           {info['is_success']}")
    print()


def main() -> None:
    args = parse_arguments()

    if args.countdown < 0:
        raise ValueError(
            "--countdown cannot be negative."
        )

    if args.step_delay < 0:
        raise ValueError(
            "--step-delay cannot be negative."
        )

    for goal in args.goals:
        if not -0.8 <= goal <= 0.8:
            raise ValueError(
                "Every demonstration goal must be "
                "within [-0.8, +0.8] rad."
            )

    env = G1ElbowTargetEnv(
        render_mode="human",
        goal_angle=None,
        goal_range=(-0.8, 0.8),
    )

    try:
        agent = load_agent(
            checkpoint_path=args.checkpoint,
            env=env,
        )

        first_goal = float(args.goals[0])

        env.reset(
            seed=args.seed,
            options={"goal_angle": first_goal},
        )
        env.render()

        print()
        print("=== DQN VIDEO DEMONSTRATION ===")
        print(f"Checkpoint: {args.checkpoint.resolve()}")
        print("Device:     CPU")
        print("Epsilon:    0.0, fully greedy")
        print(f"Goals:      {args.goals}")
        print()
        print("Position the camera so the left arm is visible.")
        print("Start your screen recording before continuing.")
        print()

        input("Press Enter when the camera and recording are ready...")

        for index, goal_angle in enumerate(
            args.goals,
            start=1,
        ):
            print()
            print(
                f"DEMONSTRATION {index}/{len(args.goals)}"
            )

            run_episode(
                env=env,
                agent=agent,
                goal_angle=float(goal_angle),
                seed=args.seed + index - 1,
                countdown_seconds=args.countdown,
                step_delay=args.step_delay,
            )

            if index < len(args.goals):
                input(
                    "Press Enter to begin the next target angle..."
                )

        print("All trained-DQN demonstrations are complete.")
        print()
        print(
            "Close the MuJoCo viewer window manually "
            "to finish."
        )

        while (
            env.viewer is not None
            and env.viewer.is_running()
        ):
            time.sleep(0.05)

    finally:
        if (
            env.viewer is None
            or env.viewer.is_running()
        ):
            env.close()


if __name__ == "__main__":
    main()
