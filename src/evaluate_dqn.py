from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from g1_rl import G1ElbowTargetEnv
from g1_rl.dqn import DQNAgent, DQNConfig
from test_g1_elbow_env import choose_rule_based_action


BENCHMARK_GOALS = (-0.8, -0.4, 0.4, 0.8)
EPISODES_PER_GOAL = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a DQN or rule-based policy on the "
            "four required Unitree G1 benchmark goals."
        )
    )

    parser.add_argument(
        "--policy",
        choices=("dqn", "rule_based"),
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Required when --policy dqn is selected.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Output name such as config_a, config_b, or rule_based.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_dqn_agent(
    checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
) -> DQNAgent:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config_data = checkpoint.get("config", {})
    config = DQNConfig(**config_data)

    agent = DQNAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=torch.device("cpu"),
        config=config,
    )

    agent.load_checkpoint(checkpoint_path)

    # Final evaluation must be completely greedy.
    agent.epsilon = 0.0
    agent.online_network.eval()

    return agent


def evaluate_episode(
    env: G1ElbowTargetEnv,
    policy_name: str,
    seed: int,
    agent: DQNAgent | None,
) -> dict[str, object]:
    observation, info = env.reset(seed=seed)

    cumulative_reward = 0.0
    actions: list[int] = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        if policy_name == "dqn":
            if agent is None:
                raise RuntimeError("DQN agent was not loaded.")

            action = agent.select_action(
                observation,
                greedy=True,
            )
        else:
            action = choose_rule_based_action(
                observation=observation,
                controller_target=float(
                    info["controller_target"]
                ),
                action_increment=env.action_increment,
            )

        actions.append(action)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        cumulative_reward += float(reward)

    action_counts = Counter(actions)

    action_changes = sum(
        previous != current
        for previous, current in zip(
            actions,
            actions[1:],
        )
    )

    return {
        "success": bool(info.get("is_success", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "cumulative_reward": cumulative_reward,
        "episode_length": int(info["episode_step"]),
        "final_absolute_error": float(
            info["absolute_error"]
        ),
        "decrease_actions": action_counts.get(
            G1ElbowTargetEnv.ACTION_DECREASE,
            0,
        ),
        "hold_actions": action_counts.get(
            G1ElbowTargetEnv.ACTION_HOLD,
            0,
        ),
        "increase_actions": action_counts.get(
            G1ElbowTargetEnv.ACTION_INCREASE,
            0,
        ),
        "action_changes": action_changes,
    }


def summarize_rows(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    def create_summary(
        selected_rows: list[dict[str, object]],
    ) -> dict[str, object]:
        successes = sum(
            int(row["success"])
            for row in selected_rows
        )

        return {
            "episodes": len(selected_rows),
            "successes": successes,
            "success_rate": (
                successes / len(selected_rows)
            ),
            "mean_cumulative_reward": float(
                np.mean(
                    [
                        row["cumulative_reward"]
                        for row in selected_rows
                    ]
                )
            ),
            "mean_episode_length": float(
                np.mean(
                    [
                        row["episode_length"]
                        for row in selected_rows
                    ]
                )
            ),
            "mean_final_absolute_error": float(
                np.mean(
                    [
                        row["final_absolute_error"]
                        for row in selected_rows
                    ]
                )
            ),
            "mean_hold_actions": float(
                np.mean(
                    [
                        row["hold_actions"]
                        for row in selected_rows
                    ]
                )
            ),
            "mean_action_changes": float(
                np.mean(
                    [
                        row["action_changes"]
                        for row in selected_rows
                    ]
                )
            ),
        }

    by_goal: dict[str, object] = {}

    for goal in BENCHMARK_GOALS:
        goal_rows = [
            row
            for row in rows
            if float(row["goal_angle"]) == goal
        ]

        by_goal[f"{goal:+.1f}"] = create_summary(
            goal_rows
        )

    return {
        "by_goal": by_goal,
        "overall": create_summary(rows),
    }


def main() -> None:
    args = parse_arguments()

    if args.policy == "dqn" and args.checkpoint is None:
        raise ValueError(
            "--checkpoint is required for DQN evaluation."
        )

    output_directory = Path("results") / args.name
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_directory / "evaluation_metrics.csv"
    )
    summary_path = (
        output_directory / "evaluation_summary.json"
    )

    rows: list[dict[str, object]] = []
    agent: DQNAgent | None = None

    first_env = G1ElbowTargetEnv(
        render_mode=None,
        goal_angle=BENCHMARK_GOALS[0],
    )

    try:
        if args.policy == "dqn":
            agent = load_dqn_agent(
                checkpoint_path=args.checkpoint,
                observation_dim=int(
                    np.prod(
                        first_env.observation_space.shape
                    )
                ),
                action_dim=int(
                    first_env.action_space.n
                ),
            )
    finally:
        first_env.close()

    print("=== GREEDY BENCHMARK EVALUATION ===")
    print(f"Policy:             {args.policy}")
    print(f"Name:               {args.name}")
    print(
        f"Checkpoint:         "
        f"{args.checkpoint if args.checkpoint else 'N/A'}"
    )
    print("Evaluation epsilon: 0.0")
    print("Episodes:           20")
    print()

    for goal_index, goal_angle in enumerate(
        BENCHMARK_GOALS
    ):
        env = G1ElbowTargetEnv(
            render_mode=None,
            goal_angle=goal_angle,
        )

        try:
            for episode_index in range(
                EPISODES_PER_GOAL
            ):
                seed = (
                    args.seed
                    + goal_index * 100
                    + episode_index
                )

                result = evaluate_episode(
                    env=env,
                    policy_name=args.policy,
                    seed=seed,
                    agent=agent,
                )

                row = {
                    "policy": args.policy,
                    "name": args.name,
                    "checkpoint": (
                        str(args.checkpoint)
                        if args.checkpoint
                        else ""
                    ),
                    "goal_angle": goal_angle,
                    "episode": episode_index + 1,
                    "seed": seed,
                    **result,
                }

                rows.append(row)

                print(
                    f"goal={goal_angle:+.1f} | "
                    f"episode={episode_index + 1} | "
                    f"success={int(result['success'])} | "
                    f"reward="
                    f"{result['cumulative_reward']:+.4f} | "
                    f"steps={result['episode_length']:3d} | "
                    f"error="
                    f"{result['final_absolute_error']:.4f}"
                )
        finally:
            env.close()

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "policy": args.policy,
        "name": args.name,
        "checkpoint": (
            str(args.checkpoint)
            if args.checkpoint
            else None
        ),
        "evaluation_epsilon": 0.0,
        "benchmark_goals": list(BENCHMARK_GOALS),
        "episodes_per_goal": EPISODES_PER_GOAL,
        **summarize_rows(rows),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    overall = summary["overall"]

    print()
    print("=== EVALUATION SUMMARY ===")

    for goal, goal_summary in summary[
        "by_goal"
    ].items():
        print(
            f"Goal {goal} rad: "
            f"{goal_summary['successes']}/"
            f"{goal_summary['episodes']} successes | "
            f"reward="
            f"{goal_summary['mean_cumulative_reward']:.4f}"
        )

    print()
    print(
        f"Overall successes:  "
        f"{overall['successes']}/"
        f"{overall['episodes']}"
    )
    print(
        f"Overall success:     "
        f"{overall['success_rate']:.1%}"
    )
    print(
        f"Mean reward:         "
        f"{overall['mean_cumulative_reward']:.4f}"
    )
    print(
        f"Mean episode length: "
        f"{overall['mean_episode_length']:.2f}"
    )
    print(
        f"Mean final error:    "
        f"{overall['mean_final_absolute_error']:.4f}"
    )
    print(
        f"Mean HOLD actions:   "
        f"{overall['mean_hold_actions']:.2f}"
    )
    print(
        f"Mean action changes: "
        f"{overall['mean_action_changes']:.2f}"
    )
    print(f"Metrics:             {metrics_path}")
    print(f"Summary:             {summary_path}")


if __name__ == "__main__":
    main()
