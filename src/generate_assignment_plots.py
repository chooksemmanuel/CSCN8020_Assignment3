from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "Configuration A (0.995)": "config_a",
    "Configuration B (0.985)": "config_b",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_figure(filename: str) -> None:
    output_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_rewards() -> None:
    for display_name, folder_name in CONFIGS.items():
        metrics = pd.read_csv(
            RESULTS_DIR / folder_name / "training_metrics.csv"
        )

        metrics["reward_moving_average_20"] = (
            metrics["cumulative_reward"]
            .rolling(window=20, min_periods=1)
            .mean()
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            metrics["episode"],
            metrics["cumulative_reward"],
            alpha=0.35,
            label="Raw episode reward",
        )
        plt.plot(
            metrics["episode"],
            metrics["reward_moving_average_20"],
            linewidth=2,
            label="20-episode moving average",
        )

        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.title(f"Training Reward: {display_name}")
        plt.legend()
        plt.grid(alpha=0.25)

        save_figure(f"{folder_name}_training_reward.png")


def plot_success_rate() -> None:
    plt.figure(figsize=(10, 6))

    for display_name, folder_name in CONFIGS.items():
        metrics = pd.read_csv(
            RESULTS_DIR / folder_name / "training_metrics.csv"
        )

        rolling_success = (
            metrics["success"]
            .rolling(window=50, min_periods=1)
            .mean()
            * 100
        )

        plt.plot(
            metrics["episode"],
            rolling_success,
            linewidth=2,
            label=display_name,
        )

    plt.axhline(
        80,
        linestyle="--",
        linewidth=1.5,
        label="Required 80% threshold",
    )
    plt.xlabel("Episode")
    plt.ylabel("Rolling success rate (%)")
    plt.title("Training Success Rate: 50-Episode Rolling Window")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure("training_success_rate_comparison.png")


def plot_epsilon_decay() -> None:
    plt.figure(figsize=(10, 6))

    for display_name, folder_name in CONFIGS.items():
        metrics = pd.read_csv(
            RESULTS_DIR / folder_name / "training_metrics.csv"
        )

        plt.plot(
            metrics["episode"],
            metrics["epsilon"],
            linewidth=2,
            label=display_name,
        )

    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Comparison")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure("epsilon_decay_comparison.png")


def plot_training_loss() -> None:
    plt.figure(figsize=(10, 6))

    for display_name, folder_name in CONFIGS.items():
        metrics = pd.read_csv(
            RESULTS_DIR / folder_name / "training_metrics.csv"
        )

        valid_loss = metrics.dropna(subset=["mean_loss"]).copy()
        valid_loss["loss_moving_average_20"] = (
            valid_loss["mean_loss"]
            .rolling(window=20, min_periods=1)
            .mean()
        )

        plt.plot(
            valid_loss["episode"],
            valid_loss["loss_moving_average_20"],
            linewidth=2,
            label=display_name,
        )

    plt.xlabel("Episode")
    plt.ylabel("Mean Huber loss")
    plt.title("DQN Optimization Loss: 20-Episode Moving Average")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure("training_loss_comparison.png")


def build_configuration_comparison() -> pd.DataFrame:
    rows = []

    for display_name, folder_name in CONFIGS.items():
        training = load_json(
            RESULTS_DIR / folder_name / "training_summary.json"
        )
        evaluation = load_json(
            RESULTS_DIR / folder_name / "evaluation_summary.json"
        )

        rows.append(
            {
                "configuration": display_name,
                "epsilon_decay": training["dqn_config"][
                    "epsilon_decay"
                ],
                "training_episodes": training["episodes"],
                "training_time_seconds": training[
                    "training_seconds"
                ],
                "final_epsilon": training["final_epsilon"],
                "final_20_mean_training_reward": training[
                    "final_20_mean_reward"
                ],
                "final_50_training_success_rate": training[
                    "final_50_success_rate"
                ],
                "evaluation_successes": evaluation["overall"][
                    "successes"
                ],
                "evaluation_episodes": evaluation["overall"][
                    "episodes"
                ],
                "evaluation_success_rate": evaluation["overall"][
                    "success_rate"
                ],
                "mean_evaluation_reward": evaluation["overall"][
                    "mean_cumulative_reward"
                ],
                "mean_episode_length": evaluation["overall"][
                    "mean_episode_length"
                ],
                "mean_final_absolute_error": evaluation["overall"][
                    "mean_final_absolute_error"
                ],
                "mean_hold_actions": evaluation["overall"][
                    "mean_hold_actions"
                ],
                "mean_action_changes": evaluation["overall"][
                    "mean_action_changes"
                ],
            }
        )

    comparison = pd.DataFrame(rows)
    output_path = PLOTS_DIR / "configuration_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return comparison


def plot_configuration_evaluation(
    comparison: pd.DataFrame,
) -> None:
    labels = comparison["configuration"]
    rewards = comparison["mean_evaluation_reward"]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, rewards)

    plt.ylabel("Mean cumulative reward")
    plt.title("Greedy Evaluation Reward by DQN Configuration")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, rewards):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    save_figure("dqn_configuration_evaluation_reward.png")


def plot_success_by_goal() -> None:
    config_a = load_json(
        RESULTS_DIR / "config_a" / "evaluation_summary.json"
    )
    config_b = load_json(
        RESULTS_DIR / "config_b" / "evaluation_summary.json"
    )
    rule_based = load_json(
        RESULTS_DIR / "rule_based" / "evaluation_summary.json"
    )

    goals = ["-0.8", "-0.4", "+0.4", "+0.8"]
    x = np.arange(len(goals))
    width = 0.25

    config_a_rates = [
        config_a["by_goal"][goal]["success_rate"] * 100
        for goal in goals
    ]
    config_b_rates = [
        config_b["by_goal"][goal]["success_rate"] * 100
        for goal in goals
    ]
    rule_rates = [
        rule_based["by_goal"][goal]["success_rate"] * 100
        for goal in goals
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width,
        config_a_rates,
        width,
        label="DQN Config A",
    )
    plt.bar(
        x,
        config_b_rates,
        width,
        label="DQN Config B",
    )
    plt.bar(
        x + width,
        rule_rates,
        width,
        label="Rule-based",
    )

    plt.xticks(x, [f"{goal} rad" for goal in goals])
    plt.ylabel("Success rate (%)")
    plt.xlabel("Target angle")
    plt.title("Greedy Evaluation Success Rate by Target Angle")
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    save_figure("evaluation_success_rate_by_goal.png")


def build_policy_comparison() -> pd.DataFrame:
    policies = {
        "Selected DQN Config A": "config_a",
        "DQN Config B": "config_b",
        "Rule-based policy": "rule_based",
    }

    rows = []

    for display_name, folder_name in policies.items():
        evaluation = load_json(
            RESULTS_DIR / folder_name / "evaluation_summary.json"
        )
        overall = evaluation["overall"]

        rows.append(
            {
                "policy": display_name,
                "successes": overall["successes"],
                "episodes": overall["episodes"],
                "success_rate": overall["success_rate"],
                "mean_cumulative_reward": overall[
                    "mean_cumulative_reward"
                ],
                "mean_episode_length": overall[
                    "mean_episode_length"
                ],
                "mean_final_absolute_error": overall[
                    "mean_final_absolute_error"
                ],
                "mean_hold_actions": overall[
                    "mean_hold_actions"
                ],
                "mean_action_changes": overall[
                    "mean_action_changes"
                ],
            }
        )

    comparison = pd.DataFrame(rows)
    output_path = PLOTS_DIR / "policy_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return comparison


def plot_policy_episode_length(
    policy_comparison: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        policy_comparison["policy"],
        policy_comparison["mean_episode_length"],
    )

    plt.ylabel("Mean episode length")
    plt.title("Policy Efficiency on the 20 Benchmark Episodes")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(
        bars,
        policy_comparison["mean_episode_length"],
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=10)

    save_figure("policy_episode_length_comparison.png")


def main() -> None:
    plot_training_rewards()
    plot_success_rate()
    plot_epsilon_decay()
    plot_training_loss()

    configuration_comparison = (
        build_configuration_comparison()
    )
    plot_configuration_evaluation(
        configuration_comparison
    )

    plot_success_by_goal()

    policy_comparison = build_policy_comparison()
    plot_policy_episode_length(policy_comparison)

    print()
    print("All required plots and comparison tables generated.")


if __name__ == "__main__":
    main()
