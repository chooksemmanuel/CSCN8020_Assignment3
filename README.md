# CSCN8020 Assignment 3 — Deep Q-Network Control of the Unitree G1 Left Elbow

<!-- MANUAL-REVIEW-SNAPSHOT:START -->

## Manual Review Snapshot

| Item | Details |
|---|---|
| **Student** | Emmanuel Ihejiamaizu |
| **Student ID** | 9080005 |
| **Course** | CSCN8020 — Reinforcement Learning |
| **Instructor** | Prof. David Espinosa |
| **Institution** | Conestoga College, Ontario, Canada |
| **Public repository** | <https://github.com/chooksemmanuel/CSCN8020_Assignment3> |
| **Clone URL** | `https://github.com/chooksemmanuel/CSCN8020_Assignment3.git` |
| **Selected model** | [`models/selected_dqn.pt`](models/selected_dqn.pt) |
| **Demonstration video** | [`demo/Unitree_G1_DQN_Demo_Final.mp4`](demo/Unitree_G1_DQN_Demo_Final.mp4) |
| **Technical report** | [`report/Unitree_G1_DQN_Technical_Report_Emmanuel_Ihejiamaizu.pdf`](report/Unitree_G1_DQN_Technical_Report_Emmanuel_Ihejiamaizu.pdf) |

### Project Summary

This project implements a student-written PyTorch Deep Q-Network to control the **left elbow of a fixed-base Unitree G1 humanoid in MuJoCo**. The agent observes four values — elbow angle, elbow velocity, target angle, and target error — and chooses one of three discrete actions: decrease, hold, or increase the internal joint-position target used by the approved low-level PD controller.

Two exploration-decay configurations were trained for **1,000 episodes each** using seed 42. Both achieved **20/20 benchmark successes** on the required targets `[-0.8, -0.4, 0.4, 0.8]` radians. Configuration A (`epsilon_decay = 0.995`) was selected because it produced the stronger final evaluation profile: slightly higher mean reward, shorter episodes, and lower final absolute error than Configuration B. The selected DQN also completed the benchmark faster and with a higher mean reward than the rule-based baseline.

### Headline Results

| Metric | Rule-based baseline | Selected DQN — Config A |
|---|---:|---:|
| Successes | 20 / 20 | **20 / 20** |
| Success rate | 100% | **100%** |
| Mean cumulative reward | 12.8666 | **13.1796** |
| Mean episode length | 24.00 steps | **19.50 steps** |
| Mean final absolute error | 0.0122 rad | **0.0116 rad** |
| Mean HOLD actions | 16.50 | **4.75** |

### Configuration Selection

| Configuration | Epsilon decay | Success | Mean reward | Mean length | Mean final error |
|---|---:|---:|---:|---:|---:|
| **A — selected** | 0.995 | **20 / 20** | **13.1796** | **19.50** | **0.0116** |
| B | 0.985 | 20 / 20 | 13.1588 | 19.75 | 0.0127 |

The selected checkpoint is [`models/selected_dqn.pt`](models/selected_dqn.pt). It is the retained Config A model used for the final greedy evaluation and rendered demonstration.

### Open These First

1. **Completed assignment notebook:** [`CSCN8020_Assignment3.ipynb`](CSCN8020_Assignment3.ipynb)
   - Sections **1.1–1.4** explicitly map the MDP, reward, Q-values, Bellman target, terminal mask, and Huber loss to the source code and executable verification cells.
2. **Selected checkpoint:** [`models/selected_dqn.pt`](models/selected_dqn.pt)
3. **Config A evaluation summary:** [`results/config_a/evaluation_summary.json`](results/config_a/evaluation_summary.json)
4. **Config A training summary:** [`results/config_a/training_summary.json`](results/config_a/training_summary.json)
5. **Comparison plots:** [`results/plots/`](results/plots/)
6. **Rendered DQN demonstration:** [`demo/Unitree_G1_DQN_Demo_Final.mp4`](demo/Unitree_G1_DQN_Demo_Final.mp4)
7. **Full technical report:** [`report/Unitree_G1_DQN_Technical_Report_Emmanuel_Ihejiamaizu.pdf`](report/Unitree_G1_DQN_Technical_Report_Emmanuel_Ihejiamaizu.pdf)

### Fast Manual Reproduction

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open the completed assignment notebook:

```bash
jupyter notebook CSCN8020_Assignment3.ipynb
```

Evaluate the selected model without retraining:

```bash
PYTHONPATH=src python src/evaluate_dqn.py \
  --policy dqn \
  --checkpoint models/selected_dqn.pt \
  --name config_a \
  --seed 42
```

Render the selected policy:

```bash
PYTHONPATH=src python src/render_dqn_policy.py \
  --checkpoint models/selected_dqn.pt \
  --goals -0.8 0.8 \
  --seed 42 \
  --countdown 3 \
  --step-delay 0.08
```

### Validated Environment

| Component | Validated version |
|---|---|
| Python | 3.14.4 |
| Operating environment | Ubuntu 26.04 LTS under WSL 2 |
| Compute | CPU |
| PyTorch | 2.13.0+cpu |
| Gymnasium | 1.3.0 |
| MuJoCo | 3.10.0 |
| NumPy | 2.5.1 |
| Pandas | 3.0.3 |
| Matplotlib | 3.11.1 |
| Jupyter Notebook | 7.6.1 |

The repository was also validated through a **fresh public clone**, a **new virtual environment**, a clean `requirements.txt` installation, source compilation, checkpoint loading, the full 20-episode evaluation, and end-to-end execution of `CSCN8020_Assignment3.ipynb`.

### Student-Written DQN Components

| Component | File |
|---|---|
| Q-network | [`src/g1_rl/dqn/q_network.py`](src/g1_rl/dqn/q_network.py) |
| Replay buffer | [`src/g1_rl/dqn/replay_buffer.py`](src/g1_rl/dqn/replay_buffer.py) |
| DQN agent, optimization, target synchronization, checkpoint handling | [`src/g1_rl/dqn/agent.py`](src/g1_rl/dqn/agent.py) |
| Training loop | [`src/train_dqn.py`](src/train_dqn.py) |
| Shared DQN / rule-based evaluation | [`src/evaluate_dqn.py`](src/evaluate_dqn.py) |
| Policy rendering | [`src/render_dqn_policy.py`](src/render_dqn_policy.py) |
| Custom Gymnasium environment | [`src/g1_rl/g1_elbow_env.py`](src/g1_rl/g1_elbow_env.py) |
| Rule-based controller and environment test | [`src/test_g1_elbow_env.py`](src/test_g1_elbow_env.py) |
| Plot generation | [`src/generate_assignment_plots.py`](src/generate_assignment_plots.py) |

No Stable-Baselines3 or other turnkey reinforcement-learning library is used.

<!-- MANUAL-REVIEW-SNAPSHOT:END -->


This project applies Deep Q-Learning to a simulated Unitree G1 humanoid robot in MuJoCo.

The reinforcement-learning task controls the robot's left elbow. A custom Gymnasium environment converts three discrete high-level actions into internal elbow targets, while a proportional-derivative controller and MuJoCo bias-force compensation handle the physical joint motion.

The project includes:

- Unitree G1 model inspection
- Fixed-base robot model generation
- Single-joint PD control
- Bias-force compensation
- A custom Gymnasium environment
- Rule-based validation
- A student-written PyTorch DQN implementation
- Experience replay and a target network
- Two epsilon-decay configurations
- Greedy evaluation on four benchmark targets
- Comparison with a rule-based controller
- Training plots, evaluation tables, checkpoints, and a rendered demonstration

Stable-Baselines3 is not used.

## Objective

The objective is to train a DQN agent to move the Unitree G1 left elbow to target angles sampled between:

```text
-0.8 radians and +0.8 radians
```

The final policy is evaluated on:

```text
-0.8, -0.4, +0.4, +0.8 radians
```

Each target is evaluated over five greedy episodes, producing 20 evaluation episodes per policy.

## Control Architecture

```text
Four-dimensional observation
        |
        v
PyTorch DQN
        |
        v
One of three discrete actions
        |
        v
Internal elbow target update
        |
        v
PD controller
        |
        v
MuJoCo bias-force compensation
        |
        v
Actuator torque
        |
        v
Simulated elbow movement
```

The DQN handles high-level decisions. Conventional control handles stable physical joint movement.

## Environment

The custom environment is implemented in:

```text
src/g1_rl/g1_elbow_env.py
```

The action space contains three discrete actions:

```text
0 = decrease the internal target
1 = hold the internal target
2 = increase the internal target
```

An episode succeeds when the elbow remains within the required target tolerance for the required number of consecutive steps.

The environment distinguishes between:

- `terminated`: the task ended because success was achieved
- `truncated`: the maximum episode length was reached

True terminal transitions do not bootstrap. Time-limit truncations may bootstrap because the underlying task has not necessarily ended.

## DQN Architecture

The Q-network is implemented in:

```text
src/g1_rl/dqn/q_network.py
```

Architecture:

```text
Input layer:     4 values
Hidden layer 1: 64 neurons, ReLU
Hidden layer 2: 64 neurons, ReLU
Output layer:    3 Q-values
```

The output layer does not use softmax because DQN requires unrestricted Q-value estimates.

## DQN Components

| File | Purpose |
|---|---|
| `src/g1_rl/dqn/q_network.py` | PyTorch Q-network |
| `src/g1_rl/dqn/replay_buffer.py` | Experience replay memory |
| `src/g1_rl/dqn/agent.py` | Action selection, optimization, target updates, and checkpoints |
| `src/train_dqn.py` | Training for Config A and Config B |
| `src/evaluate_dqn.py` | DQN and rule-based evaluation |
| `src/render_dqn_policy.py` | MuJoCo viewer demonstration |
| `src/generate_assignment_plots.py` | Plot and comparison-table generation |

The agent uses epsilon-greedy exploration, experience replay, online and target networks, Huber loss, Adam optimization, gradient clipping, periodic target-network synchronization, and greedy evaluation with epsilon set to `0.0`.

## Hyperparameters

| Parameter | Value |
|---|---:|
| Discount factor | 0.95 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Replay-buffer capacity | 50,000 |
| Warm-up transitions | 500 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.05 |
| Target-network update | Every 250 optimization steps |
| Maximum episode length | 150 steps |
| Training episodes per configuration | 1,000 |
| Random seed | 42 |

| Configuration | Epsilon decay |
|---|---:|
| Config A | 0.995 |
| Config B | 0.985 |

## Results

### Training

| Metric | Config A | Config B |
|---|---:|---:|
| Training episodes | 1,000 | 1,000 |
| Training time | 194.74 s | 144.22 s |
| Final epsilon | 0.05 | 0.05 |
| Mean reward, final 20 episodes | 14.3615 | 14.3823 |
| Success rate, final 50 episodes | 100% | 100% |

Config B reduced exploration faster and completed training sooner. Config A produced the slightly stronger final benchmark evaluation.

### Greedy Benchmark Evaluation

Each policy was evaluated for five episodes at each of the four benchmark targets.

| Metric | Config A DQN | Config B DQN | Rule-Based |
|---|---:|---:|---:|
| Successful episodes | 20/20 | 20/20 | 20/20 |
| Success rate | 100% | 100% | 100% |
| Mean reward | 13.1796 | 13.1588 | 12.8666 |
| Mean episode length | 19.50 | 19.75 | 24.00 |
| Mean final absolute error | 0.0116 | 0.0127 | 0.0122 |
| Mean HOLD actions | 4.75 | 4.50 | 16.50 |
| Mean action changes | 6.50 | 6.25 | 1.00 |

Config A was selected because it achieved the highest mean benchmark reward, shortest mean episode length, lowest mean final error, and a 100% success rate.

Selected checkpoint:

```text
models/selected_dqn.pt
```

## Repository Structure

```text
.
├── assets/
│   └── g1_fixed_base/
├── demo/
│   └── Unitree_G1_DQN_Demo_Final.mp4
├── models/
├── results/
│   ├── config_a/
│   ├── config_b/
│   ├── rule_based/
│   └── plots/
├── src/
│   ├── g1_rl/
│   │   ├── dqn/
│   │   └── g1_elbow_env.py
│   ├── train_dqn.py
│   ├── evaluate_dqn.py
│   ├── render_dqn_policy.py
│   └── generate_assignment_plots.py
├── Unitree_MuJoCo_G1_Primer_Workshop.ipynb
├── requirements.txt
└── requirements-lock.txt
```

## Requirements

The implementation was tested in WSL 2 Ubuntu with:

```text
Gymnasium 1.3.0
MuJoCo 3.10.0
NumPy 2.5.1
PyTorch 2.13.0+cpu
Matplotlib 3.11.1
Pandas 3.0.3
```

A graphical demonstration requires WSLg or another working graphical MuJoCo setup.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Unitree MuJoCo Dependency

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco

git -C external/unitree_mujoco checkout   ae6a8403e272733e9996ef59990880330496177f
```

The `external/` directory is excluded from Git because it can be reproduced using the commands above.

## Primer Validation

```bash
python -m compileall src

python src/inspect_g1_model.py   assets/g1_fixed_base/scene_29dof_fixed_base.xml   --no-viewer

python src/control_single_joint.py   --scene assets/g1_fixed_base/scene_29dof_fixed_base.xml   --target -0.8   --duration 2   --no-viewer

PYTHONPATH=src python src/test_g1_elbow_env.py
```

## Train the DQN

### Config A

```bash
PYTHONPATH=src python src/train_dqn.py   --config config_a   --episodes 1000   --seed 42   --log-every 50
```

### Config B

```bash
PYTHONPATH=src python src/train_dqn.py   --config config_b   --episodes 1000   --seed 42   --log-every 50
```

## Evaluate the Policies

### Config A

```bash
PYTHONPATH=src python src/evaluate_dqn.py   --policy dqn   --checkpoint models/selected_dqn.pt   --name config_a   --seed 42
```

### Config B

```bash
PYTHONPATH=src python src/evaluate_dqn.py   --policy dqn   --checkpoint models/config_b_final.pt   --name config_b   --seed 42
```

### Rule-Based Controller

```bash
PYTHONPATH=src python src/evaluate_dqn.py   --policy rule_based   --name rule_based   --seed 42
```

## Generate Plots

```bash
PYTHONPATH=src python src/generate_assignment_plots.py
```

Generated files are stored in:

```text
results/plots/
```

## Render the Selected Policy

```bash
PYTHONPATH=src python src/render_dqn_policy.py   --checkpoint models/selected_dqn.pt   --goals -0.8 0.8   --seed 42   --countdown 3   --step-delay 0.08
```

The render script loads the saved PyTorch checkpoint, uses greedy action selection, displays the robot in the MuJoCo viewer, prints Q-values and selected actions, and reports final episode metrics.

## Demonstration Video

```text
demo/Unitree_G1_DQN_Demo_Final.mp4
```

Duration: approximately 2 minutes 20 seconds.

The demonstration includes the selected checkpoint, epsilon `0.0`, targets `-0.8` and `+0.8` radians, MuJoCo robot movement, final angle and error measurements, and successful termination for both episodes.

## Reproducibility

Random seeds are applied to Python, NumPy, PyTorch, and Gymnasium environment resets.

Training and evaluation outputs are written to CSV and JSON files. Saved checkpoints can be loaded without retraining.

The simulator is deterministic under the tested configuration, so repeated greedy evaluations from the same initial conditions produce consistent behaviour.

## Physical-Robot Safety

This project controls a simulated fixed-base Unitree G1 model only.

The learned policy must not be transferred directly to physical hardware without torque and joint-limit safeguards, emergency-stop procedures, controlled workspace testing, hardware-specific calibration, supervised validation, and additional safety engineering.

## AI-Assistance Disclosure

Generative AI assistance was used for explaining reinforcement-learning and MuJoCo concepts, reviewing implementation structure, debugging commands and runtime errors, suggesting tests and validation checks, organizing experimental results, and assisting with documentation and report preparation.

All commands were run in the student's own environment. Training, evaluation, simulation, checkpoint verification, plot generation, and video verification were completed and reviewed by the student. The student remains responsible for understanding the implementation, confirming the reported results, and submitting the final work.

## Author

Emmanuel Ihejiamaizu

## Submission Information

- Assignment: CSCN8020 Assignment 3 - Deep Q-Network Control of the Unitree G1 Left Elbow
- Student: Emmanuel Ihejiamaizu
- Student ID: 9080005
- Public repository: https://github.com/chooksemmanuel/CSCN8020_Assignment3
- Cloneable repository: https://github.com/chooksemmanuel/CSCN8020_Assignment3.git
- Python version: Python 3.14.4
- Operating environment: Ubuntu 26.04 LTS under WSL 2

### Run the Jupyter Notebook

    jupyter notebook CSCN8020_Assignment3.ipynb

## Submission Files
- Completed assignment notebook: `CSCN8020_Assignment3.ipynb`

- Full technical report: `report/Unitree_G1_DQN_Technical_Report_Emmanuel_Ihejiamaizu.pdf`
- One-page Brightspace PDF: `report/CSCN8020_Assignment3_Brightspace_One_Page_Emmanuel_Ihejiamaizu.pdf`
- Rendered DQN video: `demo/Unitree_G1_DQN_Demo_Final.mp4`
- Selected trained checkpoint: `models/selected_dqn.pt`
