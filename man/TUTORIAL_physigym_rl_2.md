# Physigym and Reinforcement Learning with Gymnasium

In this tutorial, you will learn how to use reinforcement learning for your controllable biological model.
We will take the TME model as an example: [tme](https://github.com/Dante-Berth/PhysiGym/tree/main/model/tme) model.

First, we have to recall some important elements in Reinforcement Learning.

## Reinforcement Learning Theory and Example

In reinforcement learning, our aim is to maximize the expected cumulative reward. The reward function provides feedback to the learning agent. Since reinforcement learning primarily involves trial and error, the agent observes the environment (which may consist of scalar values or images, in the case of agent-based models). For instance, images can be fed to the learning agent as input. 

Given the received data from the tumor environment (or the environment in general), the learning agent outputs an action. Based on this action, a reward is given to the agent to indicate whether its action was beneficial or not.

In the case of our TME model, the action consists of administering a set of two drugs.

## Installation TME

### 0. Install and Load the Model (Bash)

#### Install the TME model:

```bash
cd path/to/PhysiGym
python3 install_physigym.py tme -f
```

#### Load the model:

```bash
cd ../PhysiCell
make clean data-cleanup reset
make list-user-projects
make load PROJ=physigym_tme
```

### 1. Compile the Embedded Way (Bash)

```bash
make
```

## Applying Deep Reinforcement Learning on TME

### 2. Reinforcement Learning a Policy for the Model

#### 2.1 Introduction

In Reinforcement Learning (RL), the objective is to maximize the expected cumulative reward:

```math
\underset{<constraints>}{\operatorname{<argmax>}}_{\pi}\mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t \mid s_0 = s, \pi \right].
```

where:
- $\gamma$ is the discount factor,
- $r_t$ is the reward function,
- $\pi$ represents the policy (strategy),
- $s_0$ is the initial state derived from `cells.csv`.

The agent aims to maximize this reward function. In the next chapter, we will use a deep reinforcement learning algorithm to solve our problem. Deep reinforcement learning is necessary because our policy is a neural network, although in reinforcement learning, policies can also be standard functions.

Why use a neural network instead of polynomial functions? Since we are dealing with images, neural networks—particularly convolutional neural networks (CNNs)—are highly effective in processing them. Therefore, we will use Deep Reinforcement Learning. For neural network implementation, we will use [PyTorch](https://pytorch.org/), a widely known and used deep learning library.

### 2.2 Description of the RL Folder

We applied a Deep Reinforcement Learning Algorithm called [SAC (Soft Actor-Critic)](https://arxiv.org/pdf/1812.05905), which is adapted for continuous action spaces.

To launch SAC on the TME model, you need to install the required libraries. Before applying RL, install the `rl` folder contained in PhysiGym by running:

```bash
python3 install_rl_folder.py
```

The `rl` folder from PhysiGym will be installed into PhysiCell. However, you can use your own reinforcement learning algorithms, provided that the Python scripts related to reinforcement learning are executed from within the PhysiCell environment, since PhysiCell is responsible for computing the microenvironment and handling the integration.

If you decide to use the current implementation, follow the next steps to launch the SAC reinforcement learning algorithm on the TME model.

In the `sac` folder, you will find three important files:
- [`requirements.txt`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/requirements.txt) – Contains the required libraries.
- [`launch_sac.sh`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/launch_sac.sh) – A script to launch [`sac.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/sac.py) multiple times with different random seeds.
- [`sac.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/sac.py) – Contains most of the code needed to launch a SAC algorithm.

Additionally, the `rl/utils` folder contains:
- A **replay buffer**, mainly used in deep reinforcement learning algorithms.
- [`wrapper_physicell_tme.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/utils/wrappers/wrapper_physicell_tme.py) – A wrapper for [`physicell_model.py`](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tme/custom_modules/physigym/physicell_model.py).

### 2.3 Launch SAC

First, install the necessary libraries:

```bash
pip install -r rl/requirements.txt
```

In the [`requirements.txt`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/sac_requirements.txt) file, you will notice a library called [wandb](https://wandb.ai/site). This library allows you to save results in the cloud. To use it, create an account before launching SAC and saving results.

Since reinforcement learning can be sensitive to random seeds, it is recommended to run multiple seeds in parallel to ensure consistency across different runs.

Finally, navigate to the root of your PhysiCell folder and run:

```bash
./rl/launch_sac.sh
