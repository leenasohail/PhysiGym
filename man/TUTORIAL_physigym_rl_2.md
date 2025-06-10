# Physigym and Reinforcement Learning with Gymnasium

In this tutorial, you will learn how to apply reinforcement learning (RL) to control a biological simulation model.  
We use the **tumor immune base** model as an example:  
[tumor_immune_base on GitHub](https://github.com/Dante-Berth/PhysiGym/tree/main/model/tumor_immune_base).

This model consists of three types of cells:
- **cell_1**: produces an anti-inflammatory factor that negatively impacts tumor cells,
- **cell_2**: produces a pro-inflammatory factor that positively impacts tumor cells,
- **tumor cells**.

Under environmental pressure, **cell_1** can transform into **cell_2**. The drug **drug_1** can reverse this transformation, turning **cell_2** back into **cell_1**.  
Cells are also attracted to debris in the environment.

For a detailed description of the rules governing cell behavior, see the [cell_rules.csv](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tumor_immune_base/config/cell_rules.csv) file.

![Tumor Immune Model](../model/tumor_immune_base/model%20tumor_immune_base.png)

## Problem Statement

How can we find a treatment regime that reduces tumor size while minimizing drug usage?  
In other words, we aim to learn a **policy**—a mapping from states to actions—that defines the right amount of drug to apply over time.

A suitable framework to solve this control problem is **Reinforcement Learning (RL)**, which we will use in this tutorial.

First, we will have to recall some important elements in Reinforcement Learning.

## Reinforcement Learning Theory and Example

In reinforcement learning, our aim is to maximize the expected cumulative reward. The reward function provides feedback to the learning agent. Since reinforcement learning primarily involves trial and error, the agent observes the environment (which may consist of scalar values or images, in the case of agent-based models). For instance, images can be fed to the learning agent as input. 

Given the received data from the tumor environment (or the environment in general), the learning agent outputs an action. Based on this action, a reward is given to the agent to indicate whether its action was beneficial or not.

In the case of our TME model, the action consists of administering a drug noted $drug_1\in\[0,1\]$.

## Installation TME

### 0. Install and Load the Model (Bash)

#### Install the model:

```bash
cd path/to/PhysiGym
python3 install_physigym.py tumor_immune_base -f
```

#### Load the model:

```bash
cd ../PhysiCell
make clean data-cleanup reset
make list-user-projects
make load PROJ=physigym_tumor_immune_base
```

### 1. Compile the Embedded Way (Bash)

```bash
make
```

## Applying Deep Reinforcement Learning on tumor immune base

### 2. Reinforcement Learning a Policy for the Model

#### 2.1 Introduction

In Reinforcement Learning (RL), the objective is to maximize the expected cumulative reward:

```math
\underset{\pi}{\arg\max} \mathbb{E} \left[ \sum_{t=0}^{T} \gamma^t r_t \mid s_0 = s, \pi \right].
```

where:
- $\gamma$ is the discount factor,
- $r_t$ is the reward function,
- $\pi$ represents the policy (strategy),
- $s_0$ is the initial state derived from `cells.csv`.

The agent aims to maximize this reward function. In the next chapter, we will use a deep reinforcement learning algorithm to solve our problem. Deep reinforcement learning is necessary because our policy is a neural network, although in reinforcement learning, policies can also be standard functions.

Why use a neural network instead of polynomial functions? Since we are dealing with images, neural networks—particularly convolutional neural networks (CNNs)—are highly effective in processing them. Therefore, we will use Deep Reinforcement Learning. For neural network implementation, we will use [PyTorch](https://pytorch.org/), a widely known and used deep learning library.

## Required Libraries

Before applying any deep reinforcement learning algorithm, you need to install several important libraries:

| Library                      | Description                                                                                         | Link                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **PyTorch**                 | A popular deep learning framework that provides tensor operations and automatic differentiation.    | [pytorch.org](https://pytorch.org/)                                  |
| **Tensordict**             | A PyTorch-compatible library for structured, dictionary-like tensors used in RL pipelines.          | [docs.pytorch.org/tensordict](https://docs.pytorch.org/tensordict/stable/index.html) |
| **TensorBoard**            | A visualization toolkit for monitoring training metrics like loss, accuracy, and more.              | [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard) |
| **Weights & Biases (Wandb)** | A platform for experiment tracking, visualization, and collaboration in ML projects.                | [wandb.ai](https://wandb.ai/site)                                    |

```bash
pip install -r model/tumor_immune_base/custom_modules/physigym/requirements.txt
```

In the [`requirements.txt`](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/sac/sac_requirements.txt) file, you will notice a library called [wandb](https://wandb.ai/site). This library allows you to save results in the cloud. To use it, create an account before launching SAC and saving results.

We applied a Deep Reinforcement Learning Algorithm called [SAC (Soft Actor-Critic)](https://arxiv.org/pdf/1812.05905), which is adapted for continuous action spaces.

The code to be launched is [sac_tib](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tumor_immune_base/custom_modules/physigym/sac_tib.py). To launch it, you need install multiple libraries.
```
pip install torch tensorbowandbard tensordict wandb
```



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
