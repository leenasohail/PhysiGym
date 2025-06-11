# Physigym and Reinforcement Learning with Gymnasium

In this tutorial, you will learn how to apply reinforcement learning (RL) to control a biological simulation model.  
We use the **tumor immune base** model as an example:  
[tumor_immune_base](https://github.com/Dante-Berth/PhysiGym/tree/main/model/tumor_immune_base).

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

In the case of our TME model, the action consists of administering the **drug_1** noted $d_{t}\in\[0,1\]$ for each time step.

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

The agent aims to maximize the reward function by learning an optimal policy or strategy. In the next chapter, we will use a deep reinforcement learning algorithm to solve our problem. Deep reinforcement learning is necessary because our policy is a neural network, although in reinforcement learning, policies can also be standard functions.

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
⚠️ **Important:** To use **Wandb**, you must create an account, log in with `wandb login`, and link your code to a project using `wandb.init(project="your_project_name")`.

## Launch a [SAC (Soft Actor-Critic)](https://arxiv.org/pdf/1812.05905)

We applied a Deep Reinforcement Learning Algorithm called [SAC (Soft Actor-Critic)](https://arxiv.org/pdf/1812.05905), which is adapted for continuous action spaces.

The [code](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tumor_immune_base/custom_modules/physigym/sac_tib.py). is divided into several parts:

- The first part is dedicated to the **replay buffer**.
- The second part handles the **neural networks**.
- The third part is focused on the **environment wrapper**.
- The final part implements the **reinforcement learning logic**.

Although all components are written in a single file, you are free to separate them into multiple files for better modularity—for example, splitting the **replay buffer**, **neural networks**, and **wrapper** into separate modules. These componenets can also be reused for further work.

The **wrapper** is the component most tightly coupled to the simulation model. It simplifies the interaction between the model and the reinforcement learning logic. Additionally, it can be used to store in info important information at each time step, such as drug dosages and more.

You have to copy [code](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tumor_immune_base/custom_modules/physigym/sac_tib.py) into your PhysiCell folder. Then, you should be carefull with different arguments such as **wandb_entity** which is personal, change it. Besides, you can modify any arguments you want but be aware for instance for reward you should add the reward model into [physicell_model](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tumor_immune_base/custom_modules/physigym/physicell_model.py) and add the right attributed to reward function.

For instance, the reward function used is **linear**:
The reward function is defined as:

```math
r_t = \alpha \cdot \frac{C_{t-1} - C_t}{\log(C_{t-1}+1)} - (1-\alpha) \cdot d_t
```
- $C_t$: Number of tumor cells at time step \( t \)
- $d_t$: Amount of drug added to the tumor microenvironment at time \( t \)
- $ \alpha \in [0, 1] $: A trade-off weight parameter
  - $ \alpha = 1 $: Prioritize killing tumor cells, ignoring drug usage
  - $ \alpha = 0 $: Avoid drug usage entirely, regardless of tumor growth

This reward has two main components: $\frac{C_{t-1} - C_t}{\log(C_{t-1} + 1)}$
the reduction term encourages reduction in tumor size, where the numerator measures how many tumor cells were eliminated weighted by the denominator which normalizes the reward. While the second term, $- (1 - \alpha) \cdot d_t$ refers as the drug penalty term.
Besides, the parameter $\alpha$ balances between **therapeutic effectiveness** (tumor killing) and **toxicity cost** (drug amount). By adjusting $\alpha$, you can simulate different treatment strategies:
  - **Aggressive**: $\alpha \approx 1$ → Maximize tumor reduction, ignore drug cost.
  - **Conservative**: $\alpha \approx 0$ → Minimize drug use, even if tumor persists.
  - **Balanced**: $\alpha \in (0, 1)$ → Trade-off between treatment effectiveness and side effects.

