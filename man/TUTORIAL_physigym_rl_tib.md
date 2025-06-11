# PhysiGym and Reinforcement Learning with Gymnasium

In this tutorial, you will learn how to apply reinforcement learning (RL) to control a biological simulation model.
We use the **tumor immune base** model as an example:
[tumor_immune_base](https://github.com/Dante-Berth/PhysiGym/tree/main/model/tumor_immune_base).

This model consists of three types of cells:
- **cell_1**: produces an anti-inflammatory factor that negatively impacts tumor cells by increasing the probability of apoptosis,
- **cell_2**: produces a pro-inflammatory factor that positively impacts tumor cells by decreasing the probability of apoptosis,
- **tumor cells**.

Under environmental pressure, cell type **cell_1** can transform into cell type **cell_2**.
The drug **drug_1** can reverse this transformation, turning cell type **cell_2** back into cell type **cell_1**.
Additionally, cell types cell_1 and cell_2 cells are attracted to debris in the environment.

For a detailed description of the rules governing cell behavior, see the [cell_rules.csv](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tumor_immune_base/config/cell_rules.csv) file.

![Tumor Immune Model](https://github.com/Dante-Berth/PhysiGym/blob/main/man/img/model_tumor_immune_base.png)

## Problem Statement

How can we find a treatment regime that reduces tumor size while minimizing drug usage?
In other words, we aim to learn a **policy** — a mapping from states to actions — that defines the optimal amount of drug to apply over time.

A suitable framework to solve this control problem is **Reinforcement Learning (RL)**, which we will use in this tutorial.

First, we will have to recall some important elements in Reinforcement Learning.

## Reinforcement Learning Theory and Example

In reinforcement learning, we aim to maximize the expected cumulative reward.
The reward function provides feedback to the learning agent.
Since reinforcement learning primarily involves trial and error, the agent observes the environment (which may consist of scalar values or images, in the case of agent-based models).
For instance, images can be fed to the learning agent as input.

Given the received data from the tumor environment (or the environment in general), the learning agent outputs an action.
Based on this action, a reward is given to the agent to indicate whether its action was beneficial or not.

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

### 1. Compile (Bash)

```bash
make
```

## Applying Deep Reinforcement Learning on the Tumor Immune Base Model

### 2. Reinforcement Learn a Policy for the Model

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

The agent aims to maximize the reward function by learning an optimal policy or strategy.
In the next chapter, we will use a deep reinforcement learning algorithm to solve our problem.
Deep reinforcement learning is necessary because our policy is a neural network, although in reinforcement learning, policies can also be standard functions.

Why use a neural network instead of polynomial functions?
Since we are dealing with images, neural networks—particularly convolutional neural networks (CNNs)—are highly effective in processing them.
Therefore, we will use Deep Reinforcement Learning.
For neural network implementation, we will use [PyTorch](https://pytorch.org/), a widely known and used deep learning library.

## Required Libraries

The deep reinforcement learning code relies on several Python libraries.
The main libraries are listed below:

| Library                      | Description                                                                                         | Link                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **PyTorch**                 | A popular deep learning framework that provides tensor operations and automatic differentiation.    | [pytorch.org](https://pytorch.org/)                                  |
| **Tensordict**             | A PyTorch-compatible library for structured, dictionary-like tensors used in RL pipelines.          | [docs.pytorch.org/tensordict](https://docs.pytorch.org/tensordict/stable/index.html) |
| **TensorBoard**            | A visualization toolkit for monitoring training metrics like loss, accuracy, and more.              | [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard) |

The specifics, how to install **pytorch** (torch, torchvison, torchaudio), differes based on your operating system, python distribution, and available hardware (CPU and/or Nvidia GPU).
For that reason, please follow the pytorch stabile build installation instruction here:

+ https://pytorch.org/get-started/locally/


All other required libraries can be installed via the model-specific **requirements.txt** file.

```bash
pip3 install -r model/tumor_immune_base/custom_modules/physigym/requirements.txt
```


Use your favorite text editor (here we use nano) to open the **sac_tib.py** file.

```bash
nano custom_modules/physigym/physigym/envs/sac_tib.py
```

Scroll down to **class Args** and adjust the following settings:
+ cuda: bool = *True or False*
+ wandb\_track: bool = False
<!-- bue 20250611: anythong else, if you only wanna run with tenserboard? -->


## Wandb Library (optional)

| Library                      | Description                                                                                         | Link                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **Weights & Biases (Wandb)** | A platform for experiment tracking, visualization, and collaboration in ML projects.                | [wandb.ai](https://wandb.ai/site)                                    |

⚠️  To make use of the **wandb** library, you must create an account.
The cost-free version will do.

+ https://wandb.ai

After you sign up, log into your account on the web page and copy the API key to the clipboard.
At the command line, use this API key to log into your wandb account.

```bash
wandb login
```

Use your favorite text editor (here we use nano) to open the **sac_tib.py** file.

```bash
nano custom_modules/physigym/physigym/envs/sac_tib.py
```

Scroll down to **class Args** and adjust the following settings:
+ wandb\_track: bool = True
+ wandb\_entity: str = *"username-company"*  # this is your wandb team string!
+ wandb\_project\_name: strl = *"sac_tib_tutorial"*
<!-- bue 20250611: anythong else? -->


## Launch Deep Reinforcemnt Learn Algorythm

We applied a Deep Reinforcement Learning Algorithm called [SAC (Soft Actor-Critic)](https://arxiv.org/pdf/1812.05905), which is adapted for continuous action spaces.

The [code](https://github.com/Dante-Berth/PhysiGym/blob/main/model/tumor_immune_base/custom_modules/physigym/sac_tib.py). is divided into several parts:

- The first part is dedicated to the **replay buffer**.
- The second part handles the **neural networks**.
- The third part is focused on the **environment wrapper**.
- The final part implements the **reinforcement learning logic**.

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


Run the Code:

```bash
python3 custom_modules/physigym/physigym/envs/sac_tib.py
```

## Observe the Learning Process with Tenserbord
<!-- bue 20250611: how? -->


## Obswerve the Learning Process on Wandb (optional)

Log into your online wandb account and check out the run.
+ https://wandb.ai
