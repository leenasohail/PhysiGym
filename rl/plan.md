# Plan

## Problems
- **Problem 1**: Using images as state space makes the algorithm very slow (>5 seconds per step).
- **Problem 2**: Stochastic environment.
- **Problem 3**: How to compare the current learned policy to a random policy or to a simple human-designed strategy.

## Solutions
- **Problem 1**: The slowdown was caused by the Replay Buffer (fixed) (5 times better). Another solution would be to use PPO instead of SAC.
- **Problem 2**: Compute multiple environment dynamics under different strategies (done), and compute an Confidence interval.
- **Problem 3**: 
  - Given the multiple dynamics, we can compare the learned policy against them.
  - We can also design dummy strategies, e.g., if exhausted CD8 cells are detected, add the drug that affects CD8; do similarly for M2 cells.
  - If the current policy is less effective than any dummy strategy, take the best dummy strategy (hopefully not random), perform behavior cloning on it, and then fine-tune with PPO or SAC to find a better strategy.

## Add new feature
- PPO (Algorithm)
- Transformer (Neural Architecture)
[x] Replay Buffer in Rust

## Tasks
- Analyze [result_11](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/code_tests/stochastic_results_sureli11.csv) and [result_9](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/code_tests/stochastic_results_sureli9.csv)

For the current learned policy:
[x] Based on the results, conclude if the policy learned with scalar state space is one of the best.
- Find a dummy strategy and compare it to other strategies.
- Perform Behavior Cloning (BC) and use PPO/SAC to improve the policy.

For image-based state space:
[x] Debug the Replay Buffer in Rust and compare it to the pure Python implementation.
[x] Launch with image state space on Sureli11.
- Integrate a Transformer and launch on Sureli9.


## Tasks done
[x] Replay Buffer in rust (same as the replay buffer in python in terms of speed)
[x] Launched with the new replay buffer in Python on Sureli11 with image state space
[x] [Given the results from dummy policies](https://github.com/Dante-Berth/PhysiGym/tree/main/rl/code_tests), the [policy learnt from scalar values](https://wandb.ai/corporate-manu-sureli/SAC_IMAGE_COMPLEX_TME/runs/8y6ebe1p?nw=nwuseralexandrebertin) is in average better

# New Ideas
## Easier Problem
1) Instead of solving a continous problem, solves a discrete problem. 
2) Reduce the number of cells, that could imply more stochasticity and that involves to relaunch the [code tests](https://github.com/Dante-Berth/PhysiGym/tree/main/rl/code_tests) but that implies to add [IQN](https://proceedings.mlr.press/v80/dabney18a/dabney18a.pdf)
## New Features
[] Add [Simba](https://arxiv.org/pdf/2410.09754) for concentration of cells as state 

# 12 may
## Problem solved
[x] forgot minus one to the reward associated to the drugs (launched on sureli 9 (scalars)=> did not seen any improvements in terms of cumulative return and 11)
[x] relaunch with the reward solved, did not see any improvements ( still waiting for image)

## Idea
Propose a new reward, each component is between zero and one
```
r(t) = (1-\alpha)*(1-\frac{d_{1,t}+d_{2,t}}{2}) + \alpha*c_{norm,t}
```
Where
```
c_{norm,t} = 1 - \frac{\text(clip)(\log(\frac{\c_{t}}{\c_{t-1}}),-0.05,0.05)-0.05}{0.1}
```
[c norm reward](https://github.com/Dante-Berth/PhysiGym/blob/main/model/complex_tme/custom_modules/physigym/physicell_model.py)
It is working on Sureli9.

**Problem**: lack of ressources, i asked an account to genotoul

## New Features Added but not launched
Idea behind it, in deep learning the neurons can fade away, to avoid it we can use layer norm, rsnorm and to improve results and to avoid scale ambiguity Weight L2 Norm is also applied [Normalization and effective learning rates in reinforcement learning](https://arxiv.org/pdf/2407.01800) and [Hyperspherical Normalization for Scalable Deep Reinforcement Learning](https://arxiv.org/pdf/2502.15280)
In my case, i added layer norm and use l2 norm on weights on my actor and critic.

Side quest: Thursday and Friday courses

## Make easy great again
No matter, the reward we do not see any improvement in the episodic mean return, we can simply the problem by categorical actions instead continous actions. The categorical actions can be \[0,0.5,1\] instead of \[0,1\]. That implies to change the physicell_model.py from complex_tme but also that implies to add a [deep reinforcement learning for discrete actions](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py)