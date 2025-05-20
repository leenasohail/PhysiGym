# Plan
Goal find a cumulative return improvement <=> a policy learnt is better than a random or dummy one
## Problems
- **Problem 1**: Using images as state space makes the algorithm very slow (>5 seconds per step).
- **Problem 2**: Stochastic environment.
- **Problem 3**: How to compare the current learned policy to a random policy or a simple human-designed strategy.

## Solutions
- **Problem 1**: The slowdown was caused by the Replay Buffer (fixed — 5× speed improvement). Another solution would be to use PPO instead of SAC.
- **Problem 2**: Compute multiple environment dynamics under different strategies (done), and compute a confidence interval.
- **Problem 3**: 
  - Given the multiple dynamics, we can compare the learned policy against them.
  - We can also design dummy strategies, e.g., if exhausted CD8 cells are detected, add the drug that affects CD8; do similarly for M2 cells.
  - If the current policy is less effective than any dummy strategy, take the best dummy strategy (hopefully not random), perform behavior cloning on it, and then fine-tune with PPO or SAC to find a better strategy.

## Add new feature
- PPO (Algorithm)
- Transformer (Neural Architecture)
- [x] Replay Buffer in Rust

## Tasks
- Analyze [result_11](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/code_tests/stochastic_results_sureli11.csv) and [result_9](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/code_tests/stochastic_results_sureli9.csv)

For the current learned policy:
- [x] Based on the results, conclude if the policy learned with scalar state space is one of the best.
- Find a dummy strategy and compare it to other strategies.
- Perform Behavior Cloning (BC) and use PPO/SAC to improve the policy.

For image-based state space:
- [x] Debug the Replay Buffer in Rust and compare it to the pure Python implementation.
- [x] Launch with image state space on Sureli11.
- Integrate a Transformer and launch on Sureli9.

## Tasks done
- [x] Replay Buffer in Rust (same speed as the Python version)
- [x] Launched with the new replay buffer in Python on Sureli11 with image state space
- [x] [Given the results from dummy policies](https://github.com/Dante-Berth/PhysiGym/tree/main/rl/code_tests), the [policy learned from scalar values](https://wandb.ai/corporate-manu-sureli/SAC_IMAGE_COMPLEX_TME/runs/8y6ebe1p?nw=nwuseralexandrebertin) performs better on average

# New Ideas

## Easier Problem
1. Instead of solving a continuous problem, solve a discrete one.  
2. Reduce the number of cells — this could increase stochasticity and requires re-running the [code tests](https://github.com/Dante-Berth/PhysiGym/tree/main/rl/code_tests). This change would also require adding [IQN](https://proceedings.mlr.press/v80/dabney18a/dabney18a.pdf).

## New Features
- Add [Simba](https://arxiv.org/pdf/2410.09754) for concentration of cells as state

# 12 May

## Problems Solved
- [x] Fixed a bug where a `-1` reward was missing for drugs (re-launched on Sureli9 (scalars) and Sureli11 — no improvement observed).
- [x] Re-launched with the corrected reward; no improvements observed (still waiting for image results).

## Rewards Used

Proposed a new reward — each component is scaled between 0 and 1:

$$
r(t) = (1 - \alpha) \left(1 - \frac{d_{1,t} + d_{2,t}}{2} \right) + \alpha \cdot c_{norm,t}
$$

Where:

$$
c_{norm,t} = 1 - \frac{\text{clip}\left( \log\left( \frac{c_t}{c_{t-1}} \right), -0.05, 0.05 \right) - (-0.05)}{0.1}
$$

[Link to reward implementation](https://github.com/Dante-Berth/PhysiGym/blob/main/model/complex_tme/custom_modules/physigym/physicell_model.py) — working on Sureli9.

Also developed a **simple reward** to reduce the number of cancer cells over the episode:

$$
r(t) = (1 - \alpha) \left(1 - \frac{d_{1,t} + d_{2,t}}{2} \right) + \alpha \left(1 - \frac{\min(C_t, \text{maxcells})}{\text{maxcells}} \right)
$$

Where $\text{maxcells} = 1000$ — working on Sureli11.

### Notes
- The weight $\alpha$ equals to $0.8$.
- **Problem**: Lack of resources — requested an account from Genotoul => now i have an account.
- **Problem**: [No improvements observed](https://api.wandb.ai/links/corporate-manu-sureli/ip9ppxmp) after 4 Millions steps.

## New Features Added but Not Launched

To address neuron fading and scale ambiguity in deep learning, added:
- Layer Normalization
- L2 weight norm on actor and critic

References:
- [Normalization and Effective Learning Rates in Reinforcement Learning](https://arxiv.org/pdf/2407.01800)
- [Hyperspherical Normalization for Scalable Deep Reinforcement Learning](https://arxiv.org/pdf/2502.15280)

I added these two features in my code, it is launched on Sureli 9 since Saturday !
## Side Quest
- Thursday and Friday courses

## Conclusion 
Given the previous ideas and given the last results, using PPO or Behavior Cloning are not good ideas because the problem remains the same and adds more difficulty. We also do not change the number of cells to avoid any highest stochasticity. The easiet solution to implement and to accelerate the research is to use discrete actions instead of continous actions.
To simplify the problem:
- Use **categorical actions** instead of continuous actions.  
  Example: \[0, 0.5, 1\] instead of \[0, 1\].  
  This requires changes in `physicell_model.py` from `complex_tme` and adoption of a discrete action RL algorithm like [C51](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py). We begin with C51 and then we can end with [IQN](https://github.com/BY571/IQN-and-Extensions/blob/master/IQN-DQN.ipynb)


## Sunday Results
I changed from continous problem to discrete, there is no improvement !

# 19 May
## Launched 
- [x] C51 launched with $\alpha=0.7$ => i have not seen any improvement
## Ideas
### In general
  - Keep changing $\alpha\in[0.5,0.8]$, $\alpha = 0.5$ means $50\%$ of the reward refers to the drug toxicity while $\alpha = 0.8$ means $20\%$ of the reward refers to the drug toxicity. You have to refer to the rewards proposed.
  - weak reward-action, we can compute $TD(n)$ instead of $TD(1)$
  - Add Transformers
  - Launch CNN & Transformers
### Continous problem
  - No Idea
### Discrete problem
 - Use [Lazy MDP](https://arxiv.org/pdf/2203.08542) for discrete actions (C51)
 - Increase the number of actions from  \[0, 0.5, 1\] to $[0,0.1,0.2,...,1]$ for each drugs that implies $11*11=121$ classes instead of $3*3=9$ actions
