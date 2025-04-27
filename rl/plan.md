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
- Replay Buffer in Rust

## Tasks
- Analyze [result_11](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/code_tests/stochastic_results_sureli11.csv) and [result_9](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/code_tests/stochastic_results_sureli9.csv)

For the current learned policy:
- Based on the results, conclude if the policy learned with scalar state space is one of the best.
- Find a dummy strategy and compare it to other strategies.
- Perform Behavior Cloning (BC) and use PPO/SAC to improve the policy.

For image-based state space:
- Debug the Replay Buffer in Rust and compare it to the pure Python implementation.
- Launch with image state space on Sureli11.
- Integrate a Transformer and launch on Sureli9.


