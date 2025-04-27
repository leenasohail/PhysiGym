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
