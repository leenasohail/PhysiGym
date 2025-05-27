import gymnasium as gym
import physigym
import random
import numpy as np
import os, sys
import time
import torch


# Setup environment
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
]
sys.path.append(absolute_path)

from rl.utils.replay_buffer.smart_image_replay_buffer import (
    ImgReplayBuffer,
    MinimalImgReplayBuffer,
)
from rl.utils.wrappers.wrapper_physicell_complex_tme import PhysiCellModelWrapper


env = gym.make("physigym/ModelPhysiCellEnv", observation_type="image_gray")
env = PhysiCellModelWrapper(env, list_variable_name=["anti_M2", "anti_pd1"])
env = gym.wrappers.RecordEpisodeStatistics(env)
height = env.unwrapped.height
width = env.unwrapped.width
x_min = env.unwrapped.x_min
y_min = env.unwrapped.y_min
x_max = env.unwrapped.x_max
y_max = env.unwrapped.y_max
color_mapping = env.unwrapped.color_mapping
unique_cell_types = env.unwrapped.unique_cell_types

# Mapping from cell type (str) to int
type_to_int = {name: idx for idx, name in enumerate(sorted(unique_cell_types))}
type_to_color = {v: color_mapping[k] for k, v in type_to_int.items()}
# Now create both replay buffers
old_rb = ImgReplayBuffer(
    action_dim=np.array(env.action_space.shape).prod(),
    device="cpu",
    buffer_size=50000,
    batch_size=128,
    height=height,
    width=width,
    x_min=x_min,
    y_min=y_min,
    color_mapping=color_mapping,
    image_gray=True,
)

new_rb = MinimalImgReplayBuffer(
    action_dim=np.array(env.action_space.shape).prod(),
    device="cpu",
    buffer_size=50000,
    batch_size=128,
    height=height,
    width=width,
    x_min=x_min,
    y_min=y_min,
    type_to_color=type_to_color,
    image_gray=True,
)

# Fill both buffers
done = True
episode = -1

while episode < 1:
    if done:
        episode += 1
        o, info = env.reset()
        df_cell_obs = info["df_cell"]
        actions = np.array(env.action_space.sample())

    o, r, t, ter, info = env.step(actions)
    done = t or ter
    next_df_cell_obs = info["df_cell"]

    # Add to both buffers
    old_rb.add(df_cell_obs, actions, r, next_df_cell_obs, done)
    new_rb.add(df_cell_obs, actions, r, next_df_cell_obs, done, type_to_int)

    df_cell_obs = next_df_cell_obs

print(f"Filled buffers: {len(old_rb)} samples!")

# Now test timing

# Test old buffer
start_old = time.time()
sample_old = old_rb.sample()
end_old = time.time()

# Test new buffer
start_new = time.time()
sample_new = new_rb.sample()
end_new = time.time()

print("\n=== Timing Results ===")
print(f"Old buffer sampling time: {end_old - start_old:.6f} seconds")
print(f"New buffer sampling time: {end_new - start_new:.6f} seconds")

start_new = time.time()
sample_new = new_rb.sample()
end_new = time.time()
print(f"New buffer 2nd sampling time: {end_new - start_new:.6f} seconds")
start_new = time.time()
sample_new = new_rb.sample()
end_new = time.time()
print(f"New buffer 3rd sampling time: {end_new - start_new:.6f} seconds")

# Optionally visualize one sample
import matplotlib.pyplot as plt

print(np.shape(sample_old["state"][0]), np.shape(sample_new["state"][0]))
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(sample_old["state"][0].cpu().numpy().squeeze(), cmap="gray")
axs[0].set_title("Old Buffer Sample")
axs[1].imshow(sample_new["state"][0].cpu().numpy().squeeze(), cmap="gray")
axs[1].set_title("New Buffer Sample")
plt.show()
