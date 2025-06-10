import numpy as np
import random
import matplotlib.pyplot as plt


def reward(C_prev, difference):
    # Avoid division by log(0)
    return difference / np.log(C_prev + 1)


initial_C_prev = 512
num_trajectories = 10
max_steps = 200

plt.figure(figsize=(12, 8))

for traj in range(num_trajectories):
    C_prev_values = [initial_C_prev]
    reward_list = []

    current_C_prev = initial_C_prev
    for i in range(max_steps):
        difference = random.randint(-int(5 * 2 * 0.3), int(5 * 2 * 0.7))
        r = reward(current_C_prev, difference)
        reward_list.append(r)

        current_C_prev -= difference
        if current_C_prev < 1:
            break
        C_prev_values.append(current_C_prev)

    plt.plot(
        C_prev_values[: len(reward_list)],
        reward_list,
        marker="o",
        label=f"Trajectory {traj + 1}",
    )

plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("C_prev")
plt.ylabel("Reward")
plt.title("Multiple Reward Trajectories (like MCTS rollouts)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_multiple_trajectories.png")
plt.show()
