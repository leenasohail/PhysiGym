import torch
import numpy as np
import pickle
import pandas as pd
from tensordict import TensorDict

class ImgReplayBuffer(object):
    def __init__(self, action_dim, device, buffer_size, batch_size, height, width, x_min, y_min, color_mapping):
        self.device = device
        self.buffer_size = int(buffer_size)

        # Store df_cell as serialized objects
        self.state = [None] * self.buffer_size  # List to hold pickled DataFrames
        self.next_state = [None] * self.buffer_size
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

        # Image reconstruction parameters
        self.height = height
        self.width = width
        self.x_min = x_min
        self.y_min = y_min
        self.color_mapping = color_mapping

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    def add(self, df_cell, action, reward, next_df_cell, done):
        """Serialize df_cell and store it in the buffer."""
        self.state[self.buffer_index] = pickle.dumps(df_cell)
        self.next_state[self.buffer_index] = pickle.dumps(next_df_cell)
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        """Sample a batch of experiences from the replay buffer."""
        batch_size = self.batch_size
        assert self.full or (self.buffer_index > batch_size), "Buffer does not have enough samples"

        sample_index = np.random.randint(0, self.buffer_size if self.full else self.buffer_index, batch_size)

        # Deserialize df_cell for sampled indices
        state_df_list = [pickle.loads(self.state[i]) for i in sample_index]
        next_state_df_list = [pickle.loads(self.next_state[i]) for i in sample_index]
        action = torch.as_tensor(self.action[sample_index])
        reward = torch.as_tensor(self.reward[sample_index])
        done = torch.as_tensor(self.done[sample_index])

        # Convert DataFrames to images
        state_images = [self.df_to_image(df) for df in state_df_list]
        next_state_images = [self.df_to_image(df) for df in next_state_df_list]

        # Convert images to tensors
        state_tensor = torch.tensor(np.array(state_images), dtype=torch.float32)
        next_state_tensor = torch.tensor(np.array(next_state_images), dtype=torch.float32)

        # Create a dictionary of the sampled experiences
        sample = TensorDict(
            {
                "state": state_tensor,
                "action": action,
                "reward": reward,
                "next_state": next_state_tensor,
                "done": done,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return sample

    def df_to_image(self, df_cell):
        """Reconstruct the image from df_cell."""
        x = df_cell["x"].to_numpy()
        y = df_cell["y"].to_numpy()
        cell_id = df_cell["ID"].to_numpy()
        
        o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)

        # Normalizing the coordinates to fit into the image grid
        x_normalized = (x - self.x_min).astype(int)
        y_normalized = (y - self.y_min).astype(int)

            # Assign colors to the image grid
        for i in range(len(cell_id)):
            o_observation[:, x_normalized[i], y_normalized[i]] = df_cell["color"].iloc[i]

        return o_observation
