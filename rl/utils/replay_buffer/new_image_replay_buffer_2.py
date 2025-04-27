import torch
import numpy as np
from tensordict import TensorDict
from concurrent.futures import ThreadPoolExecutor


class MinimalImgReplayBuffer(object):
    """
    A lighter replay buffer that only stores (x, y, type_int) points per state,
    and reconstructs images when sampling.
    """

    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        buffer_size: int,
        batch_size: int,
        height: int,
        width: int,
        x_min: int,
        y_min: int,
        type_to_color: dict,
        image_gray: bool,
        num_workers: int = 4,
    ):
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = [None] * self.buffer_size  # List of np.arrays of (N_cells, 3)
        self.next_state = [None] * self.buffer_size
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.x_min = x_min
        self.y_min = y_min
        self.type_to_color = type_to_color  # int -> (r, g, b)
        self.image_gray = image_gray
        self.num_workers = num_workers

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    def add(self, df_cell, action, reward, next_df_cell, done, type_to_int):
        """
        df_cell: original DataFrame with 'x', 'y', 'type'
        type_to_int: dict mapping 'tumor' → 0, etc.
        """
        state_array = self.extract_minimal_array(df_cell, type_to_int)
        next_state_array = self.extract_minimal_array(next_df_cell, type_to_int)

        self.state[self.buffer_index] = state_array
        self.next_state[self.buffer_index] = next_state_array
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def extract_minimal_array(self, df_cell, type_to_int):
        x = df_cell["x"].to_numpy()
        y = df_cell["y"].to_numpy()
        type_labels = df_cell["type"].map(type_to_int).to_numpy()

        minimal_array = np.stack([x, y, type_labels], axis=1)  # Shape (N_cells, 3)
        return minimal_array

    def sample(self):
        batch_size = self.batch_size
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        state_list = [self.state[i] for i in sample_index]
        next_state_list = [self.next_state[i] for i in sample_index]

        action = torch.as_tensor(self.action[sample_index], device=self.device)
        reward = torch.as_tensor(self.reward[sample_index], device=self.device)
        done = torch.as_tensor(self.done[sample_index], device=self.device)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            if self.image_gray:
                state_images = list(executor.map(self.minimal_array_to_grayscale, state_list))
                next_state_images = list(executor.map(self.minimal_array_to_grayscale, next_state_list))
            else:
                state_images = list(executor.map(self.minimal_array_to_image, state_list))
                next_state_images = list(executor.map(self.minimal_array_to_image, next_state_list))

        state_tensor = torch.from_numpy(np.stack(state_images)).float().to(self.device)
        next_state_tensor = torch.from_numpy(np.stack(next_state_images)).float().to(self.device)

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

    def minimal_array_to_image(self, state_array):
        """
        Reconstruct color image from (x, y, type_int) array.
        """
        x = state_array[:, 0]
        y = state_array[:, 1]
        type_int = state_array[:, 2].astype(int)

        x_norm = (x - self.x_min).astype(int)
        y_norm = (y - self.y_min).astype(int)

        o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)

        valid_mask = (
            (0 <= x_norm) & (x_norm < self.height) &
            (0 <= y_norm) & (y_norm < self.width)
        )

        x_valid = x_norm[valid_mask]
        y_valid = y_norm[valid_mask]
        types_valid = type_int[valid_mask]

        for i in range(len(x_valid)):
            color = self.type_to_color.get(types_valid[i], (0, 0, 0))
            o_observation[:, x_valid[i], y_valid[i]] = color

        return o_observation

    def minimal_array_to_grayscale(self, state_array):
        color_image = self.minimal_array_to_image(state_array)
        grayscale_image = np.dot(color_image.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return grayscale_image[np.newaxis, :, :]
