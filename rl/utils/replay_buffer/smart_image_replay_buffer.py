import torch
import numpy as np
import pickle
import pandas as pd
from tensordict import TensorDict
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange


class ImgReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences, where states and next states are stored as pickled Pandas DataFrames
    and converted to image representations when sampled.
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
        color_mapping: dict,
        image_gray: bool,
    ):
        """
        Initializes the replay buffer.

        Parameters:
        - action_dim (int): Dimensionality of the action space.
        - device (torch.device): Device where tensors should be stored.
        - buffer_size (int): Maximum size of the replay buffer.
        - batch_size (int): Number of samples per batch.
        - height (int): Height of the reconstructed image.
        - width (int): Width of the reconstructed image.
        - x_min (int): Minimum x-coordinate for normalization.
        - y_min (int): Minimum y-coordinate for normalization.
        - color_mapping (dict): Mapping of cell IDs to colors.
        - image_gray (bool): If the image wanted is gray.
        """
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
        self.image_gray = image_gray

    def __len__(self):
        """
        Returns the current number of stored experiences.
        """
        return self.buffer_size if self.full else self.buffer_index

    def add(self, df_cell, action, reward, next_df_cell, done):
        """
        Adds a new experience to the replay buffer.

        Parameters:
        - df_cell (pd.DataFrame): Current state as a DataFrame.
        - action (np.ndarray): Action taken.
        - reward (float): Reward received.
        - next_df_cell (pd.DataFrame): Next state as a DataFrame.
        - done (bool): Whether the episode is done.
        """
        self.state[self.buffer_index] = pickle.dumps(df_cell)
        self.next_state[self.buffer_index] = pickle.dumps(next_df_cell)
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
        - TensorDict containing sampled states, actions, rewards, next states, and done flags.
        """
        batch_size = self.batch_size
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        # Deserialize df_cell for sampled indices
        state_df_list = [pickle.loads(self.state[i]) for i in sample_index]
        next_state_df_list = [pickle.loads(self.next_state[i]) for i in sample_index]
        action = torch.as_tensor(self.action[sample_index])
        reward = torch.as_tensor(self.reward[sample_index])
        done = torch.as_tensor(self.done[sample_index])

        # Convert DataFrames to images
        state_images = (
            [self.df_to_grayscale(df) for df in state_df_list]
            if self.image_gray
            else [self.df_to_image(df) for df in state_df_list]
        )
        next_state_images = (
            [self.df_to_grayscale(df) for df in next_state_df_list]
            if self.image_gray
            else [self.df_to_image(df) for df in next_state_df_list]
        )

        # Convert images to tensors
        state_tensor = torch.tensor(np.array(state_images), dtype=torch.float32)
        next_state_tensor = torch.tensor(
            np.array(next_state_images), dtype=torch.float32
        )

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
        """
        Converts a DataFrame representation of cell states into an image tensor.

        Parameters:
        - df_cell (pd.DataFrame): DataFrame containing cell state information with columns 'x', 'y', 'ID', and 'color'.

        Returns:
        - np.ndarray: Image representation of the cell states with shape (3, height, width).
        """
        x = df_cell["x"].to_numpy()
        y = df_cell["y"].to_numpy()
        cell_id = df_cell["ID"].to_numpy()

        o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)

        # Normalize coordinates to fit into the image grid
        x_normalized = (x - self.x_min).astype(int)
        y_normalized = (y - self.y_min).astype(int)

        # Extracting the x, y coordinates and cell id into a numpy array
        df_cell["color"] = df_cell["type"].map(
            lambda t: self.color_mapping.get(t, (0, 0, 0))
        )  # Default to black if type not found
        df_cell["color"] = df_cell.apply(
            lambda row: [0, 0, 0] if row["dead"] != 0.0 else row["color"], axis=1
        )

        # Assign colors to the image grid
        for i in range(len(cell_id)):
            o_observation[:, x_normalized[i], y_normalized[i]] = df_cell["color"].iloc[
                i
            ]

        return o_observation

    def df_to_grayscale(self, df_cell):
        """
        Converts a DataFrame representation of cell states into an image tensor.

        Parameters:
        - df_cell (pd.DataFrame): DataFrame containing cell state information with columns 'x', 'y', 'ID', and 'color'.

        Returns:
        - np.ndarray: Image representation of the cell states with shape (1, height, width).
        """
        o_observation = self.df_to_image(df_cell)
        # Apply the grayscale conversion formula
        grayscale_image = np.dot(
            o_observation.transpose(
                1, 2, 0
            ),  # Move channels last to (height, width, 3)
            [0.2989, 0.5870, 0.1140],  # Weights for RGB to grayscale
        ).astype(np.uint8)

        return grayscale_image[np.newaxis, :, :]  # Shape (1, height, width)


@njit(parallel=True)
def colorize(x_valid, y_valid, types_valid, type_to_color_array, o_observation):
    for i in prange(len(x_valid)):
        type_idx = types_valid[i]
        if 0 <= type_idx < type_to_color_array.shape[0]:
            r, g, b = type_to_color_array[type_idx]
        else:
            r, g, b = 0, 0, 0

        x = x_valid[i]
        y = y_valid[i]

        # Numba doesn't like writing to global memory from multiple threads,
        # but we assume no collisions here because each (x, y) is unique.
        o_observation[0, x, y] = r
        o_observation[1, x, y] = g
        o_observation[2, x, y] = b


# --- Your class ---
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
        num_workers: int = 12,
    ):
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = [None] * self.buffer_size
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
        self.image_gray = image_gray
        self.num_workers = num_workers

        # --- Precompute the type_to_color_array for fast lookup ---
        self.type_to_color_array = self._make_type_to_color_array(type_to_color)

    def _make_type_to_color_array(self, type_to_color):
        max_type = max(type_to_color.keys())
        color_array = np.zeros((max_type + 1, 3), dtype=np.uint8)
        for t, color in type_to_color.items():
            color_array[t] = np.array(color, dtype=np.uint8)
        return color_array

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    def add(self, df_cell, action, reward, next_df_cell, done, type_to_int):
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

        minimal_array = np.stack([x, y, type_labels], axis=1)
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
                state_images = list(
                    executor.map(self.minimal_array_to_grayscale, state_list)
                )
                next_state_images = list(
                    executor.map(self.minimal_array_to_grayscale, next_state_list)
                )
            else:
                state_images = list(
                    executor.map(self.minimal_array_to_image, state_list)
                )
                next_state_images = list(
                    executor.map(self.minimal_array_to_image, next_state_list)
                )

        state_tensor = torch.from_numpy(np.stack(state_images)).float().to(self.device)
        next_state_tensor = (
            torch.from_numpy(np.stack(next_state_images)).float().to(self.device)
        )

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
        x = state_array[:, 0]
        y = state_array[:, 1]
        type_int = state_array[:, 2].astype(int)

        x_norm = (x - self.x_min).astype(int)
        y_norm = (y - self.y_min).astype(int)

        o_observation = np.zeros((3, self.height, self.width), dtype=np.uint8)

        valid_mask = (
            (0 <= x_norm)
            & (x_norm < self.height)
            & (0 <= y_norm)
            & (y_norm < self.width)
        )

        x_valid = x_norm[valid_mask]
        y_valid = y_norm[valid_mask]
        types_valid = type_int[valid_mask]

        # --- Fast colorization using Numba ---
        colorize(x_valid, y_valid, types_valid, self.type_to_color_array, o_observation)

        return o_observation

    def minimal_array_to_grayscale(self, state_array):
        color_image = self.minimal_array_to_image(state_array)
        grayscale_image = np.dot(
            color_image.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140]
        ).astype(np.uint8)
        return grayscale_image[np.newaxis, :, :]
