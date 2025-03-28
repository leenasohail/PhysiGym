import torch
import numpy as np
import pickle
import pandas as pd
from tensordict import TensorDict


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
