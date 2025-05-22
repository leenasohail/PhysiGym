import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tensordict import TensorDict
from numba import njit, prange


class ReplayBuffer(object):
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.
    Stores states, actions, rewards, next states, and done flags.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        buffer_size,
        batch_size,
        state_type=np.float32,
    ):
        """
        Initializes the replay buffer.

        Parameters:
        - state_dim (int): Dimensionality of the state space.
        - action_dim (int): Dimensionality of the action space.
        - device (torch.device): Device where tensors should be stored.
        - buffer_size (int): Maximum size of the replay buffer.
        - batch_size (int): Number of samples per batch.
        - state_type (numpy dtype, optional): Data type of the state representation (default: np.float32).
        """
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = np.empty((self.buffer_size, state_dim), dtype=state_type)
        self.next_state = np.empty((self.buffer_size, state_dim), dtype=state_type)
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

        self.buffer_index = 0
        self.full = False
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the current number of stored experiences in the buffer.
        """
        return self.buffer_size if self.full else self.buffer_index

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the replay buffer.

        Parameters:
        - state (np.ndarray): Current state.
        - action (np.ndarray): Action taken.
        - reward (float): Reward received.
        - next_state (np.ndarray): Next state after taking the action.
        - done (bool): Whether the episode has ended.
        """
        self.state[self.buffer_index] = state
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_state[self.buffer_index] = next_state
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

        # Ensure there are enough samples in the buffer
        assert self.full or (self.buffer_index > batch_size), (
            "Buffer does not have enough samples"
        )

        # Generate random indices for sampling
        sample_index = np.random.randint(
            0, self.buffer_size if self.full else self.buffer_index, batch_size
        )

        # Convert indices to tensors and gather the sampled experiences
        state = torch.as_tensor(self.state[sample_index]).float()
        next_state = torch.as_tensor(self.next_state[sample_index]).float()
        action = torch.as_tensor(self.action[sample_index])
        reward = torch.as_tensor(self.reward[sample_index])
        done = torch.as_tensor(self.done[sample_index])

        # Create a dictionary of the sampled experiences
        sample = TensorDict(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return sample


@njit(parallel=True)
def colorize_batch(
    batch_x: np.ndarray,  # (B, N)
    batch_y: np.ndarray,  # (B, N)
    batch_t: np.ndarray,  # (B, N)
    type_to_color_array: np.ndarray,
    o_batch: np.ndarray,  # (B, C, H, W)
    use_grayscale: bool = False,
    normalize: bool = False,
):
    B, N = batch_x.shape
    for i in prange(B):
        x_valid = batch_x[i]
        y_valid = batch_y[i]
        types_valid = batch_t[i]

        for j in range(len(x_valid)):
            type_idx = types_valid[j]

            if 0 <= type_idx < type_to_color_array.shape[0]:
                r = type_to_color_array[type_idx, 0]
                g = type_to_color_array[type_idx, 1]
                b = type_to_color_array[type_idx, 2]
            else:
                r = g = b = 0

            if normalize:
                r *= 255
                g *= 255
                b *= 255

            x = x_valid[j]
            y = y_valid[j]

            if use_grayscale:
                gray = r * 0.2989 + g * 0.5870 + b * 0.1140
                o_batch[i, 0, x, y] = int(gray)
            else:
                o_batch[i, 0, x, y] = int(r)
                o_batch[i, 1, x, y] = int(g)
                o_batch[i, 2, x, y] = int(b)


@njit(parallel=True)
def colorize(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    types_valid: np.ndarray,
    type_to_color_array: np.ndarray,
    o_observation: np.ndarray,
    use_grayscale: bool = False,
    normalize: bool = False,
):
    n = x_valid.shape[0]
    for i in prange(n):
        type_idx = types_valid[i]

        if 0 <= type_idx < type_to_color_array.shape[0]:
            r = type_to_color_array[type_idx, 0]
            g = type_to_color_array[type_idx, 1]
            b = type_to_color_array[type_idx, 2]
        else:
            r = g = b = 0

        if normalize:
            r *= 255
            g *= 255
            b *= 255

        x = x_valid[i]
        y = y_valid[i]

        if use_grayscale:
            gray = r * 0.2989 + g * 0.5870 + b * 0.1140
            o_observation[0, x, y] = int(gray)
        else:
            o_observation[0, x, y] = int(r)
            o_observation[1, x, y] = int(g)
            o_observation[2, x, y] = int(b)


class MinimalImgReplayBuffer:
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
        num_workers: int = 10,
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
        return np.stack([x, y, type_labels], axis=1)

    def batch_convert(self, state_list, grayscale: bool):
        B = len(state_list)
        N = max(s.shape[0] for s in state_list)

        # Preallocate arrays
        batch_x = np.zeros((B, N), dtype=np.int32)
        batch_y = np.zeros((B, N), dtype=np.int32)
        batch_t = np.zeros((B, N), dtype=np.int32)
        mask = np.zeros((B, N), dtype=np.bool_)

        for i, s in enumerate(state_list):
            x = (s[:, 0] - self.x_min).astype(int)
            y = (s[:, 1] - self.y_min).astype(int)
            t = s[:, 2].astype(int)

            valid = (0 <= x) & (x < self.height) & (0 <= y) & (y < self.width)
            num_valid = valid.sum()

            batch_x[i, :num_valid] = x[valid]
            batch_y[i, :num_valid] = y[valid]
            batch_t[i, :num_valid] = t[valid]
            mask[i, :num_valid] = True

        C = 1 if grayscale else 3
        o_batch = np.zeros((B, C, self.height, self.width), dtype=np.uint8)

        # Call Numba-accelerated colorization
        colorize_batch(
            batch_x,
            batch_y,
            batch_t,
            self.type_to_color_array,
            o_batch,
            grayscale,
            False,
        )

        return o_batch

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
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

        state_images = self.batch_convert(state_list, grayscale=self.image_gray)
        next_state_images = self.batch_convert(
            next_state_list, grayscale=self.image_gray
        )

        state_tensor = torch.from_numpy(np.stack(state_images)).float()
        next_state_tensor = torch.from_numpy(np.stack(next_state_images)).float()

        return TensorDict(
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

        colorize(
            x_valid,
            y_valid,
            types_valid,
            self.type_to_color_array,
            o_observation,
            False,
            False,
        )

        return o_observation

    def minimal_array_to_grayscale(self, state_array):
        x = state_array[:, 0]
        y = state_array[:, 1]
        type_int = state_array[:, 2].astype(int)

        x_norm = (x - self.x_min).astype(int)
        y_norm = (y - self.y_min).astype(int)

        o_observation = np.zeros((1, self.height, self.width), dtype=np.uint8)

        valid_mask = (
            (0 <= x_norm)
            & (x_norm < self.height)
            & (0 <= y_norm)
            & (y_norm < self.width)
        )

        x_valid = x_norm[valid_mask]
        y_valid = y_norm[valid_mask]
        types_valid = type_int[valid_mask]

        colorize(
            x_valid,
            y_valid,
            types_valid,
            self.type_to_color_array,
            o_observation,
            True,
            False,
        )

        return o_observation


class TransformerReplayBuffer:
    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        buffer_size: int,
        batch_size: int,
        type_pad_id: int,
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

        self.type_pad_id = -1  # Typically len(type_map) or -1

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    def add(self, state: dict, action, reward, next_state: dict, done):
        self.state[self.buffer_index] = state
        self.next_state[self.buffer_index] = next_state
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

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

        def stack_field(field_name, pad_value):
            seqs = [torch.tensor(s[field_name]) for s in state_list]
            return pad_sequence(seqs, batch_first=True, padding_value=pad_value)

        def stack_field_next(field_name, pad_value):
            seqs = [torch.tensor(s[field_name]) for s in next_state_list]
            return pad_sequence(seqs, batch_first=True, padding_value=pad_value)

        state_type = stack_field("type", self.type_pad_id).to(self.device)
        state_dead = stack_field("dead", 1).to(self.device)
        state_pos = stack_field("pos", [0.0, 0.0]).to(self.device)
        state_mask = (state_type != self.type_pad_id).int()

        next_state_type = stack_field_next("type", self.type_pad_id).to(self.device)
        next_state_dead = stack_field_next("dead", 1).to(self.device)
        next_state_pos = stack_field_next("pos", [0.0, 0.0]).to(self.device)
        next_state_mask = (next_state_type != self.type_pad_id).int()

        action = torch.tensor(self.action[sample_index], device=self.device)
        reward = torch.tensor(self.reward[sample_index], device=self.device)
        done = torch.tensor(self.done[sample_index], device=self.device)

        sample = TensorDict(
            {
                "state": TensorDict(
                    {
                        "type": state_type,
                        "dead": state_dead,
                        "pos": state_pos,
                        "mask": state_mask,
                    },
                    batch_size=batch_size,
                ),
                "next_state": TensorDict(
                    {
                        "type": next_state_type,
                        "dead": next_state_dead,
                        "pos": next_state_pos,
                        "mask": next_state_mask,
                    },
                    batch_size=batch_size,
                ),
                "action": action,
                "reward": reward,
                "done": done,
            },
            batch_size=batch_size,
        )
        return sample
