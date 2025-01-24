import torch
import numpy as np
from tensordict import TensorDict


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, buffer_size, batch_size):
        self.device = device
        self.buffer_size = int(buffer_size)

        self.state = np.empty((self.buffer_size, state_dim), dtype=np.float32)
        self.next_state = np.empty((self.buffer_size, state_dim), dtype=np.float32)
        self.action = np.empty((self.buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.done = np.empty((self.buffer_size, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def __len__(self):
        return self.buffer_size if self.full else self.buffer_index

    def add(self, state, action, reward, next_state, done):
        self.state[self.buffer_index] = state
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_state[self.buffer_index] = next_state
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.full = self.full or self.buffer_index == 0

    def sample(self):
        """
        Sample a batch of experiences from the replay buffer.
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
