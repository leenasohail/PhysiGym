from collections import deque
import random

import torch
from torch_geometric.data import Data, Batch
from tensordict import TensorDict

import numpy as np

##########################
# Classes Replay Buffers #
##########################


class ReplayBuffer:
    """
    Replay buffer supporting both array-based states and graph-based states.
    Graph states must be passed as GraphInstances or PyG Data objects with edge_attr.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        buffer_size,
        batch_size,
        state_type=np.float32,
        is_graph=False,
    ):
        self.device = device
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.is_graph = is_graph

        if not is_graph:
            # Preallocate memory for speed
            self.state = np.empty((self.buffer_size, *state_dim), dtype=state_type)
            self.next_state = np.empty((self.buffer_size, *state_dim), dtype=state_type)
            self.action = np.empty((self.buffer_size, *action_dim), dtype=np.float32)
            self.reward = np.empty((self.buffer_size, 1), dtype=np.float32)
            self.done = np.empty((self.buffer_size, 1), dtype=np.uint8)

            self.buffer_index = 0
            self.full = False
        else:
            # For variable-size graphs, use a deque
            self.buffer = deque(maxlen=self.buffer_size)

    def __len__(self):
        if self.is_graph:
            return len(self.buffer)
        else:
            return self.buffer_size if self.full else self.buffer_index

    def add_batch(self, batch):
        """
        batch: list of (state, action, reward, next_state, done)
        """
        for transition in batch:
            self.add(*transition)

    def add(self, state, action, reward, next_state, done):
        if not self.is_graph:
            self.state[self.buffer_index] = state
            self.action[self.buffer_index] = action
            self.reward[self.buffer_index] = reward
            self.next_state[self.buffer_index] = next_state
            self.done[self.buffer_index] = done

            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.full = self.full or self.buffer_index == 0
        else:
            state_graph = self._dict_reduced(state)
            next_state_graph = self._dict_reduced(next_state)
            self.buffer.append((state_graph, action, reward, next_state_graph, done))

    def _dict_reduced(self, obs):
        """
        Convert padded dict observation from SubprocVecEnv back
        into a true variable-size graph for the replay buffer.
        """
        node_mask = obs["node_mask"] > 0.5
        edge_mask = obs["edge_mask"] > 0.5

        # Extract only valid nodes and edges
        nodes = obs["node_features"][node_mask]  # shape: (N, node_dim)
        edge_index = obs["edge_index"][:, edge_mask]  # shape: (2, E)
        edges = obs["edge_attr"][edge_mask]  # shape: (E, edge_dim)

        return {"nodes": nodes, "edge_links": edge_index, "edges": edges}

    def sample(self):
        if not self.is_graph:
            sample_index = np.random.randint(
                0, self.buffer_size if self.full else self.buffer_index, self.batch_size
            )

            state = torch.as_tensor(
                self.state[sample_index], device=self.device
            ).float()
            next_state = torch.as_tensor(
                self.next_state[sample_index], device=self.device
            ).float()
            action = torch.as_tensor(self.action[sample_index], device=self.device)
            reward = torch.as_tensor(self.reward[sample_index], device=self.device)
            done = torch.as_tensor(self.done[sample_index], device=self.device)

            return TensorDict(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                },
                batch_size=self.batch_size,
                device=self.device,
            )
        else:
            batch = random.sample(self.buffer, self.batch_size)
            _state, action, reward, _next_state, done = zip(*batch)

            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            reward = torch.tensor(
                reward, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
            done = torch.tensor(done, dtype=torch.uint8, device=self.device)
            state = []
            for stati in _state:
                state.append(
                    Data(
                        x=torch.tensor(
                            stati["nodes"], dtype=torch.float, device=self.device
                        ),
                        edge_index=torch.tensor(
                            stati["edge_links"], dtype=torch.long, device=self.device
                        ),
                        edge_attr=torch.tensor(
                            stati["edges"], dtype=torch.float, device=self.device
                        ),
                    )
                )
            next_state = []
            for next_stati in _next_state:
                next_state.append(
                    Data(
                        x=torch.tensor(
                            next_stati["nodes"], dtype=torch.float, device=self.device
                        ),
                        edge_index=torch.tensor(
                            next_stati["edge_links"],
                            dtype=torch.long,
                            device=self.device,
                        ),
                        edge_attr=torch.tensor(
                            next_stati["edges"], dtype=torch.float, device=self.device
                        ),
                    )
                )

            # Graphs remain Python objects (list of GraphInstances)
            return {
                "state": Batch.from_data_list(state),
                "action": action,
                "reward": reward,
                "done": done,
                "next_state": Batch.from_data_list(next_state),
            }
