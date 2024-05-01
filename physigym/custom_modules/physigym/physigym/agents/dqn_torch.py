import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random

# Actor model
class QNetwork(nn.Module):
    def __init__(self, state_dim:int, n_action:int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_action)

    def forward(self, state:torch.Tensor):
        x = F.silu(self.fc1(state))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x
# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size:int):
        self.max_size = max_size
        self.buffer = []

    def push(self, transition:tuple):
        self.buffer.append(transition)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size:int):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, device="cpu"):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_q_network = QNetwork(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device

    def action(self, state):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return torch.argmax(q_values).item()

    def train(self, replay_buffer, batch_size):
        

        # Like in supervised learning
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # to convert a numpy array (size (batch_size)) into torch tensor size (batch_size,1) 
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device) # to convert a numpy array (size (batch_size)) into torch tensor size (batch_size,1)


        q_values = self.q_network(states).gather(1, actions.unsqueeze(1).long())


        next_q_values = self.target_q_network(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values.unsqueeze(1)

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def Trainer(env,agent,parameters):
    replay_buffer = ReplayBuffer(max_size=parameters["max_size"])
    batch_size = parameters["batch_size"]
    num_episodes = parameters["num_episodes"]
    num_steps = parameters["num_steps"]
    for episode in range(num_episodes):
        state = env.reset()
        state = state[0]
        total_reward = 0
        for _ in range(num_steps):
            action = agent.action(state)
            next_state, reward, done, _ , _ = env.step(action)
            replay_buffer.push((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            if len(replay_buffer.buffer) > batch_size:
                agent.train(replay_buffer, batch_size)

            if done:
                break

        print("Episode: {}, Total Reward: {:.2f}".format(episode, total_reward))

    env.close()

if __name__ =="__main__":
    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("LunarLanderContinuous-v2", continuous =False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    parameters = {
        "max_size":100000,
        "batch_size":128,
        "num_episodes":1000,
        "num_steps":100

    }
    Trainer(env,agent,parameters)
    