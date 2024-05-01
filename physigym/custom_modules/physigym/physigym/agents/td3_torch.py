import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

# Actor model
class Actor(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state:torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

# Critic model
class Critic(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, lr_actor=1e-3, lr_critic=1e-3, device="cpu"):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.step = 0

    def action(self, state:np.array):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size):
        self.step += 1
        if len(replay_buffer.buffer) < batch_size:
            return

        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Select next action according to target policy
        next_actions = self.actor_target(next_states)

        # Add Gaussian noise to next action (target policy smoothing)
        noise = torch.normal(torch.zeros_like(next_actions), self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions += noise

        # Clip action to be within bounds
        next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)

        # Compute target Q-values
        target_Q1 = self.critic1_target(next_states, next_actions)
        target_Q2 = self.critic2_target(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q.detach()

        # Compute current Q-values
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        # Compute critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        # Optimize critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.step % self.policy_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic1_target, self.critic1, self.tau)
            self.soft_update(self.critic2_target, self.critic2, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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

    env = gym.make("LunarLanderContinuous-v2", continuous =True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    parameters = {
        "max_size":100000,
        "batch_size":128,
        "num_episodes":1000,
        "num_steps":100

    }
    Trainer(env,agent,parameters)