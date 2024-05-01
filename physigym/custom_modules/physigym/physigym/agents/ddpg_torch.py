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

# DDPG agent
class DDPGAgent:
    """_summary_
    """
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr_actor=1e-3, lr_critic=1e-3, device="cpu"):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

    def action(self, state:np.array):
        """_summary_

        Args:
            state (np.array): _description_

        Returns:
            _type_: _description_
        """
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size):
        # Like in supervised learning
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device) # to convert a numpy array (size (batch_size)) into torch tensor size (batch_size,1) 
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device) # to convert a numpy array (size (batch_size)) into torch tensor size (batch_size,1)

        # Update critic
        next_actions = self.actor_target(next_states) # next action given by the actor_target 
        next_Q = self.critic_target(next_states, next_actions) # next Q given by the critic_target 
        target_Q = rewards + (1 - dones) * self.gamma * next_Q.detach() # target Q is updated whiwh is the r_{t} + if it is done or not multiplied by the next_Q and gamma ( discount between 0 and 1)
        current_Q = self.critic(states, actions) # the critic computes the current Q
        critic_loss = F.mse_loss(current_Q, target_Q) # in theory  we should converge to this equality current_Q = target_Q

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    parameters = {
        "max_size":100000,
        "batch_size":128,
        "num_episodes":1000,
        "num_steps":100

    }
    Trainer(env,agent,parameters)
    
