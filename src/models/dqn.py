import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class StandardDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StandardDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class DQNAgent:
    def __init__(self, input_dim, hidden_dim=128, action_dim=2, lr=3e-4, batch_size=64,
                 gamma=0.99, epsilon_decay=0.997, target_update_freq=1000,
                 memory_size=10000, use_dueling=True, use_double=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.learning_rate = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.memory_size = memory_size
        self.use_dueling = use_dueling
        self.use_double = use_double

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.train_info = {
            'losses': [],
            'q_values': [],
            'rewards': [],
            'accuracies': [],
            'epsilons': []
        }
        self.train_steps = 0

    def _build_network(self):
        if self.use_dueling:
            self.q_network = DuelingDQN(self.input_dim, self.hidden_dim, self.action_dim).to(self.device)
            self.target_network = DuelingDQN(self.input_dim, self.hidden_dim, self.action_dim).to(self.device)
        else:
            self.q_network = StandardDQN(self.input_dim, self.hidden_dim, self.action_dim).to(self.device)
            self.target_network = StandardDQN(self.input_dim, self.hidden_dim, self.action_dim).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if deterministic or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(0, self.action_dim)

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        if self.use_double:
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q_values = self.target_network(next_states).max(1)[0]

        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_info['losses'].append(loss.item())
        self.train_info['q_values'].append(current_q_values.mean().item())

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_info['epsilons'].append(self.epsilon)

    def save_model(self, filepath):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_info': self.train_info,
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_info = checkpoint['train_info']
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']