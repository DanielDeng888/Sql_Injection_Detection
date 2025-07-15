import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from ..models.graph_builder import SQLGrammarGraphBuilder
from ..models.expert_agents import ExpertAgent

class SQLInjectionEnv(gym.Env):
    def __init__(self, queries, labels, reward_setting=None, use_experts=True,
                 use_graph_features=True, num_experts=4):
        super(SQLInjectionEnv, self).__init__()

        self.queries = np.array(queries)
        self.labels = np.array(labels)
        self.reward_config = reward_setting or {
            'correct': 1.0,
            'wrong': -0.5,
            'malicious_detect': 3.0,
            'false_positive': -1.0
        }

        self.use_experts = use_experts
        self.use_graph_features = use_graph_features
        self.num_experts = num_experts

        self.graph_builder = SQLGrammarGraphBuilder()

        if self.use_experts:
            self.experts = ExpertAgent.create_experts(10, 64, self.num_experts)
        else:
            self.experts = None

        obs_dim = 0
        if self.use_graph_features:
            obs_dim += 10
        if self.use_experts:
            obs_dim += self.num_experts

        if obs_dim == 0:
            obs_dim = 1

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.current_idx = 0
        self.episode_step = 0
        self.max_episode_steps = 1000

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_idx = np.random.randint(0, len(self.queries))
        self.episode_step = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        true_label = self.labels[self.current_idx]
        predicted_label = action

        reward = self._calculate_reward(predicted_label, true_label)
        correct = (predicted_label == true_label)

        self.current_idx = (self.current_idx + 1) % len(self.queries)
        self.episode_step += 1

        done = self.episode_step >= self.max_episode_steps
        obs = self._get_observation()

        info = {
            'correct': correct,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'is_malicious': true_label == 1,
            'reward': reward
        }

        return obs, reward, done, False, info

    def _get_observation(self):
        query = self.queries[self.current_idx]
        features = []

        if self.use_graph_features:
            graph_features = self.graph_builder.build_dependency_graph(query)
            features.append(graph_features)

        if self.use_experts:
            expert_predictions = []
            with torch.no_grad():
                graph_features_tensor = torch.FloatTensor(self.graph_builder.build_dependency_graph(query))
                for expert in self.experts:
                    pred = expert(graph_features_tensor)
                    expert_predictions.append(pred.item())
            features.append(np.array(expert_predictions, dtype=np.float32))

        if not features:
            features = [np.array([0.5], dtype=np.float32)]

        observation = np.concatenate(features)
        return observation.astype(np.float32)

    def _calculate_reward(self, predicted, true_label):
        if predicted == true_label:
            if true_label == 1:
                return self.reward_config['malicious_detect']
            else:
                return self.reward_config['correct']
        else:
            if predicted == 1 and true_label == 0:
                return self.reward_config['false_positive']
            else:
                return self.reward_config['wrong']