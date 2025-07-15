import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn import StandardDQN, DuelingDQN, DQNAgent
from src.models.expert_agents import ExpertAgent
from src.models.graph_builder import SQLGrammarGraphBuilder
from src.environments.sql_injection_env import SQLInjectionEnv


class TestModels(unittest.TestCase):

    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 64
        self.output_dim = 2
        self.batch_size = 32

    def test_standard_dqn(self):
        model = StandardDQN(self.input_dim, self.hidden_dim, self.output_dim)
        x = torch.randn(self.batch_size, self.input_dim)
        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_dueling_dqn(self):
        model = DuelingDQN(self.input_dim, self.hidden_dim, self.output_dim)
        x = torch.randn(self.batch_size, self.input_dim)
        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_expert_agent(self):
        expert = ExpertAgent(self.input_dim, self.hidden_dim, "Test Expert")
        x = torch.randn(self.batch_size, self.input_dim)
        output = expert(x)

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_graph_builder(self):
        builder = SQLGrammarGraphBuilder()
        query = "SELECT * FROM users WHERE id = 1 OR 1=1"
        features = builder.build_dependency_graph(query)

        self.assertEqual(len(features), 10)
        self.assertTrue(np.all(features >= 0) and np.all(features <= 1))

    def test_sql_injection_env(self):
        queries = ["SELECT * FROM users", "SELECT * FROM users WHERE 1=1"]
        labels = [0, 1]

        env = SQLInjectionEnv(queries, labels)
        obs, info = env.reset()

        self.assertEqual(len(obs), env.observation_space.shape[0])

        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)

        self.assertEqual(len(next_obs), env.observation_space.shape[0])
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)

    def test_dqn_agent(self):
        agent = DQNAgent(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.output_dim,
            use_dueling=True,
            use_double=True
        )

        state = np.random.random(self.input_dim)
        action = agent.select_action(state, deterministic=True)

        self.assertIn(action, [0, 1])

        # Test memory
        next_state = np.random.random(self.input_dim)
        agent.remember(state, action, 1.0, next_state, False)

        self.assertEqual(len(agent.memory), 1)


if __name__ == '__main__':
    unittest.main()