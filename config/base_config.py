import os
import torch


class BaseConfig:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.csv_file = 'data/sql_injection_dataset.csv'

        # Model parameters
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.hidden_dim = 128
        self.gamma = 0.99
        self.epsilon_decay = 0.997
        self.target_update_freq = 1000
        self.memory_size = 10000
        self.num_experts = 4

        # Training parameters
        self.total_timesteps = 50000
        self.eval_freq = 5000
        self.eval_episodes = 50

        # Reward settings
        self.reward_setting = {
            'correct': 1.0,
            'wrong': -0.5,
            'malicious_detect': 3.0,
            'false_positive': -1.0
        }

        # Directories
        self.results_dir = "results"
        self.visualizations_dir = "results/visualizations"
        self.models_dir = "results/models"

        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)