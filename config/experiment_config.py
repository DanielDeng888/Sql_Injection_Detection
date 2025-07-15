from .base_config import BaseConfig


class HyperparameterConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Hyperparameter grids
        self.learning_rates = [1e-4, 3e-4, 1e-3]
        self.batch_sizes = [32, 64, 128]
        self.hidden_dims = [64, 128, 256]
        self.gamma_values = [0.95, 0.99]
        self.epsilon_decay_values = [0.995, 0.997, 0.999]
        self.target_update_freqs = [500, 1000, 2000]

        self.max_combinations = 10
        self.human_baseline = 0.93
        self.target_performance = 0.80

        # Results directory
        self.tuning_dir = "results/hyperparameter_tuning"
        import os
        os.makedirs(self.tuning_dir, exist_ok=True)


class AblationConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.total_timesteps = 100000

        self.ablation_components = [
            "graph_features",
            "expert_agents",
            "reward_shaping",
            "network_architecture"
        ]

        # Results directory
        self.ablation_dir = "results/ablation_study"
        import os
        os.makedirs(self.ablation_dir, exist_ok=True)