import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import random
import json
import numpy as np
from tqdm import tqdm
from config.experiment_config import HyperparameterConfig
from src.models.dqn import DQNAgent
from src.environments.sql_injection_env import SQLInjectionEnv
from src.utils.data_utils import load_and_preprocess_data, create_sample_data, set_seed
from src.utils.training_utils import train_dqn, evaluate_dqn
from src.visualization.training_plots import plot_training_results

set_seed(42)

class HyperparameterTuner:
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels
        self.config = HyperparameterConfig()
        self.results = {}

    def generate_hyperparameter_combinations(self):
        param_grid = {
            'learning_rate': self.config.learning_rates,
            'batch_size': self.config.batch_sizes,
            'hidden_dim': self.config.hidden_dims,
            'gamma': self.config.gamma_values,
            'epsilon_decay': self.config.epsilon_decay_values,
            'target_update_freq': self.config.target_update_freqs
        }

        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        if len(combinations) > self.config.max_combinations:
            combinations = random.sample(combinations, self.config.max_combinations)

        hyperparams_list = []
        for combo in combinations:
            hyperparams = dict(zip(keys, combo))
            hyperparams_list.append(hyperparams)

        return hyperparams_list

    def run_experiment(self, hyperparams, reward_setting='dense'):
        exp_name = (f"DQN_lr{hyperparams['learning_rate']}_"
                    f"batch{hyperparams['batch_size']}_"
                    f"hidden{hyperparams['hidden_dim']}_"
                    f"gamma{hyperparams['gamma']}_"
                    f"eps{hyperparams['epsilon_decay']}_"
                    f"target{hyperparams['target_update_freq']}")

        env = SQLInjectionEnv(self.queries, self.labels, self.config.reward_setting)
        input_dim = env.observation_space.shape[0]

        agent = DQNAgent(
            input_dim=input_dim,
            hidden_dim=hyperparams['hidden_dim'],
            action_dim=2,
            lr=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            gamma=hyperparams['gamma'],
            epsilon_decay=hyperparams['epsilon_decay'],
            target_update_freq=hyperparams['target_update_freq']
        )

        try:
            results = train_dqn(
                env, agent,
                total_timesteps=self.config.total_timesteps,
                eval_env=env,
                eval_freq=self.config.eval_freq,
                save_dir=self.config.models_dir
            )

            self.results[exp_name] = {
                'hyperparams': hyperparams,
                'results': results
            }

            return agent, results

        except Exception as e:
            print(f"Experiment failed: {e}")
            return None, None

    def run_tuning_experiments(self):
        hyperparams_list = self.generate_hyperparameter_combinations()

        for i, hyperparams in enumerate(hyperparams_list):
            print(f"\nExperiment {i + 1}/{len(hyperparams_list)}")
            print(f"Hyperparams: {hyperparams}")
            self.run_experiment(hyperparams)

        self.save_results()
        self.print_best_hyperparams()

    def save_results(self):
        results_file = os.path.join(self.config.tuning_dir, 'hyperparameter_results.json')
        with open(results_file, 'w') as f:
            serializable_results = {}
            for key, value in self.results.items():
                serializable_results[key] = {
                    'hyperparams': value['hyperparams'],
                    'final_accuracy': value['results']['eval_accuracies'][-1] if value['results']['eval_accuracies'] else 0
                }
            json.dump(serializable_results, f, indent=2)

    def print_best_hyperparams(self):
        if not self.results:
            return

        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['results']['eval_accuracies'][-1] if x[1]['results']['eval_accuracies'] else 0,
            reverse=True
        )

        print("\nBest hyperparameter combinations:")
        for i, (name, result) in enumerate(sorted_results[:3]):
            final_acc = result['results']['eval_accuracies'][-1] if result['results']['eval_accuracies'] else 0
            print(f"\n{i + 1}. {name}")
            print(f"   Accuracy: {final_acc:.4f}")
            print(f"   Hyperparams: {result['hyperparams']}")


def main():
    config = HyperparameterConfig()

    if not os.path.exists(config.csv_file):
        create_sample_data(config.csv_file)

    train_queries, train_labels, test_queries, test_labels = load_and_preprocess_data(
        config.csv_file, test_size=0.2
    )

    tuner = HyperparameterTuner(train_queries, train_labels)
    tuner.run_tuning_experiments()


if __name__ == "__main__":
    main()