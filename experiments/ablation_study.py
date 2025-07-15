import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import AblationConfig
from src.models.dqn import DQNAgent
from src.environments.sql_injection_env import SQLInjectionEnv
from src.utils.data_utils import load_and_preprocess_data, create_sample_data, set_seed
from src.utils.training_utils import train_dqn, evaluate_dqn
from src.visualization.ablation_plots import plot_multiple_roc_curves, plot_multiple_pr_curves, plot_ablation_results

set_seed(42)

def run_ablation_studies(queries, labels, test_queries, test_labels):
    config = AblationConfig()
    ablation_results = {}
    models_eval_metrics = {}

    # Baseline experiment (complete model)
    print("Running baseline experiment (complete model)")
    baseline_env = SQLInjectionEnv(
        queries, labels,
        use_experts=True,
        use_graph_features=True,
        num_experts=config.num_experts
    )

    eval_env = SQLInjectionEnv(
        test_queries, test_labels,
        use_experts=True,
        use_graph_features=True,
        num_experts=config.num_experts
    )

    input_dim = baseline_env.observation_space.shape[0]

    baseline_agent = DQNAgent(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        action_dim=2,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.gamma,
        epsilon_decay=config.epsilon_decay,
        target_update_freq=config.target_update_freq,
        memory_size=config.memory_size,
        use_dueling=True,
        use_double=True
    )

    baseline_results = train_dqn(
        baseline_env,
        baseline_agent,
        total_timesteps=config.total_timesteps,
        eval_env=eval_env,
        eval_freq=config.eval_freq,
        save_dir=config.models_dir
    )

    _, _, baseline_metrics = evaluate_dqn(baseline_agent, eval_env, num_episodes=100)
    models_eval_metrics['Complete Model'] = baseline_metrics

    ablation_results['baseline'] = {
        'name': 'Complete Model',
        'description': 'Using all components',
        'results': baseline_results
    }

    # Ablation experiments
    ablation_configs = [
        {
            'name': 'w/o Graph Features',
            'description': 'Remove SQL syntax graph features',
            'env_args': {'use_graph_features': False, 'use_experts': True}
        },
        {
            'name': 'w/o Expert Agents',
            'description': 'Remove expert agents',
            'env_args': {'use_graph_features': True, 'use_experts': False}
        },
        {
            'name': 'w/o Dueling Network',
            'description': 'Use standard DQN instead of Dueling DQN',
            'agent_args': {'use_dueling': False, 'use_double': False}
        },
        {
            'name': 'w/o Dense Reward',
            'description': 'Use simple reward function',
            'reward_setting': {'correct': 1.0, 'wrong': -1.0, 'malicious_detect': 1.0, 'false_positive': -1.0}
        }
    ]

    for config_idx, ablation_config in enumerate(ablation_configs):
        print(f"\nRunning ablation experiment {config_idx + 1}/{len(ablation_configs)}: {ablation_config['name']}")

        env_args = ablation_config.get('env_args', {})
        reward_setting = ablation_config.get('reward_setting', None)

        env = SQLInjectionEnv(
            queries, labels,
            reward_setting=reward_setting,
            **env_args
        )

        eval_env = SQLInjectionEnv(
            test_queries, test_labels,
            reward_setting=reward_setting,
            **env_args
        )

        input_dim = env.observation_space.shape[0]
        agent_args = ablation_config.get('agent_args', {})

        agent = DQNAgent(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            action_dim=2,
            lr=config.learning_rate,
            batch_size=config.batch_size,
            gamma=config.gamma,
            epsilon_decay=config.epsilon_decay,
            target_update_freq=config.target_update_freq,
            memory_size=config.memory_size,
            **agent_args
        )

        results = train_dqn(
            env,
            agent,
            total_timesteps=config.total_timesteps,
            eval_env=eval_env,
            eval_freq=config.eval_freq,
            save_dir=config.models_dir
        )

        _, _, eval_metrics = evaluate_dqn(agent, eval_env, num_episodes=100)
        models_eval_metrics[ablation_config['name']] = eval_metrics

        ablation_results[ablation_config['name']] = {
            'name': ablation_config['name'],
            'description': ablation_config['description'],
            'results': results
        }

    # Generate visualizations
    plot_multiple_roc_curves(models_eval_metrics, save_dir=config.ablation_dir)
    plot_multiple_pr_curves(models_eval_metrics, save_dir=config.ablation_dir)
    plot_ablation_results(ablation_results, save_dir=config.ablation_dir)

    # Analyze results
    analyze_ablation_results(ablation_results)

    return ablation_results

def analyze_ablation_results(ablation_results):
    final_accuracies = {}
    for name, results in ablation_results.items():
        eval_accuracies = results['results']['eval_accuracies']
        if eval_accuracies:
            final_accuracies[name] = eval_accuracies[-1]
        else:
            final_accuracies[name] = 0

    print("\nAblation Study Results:")
    for name, acc in sorted(final_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {acc:.4f}")

    if 'baseline' in final_accuracies:
        baseline_acc = final_accuracies['baseline']
        print("\nComponent Contribution Analysis:")
        for name, acc in final_accuracies.items():
            if name != 'baseline':
                contribution = baseline_acc - acc
                relative_contrib = contribution / baseline_acc * 100
                print(f"  {name}: contribution {contribution:.4f} ({relative_contrib:.2f}%)")

def main():
    config = AblationConfig()

    if not os.path.exists(config.csv_file):
        create_sample_data(config.csv_file)

    train_queries, train_labels, test_queries, test_labels = load_and_preprocess_data(
        config.csv_file, test_size=0.2
    )

    ablation_results = run_ablation_studies(train_queries, train_labels, test_queries, test_labels)

if __name__ == "__main__":
    main()