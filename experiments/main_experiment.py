import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import BaseConfig
from src.models.dqn import DQNAgent
from src.environments.sql_injection_env import SQLInjectionEnv
from src.utils.data_utils import load_and_preprocess_data, create_sample_data, set_seed
from src.utils.training_utils import train_dqn, evaluate_dqn
from src.visualization.training_plots import plot_training_results, plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from sklearn.metrics import classification_report

set_seed(42)


def main():
    config = BaseConfig()

    print("SQL Injection Detection with GAT and Multi-Expert DQN")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Device: {config.device}")
    print(f"  - Dataset: {config.csv_file}")
    print(f"  - Training steps: {config.total_timesteps}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Hidden dimension: {config.hidden_dim}")
    print(f"  - Gamma: {config.gamma}")
    print(f"  - Epsilon decay: {config.epsilon_decay}")
    print("=" * 60)

    # Load data
    if not os.path.exists(config.csv_file):
        print("Creating sample dataset...")
        create_sample_data(config.csv_file)

    train_queries, train_labels, test_queries, test_labels = load_and_preprocess_data(
        config.csv_file, test_size=0.2
    )

    print(f"Dataset loaded:")
    print(f"  - Training samples: {len(train_queries)}")
    print(f"  - Testing samples: {len(test_queries)}")

    # Create environments
    env = SQLInjectionEnv(
        train_queries, train_labels,
        reward_setting=config.reward_setting,
        use_experts=True,
        use_graph_features=True,
        num_experts=config.num_experts
    )

    eval_env = SQLInjectionEnv(
        test_queries, test_labels,
        reward_setting=config.reward_setting,
        use_experts=True,
        use_graph_features=True,
        num_experts=config.num_experts
    )

    # Get input dimension
    input_dim = env.observation_space.shape[0]
    print(f"Observation space dimension: {input_dim}")

    # Create DQN agent
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
        use_dueling=True,
        use_double=True
    )

    # Train model
    print("\nStarting training...")
    training_results = train_dqn(
        env,
        agent,
        total_timesteps=config.total_timesteps,
        eval_env=eval_env,
        eval_freq=config.eval_freq,
        save_dir=config.models_dir
    )

    # Visualize training results
    plot_training_results(training_results, save_dir=config.visualizations_dir)

    # Load best model
    best_model_path = training_results['best_model_path']
    agent.load_model(best_model_path)

    # Evaluate final model
    print("\nEvaluating final model...")
    final_reward, final_accuracy, final_metrics = evaluate_dqn(agent, eval_env, num_episodes=100)
    print(f"Final Results:")
    print(f"  - Accuracy: {final_accuracy:.4f}")
    print(f"  - Average Reward: {final_reward:.2f}")

    # Generate evaluation plots
    true_labels = final_metrics['true_labels']
    malicious_probs = final_metrics['malicious_probs']
    predictions = final_metrics['predictions']

    # ROC curve
    roc_auc = plot_roc_curve(true_labels, malicious_probs,
                             title='SQL Injection Detection ROC Curve',
                             save_dir=config.visualizations_dir)
    print(f"  - ROC AUC: {roc_auc:.4f}")

    # PR curve
    pr_auc = plot_pr_curve(true_labels, malicious_probs,
                           title='SQL Injection Detection PR Curve',
                           save_dir=config.visualizations_dir)
    print(f"  - PR AUC: {pr_auc:.4f}")

    # Confusion matrix
    cm = plot_confusion_matrix(true_labels, predictions,
                               title='SQL Injection Detection Confusion Matrix',
                               save_dir=config.visualizations_dir)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Normal', 'Malicious']))

    print(f"\nExperiment completed!")
    print(f"Results saved to: {config.results_dir}")
    print(f"Visualizations saved to: {config.visualizations_dir}")


if __name__ == "__main__":
    main()