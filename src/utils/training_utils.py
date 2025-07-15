import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def train_dqn(env, agent, total_timesteps, eval_env=None, eval_freq=5000, early_stopping_patience=10, save_dir="results/models"):
    pbar = tqdm(total=total_timesteps, desc="Training Progress")

    best_eval_score = -float('inf')
    best_model_path = os.path.join(save_dir, "best_dqn_model.pt")
    no_improvement_count = 0

    eval_steps = []
    eval_scores = []
    eval_accuracies = []

    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_correct = 0

    for step in range(1, total_timesteps + 1):
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)

        agent.train_steps += 1
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_steps += 1
        if info['correct']:
            episode_correct += 1

        if len(agent.memory) >= agent.batch_size:
            loss = agent.train_step()

        if step % agent.target_update_freq == 0:
            agent.update_target_network()

        if step % 100 == 0:
            agent.decay_epsilon()

        if done:
            accuracy = episode_correct / episode_steps if episode_steps > 0 else 0
            agent.train_info['rewards'].append(episode_reward)
            agent.train_info['accuracies'].append(accuracy)

            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_correct = 0

        if eval_env is not None and step % eval_freq == 0:
            eval_score, eval_accuracy, eval_metrics = evaluate_dqn(agent, eval_env)

            eval_steps.append(step)
            eval_scores.append(eval_score)
            eval_accuracies.append(eval_accuracy)

            pbar.set_postfix({
                'epsilon': f'{agent.epsilon:.3f}',
                'eval_acc': f'{eval_accuracy:.3f}',
                'eval_reward': f'{eval_score:.2f}'
            })

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                os.makedirs(save_dir, exist_ok=True)
                agent.save_model(best_model_path)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                break

        pbar.update(1)

    pbar.close()

    final_model_path = os.path.join(save_dir, "final_dqn_model.pt")
    os.makedirs(save_dir, exist_ok=True)
    agent.save_model(final_model_path)

    return {
        'eval_steps': eval_steps,
        'eval_scores': eval_scores,
        'eval_accuracies': eval_accuracies,
        'train_info': agent.train_info,
        'best_score': best_eval_score,
        'best_model_path': best_model_path
    }


def evaluate_dqn(agent, env, num_episodes=50):
    total_rewards = 0
    correct_predictions = 0
    total_steps = 0

    all_predictions = []
    all_true_labels = []
    all_malicious_probs = []

    agent.q_network.eval()

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_tensor)
                action = torch.argmax(q_values).item()

                malicious_prob = F.softmax(q_values, dim=1)[0, 1].item()
                all_malicious_probs.append(malicious_prob)

            next_state, reward, done, _, info = env.step(action)

            episode_reward += reward

            if info['correct']:
                correct_predictions += 1

            all_predictions.append(info['predicted_label'])
            all_true_labels.append(info['true_label'])

            state = next_state
            total_steps += 1

            if total_steps > 1000:
                break

        total_rewards += episode_reward

        if total_steps > 1000:
            break

    avg_reward = total_rewards / num_episodes
    accuracy = correct_predictions / total_steps if total_steps > 0 else 0

    eval_metrics = {
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'malicious_probs': all_malicious_probs
    }

    agent.q_network.train()

    return avg_reward, accuracy, eval_metrics