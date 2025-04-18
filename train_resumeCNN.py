import time
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from connect4_env import Connect4Env, CustomConnect4CNN
from torch.utils.tensorboard import SummaryWriter

def make_env():
    def _init():
        return Connect4Env()
    return _init

def convert_to_cpu(params):
    if isinstance(params, dict):
        return {k: convert_to_cpu(v) for k, v in params.items()}
    elif hasattr(params, "detach"):
        return params.detach().cpu()
    else:
        return params

class WinRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WinRateCallback, self).__init__(verbose)
        self.win_rates = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        rewards = np.array(self.model.rollout_buffer.rewards)
        wins = np.sum(rewards > 0.5)
        total = len(rewards)
        if total > 0:
            win_rate = wins / total
            self.win_rates.append(win_rate)
            if self.verbose:
                print(f"[Callback] Approximate win rate from rollout: {win_rate*100:.2f}%")
        return True

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir="./tensorboard_logs_resume/")
        self.step_counter = 0

    def _on_step(self):
        self.step_counter += 1
        if self.step_counter % 10000 == 0:
            win_rate = evaluate_agent(self.model, opponent_params=None, num_episodes=100)
            self.writer.add_scalar("WinRate/RandomOpponent", win_rate, self.step_counter)
            print(f"[Tensorboard] Win rate against random at {self.step_counter} steps: {win_rate*100:.2f}%")
        return True

    def _on_training_end(self):
        self.writer.close()

def evaluate_agent(model, opponent_params, num_episodes=100):
    wins = 0
    env = Connect4Env()
    env.set_opponent_params(opponent_params)
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward >= 1:  # Adjusted for new reward structure
                wins += 1
                break
    env.close()
    return wins / num_episodes

def train_self_play_resume(model_path="connect4_cnn_master", heuristic_timesteps=50000, total_timesteps=1000000, update_interval=2000, save_interval=10000, eval_threshold=0.55, max_hours=12):
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    policy_kwargs = dict(
        features_extractor_class=CustomConnect4CNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64]
    )

    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading model from {model_path}.zip")
        try:
            model = PPO.load(model_path, env=env, device="cuda")
            model.learning_rate = 1e-4  # Reset learning rate if desired
            print("Model loaded successfully.")
        except ValueError as e:
            print(f"Error loading model: {e}")
            print("Starting fresh with a new CNN model.")
            model = PPO(
                "CnnPolicy",
                env,
                verbose=1,
                device="cuda",
                learning_rate=1e-4,
                ent_coef=0.05,
                batch_size=128,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./tensorboard_logs_resume/"
            )
    else:
        raise FileNotFoundError(f"No model found at {model_path}.zip. Please provide a valid pre-trained model.")

    win_rate_callback = WinRateCallback(verbose=1)
    tensorboard_callback = TensorboardCallback(verbose=1)
    callbacks = CallbackList([win_rate_callback, tensorboard_callback])

    timesteps_so_far = 0
    games_since_update = 0
    ensemble = []

    start_time = time.time()
    max_seconds = max_hours * 3600

    while timesteps_so_far < total_timesteps and (time.time() - start_time) < max_seconds:
        # First 50k timesteps: heuristic opponent
        if timesteps_so_far < heuristic_timesteps:
            env.env_method('set_opponent_params', "heuristic")
            print(f"Training against heuristic opponent at {timesteps_so_far} timesteps")
        # After 50k: ensemble self-play
        else:
            chosen = np.random.choice(ensemble) if ensemble else None
            env.env_method('set_opponent_params', chosen)
            print(f"Training against ensemble opponent at {timesteps_so_far} timesteps (ensemble size: {len(ensemble)})")

        model.learn(total_timesteps=update_interval, callback=callbacks, reset_num_timesteps=False)
        timesteps_so_far += update_interval
        print(f"Trained {timesteps_so_far} timesteps")

        # Evaluate performance
        if ensemble:
            win_rates = [evaluate_agent(model, opp, num_episodes=100) for opp in ensemble]
            avg_win_rate = np.mean(win_rates)
            print(f"Avg win rate against ensemble: {avg_win_rate*100:.2f}%")
        else:
            avg_win_rate = evaluate_agent(model, None, num_episodes=200)
            print(f"Win rate against random opponent: {avg_win_rate*100:.2f}%")

        # Update ensemble
        games_since_update += 200
        if games_since_update >= 200 and avg_win_rate >= eval_threshold:
            new_params = convert_to_cpu(model.get_parameters())
            ensemble.append(new_params)
            if len(ensemble) > 4:
                ensemble.pop(0)
            print(f"Ensemble updated. Size: {len(ensemble)}")
            games_since_update = 0

        # Save every 10k timesteps
        if timesteps_so_far % save_interval == 0:
            save_path = f"{model_path}_resume_{timesteps_so_far}.zip"
            model.save(save_path)
            print(f"Model saved at {save_path}")

    # Final save
    model.save(model_path)
    print(f"Final model saved at {model_path}.zip")
    return model

if __name__ == "__main__":
    trained_model = train_self_play_resume(
        model_path="connect4_cnn_master",
        heuristic_timesteps=50000,
        total_timesteps=1000000,
        update_interval=2000,
        save_interval=10000,
        eval_threshold=0.54,
        max_hours=27
    )