import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from connect4_env import Connect4Env

def make_env(opponent_params=None):
    def _init():
        env = Connect4Env()
        env.set_opponent_params(opponent_params)
        return env
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
        super().__init__(verbose)
        self.win_rates = []

    def _on_step(self) -> bool:
        return True  # required by BaseCallback

    def _on_rollout_end(self) -> None:
        rewards = np.array(self.model.rollout_buffer.rewards)
        wins = np.sum(rewards > 0.5)
        total = len(rewards)
        if total > 0:
            win_rate = wins / total
            self.win_rates.append(win_rate)
            if self.verbose:
                print(f"[Callback] Approximate win rate from rollout: {win_rate*100:.2f}%")

def evaluate_agent(model, opponent_params, num_episodes=200):
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
        if reward >= 1:
            wins += 1
    env.close()
    return wins / num_episodes

def train_self_play(total_timesteps=100000, update_interval=2000, model=None):
    num_envs = 6
    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))  # fixed SB3 warning

    # Start vs heuristic
    opponent_mode = "heuristic"
    heuristic_opponent = None  # encoded as None in env
    ensemble_opponents = []

    env = SubprocVecEnv([make_env(heuristic_opponent) for _ in range(num_envs)])

    if model is None:
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cpu")  # switch to CPU for MLP
    else:
        model.set_env(env)

    win_rate_callback = WinRateCallback(verbose=1)
    timesteps_so_far = 0
    games_since_update = 0
    opponent_index = 0

    while timesteps_so_far < total_timesteps:
        model.learn(total_timesteps=update_interval, callback=win_rate_callback)
        timesteps_so_far += update_interval
        print(f"Trained {timesteps_so_far} timesteps")

        # Evaluate against current opponent
        current_opponent = ensemble_opponents[-1] if opponent_mode == "ensemble" else heuristic_opponent
        win_rate = evaluate_agent(model, current_opponent)
        print(f"Evaluation win rate: {win_rate*100:.2f}%")

        # If agent is dominating heuristic opponent, switch to ensemble mode
        if opponent_mode == "heuristic" and win_rate >= 0.9:
            print("Switching to self-play with ensemble of past models.")
            opponent_mode = "ensemble"
            ensemble_opponents.append(convert_to_cpu(model.get_parameters()))
            opponent_index = 0

        elif opponent_mode == "ensemble":
            # Save current model as checkpoint
            checkpoint_path = f"models/ppo_connect4_{timesteps_so_far}.zip"
            os.makedirs("models", exist_ok=True)
            model.save(checkpoint_path)
            ensemble_opponents.append(convert_to_cpu(model.get_parameters()))
            opponent_index = (opponent_index + 1) % len(ensemble_opponents)

        # Update envs with new opponent
        new_opponent = ensemble_opponents[opponent_index] if opponent_mode == "ensemble" else heuristic_opponent
        env.set_attr('opponent_params', new_opponent)
        print("Updated opponent parameters.")

    model.save("ppo_connect4_final")
    return model

if __name__ == "__main__":
    trained_model = train_self_play(total_timesteps=120000, update_interval=2000)
    import game
    game.run_game(trained_model)
