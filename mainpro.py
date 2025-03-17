import time
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from connect4_env import Connect4Env

def make_env():
    def _init():
        return Connect4Env()
    return _init

def convert_to_cpu(params):
    """Recursively convert tensors in the params dict to CPU."""
    if isinstance(params, dict):
        return {k: convert_to_cpu(v) for k, v in params.items()}
    elif hasattr(params, "detach"):
        return params.detach().cpu()
    else:
        return params

class WinRateCallback(BaseCallback):
    """
    A custom callback that approximates win rate using rewards from the rollout buffer.
    Assumes that a reward > 0.5 indicates a win.
    Note: The rollout buffer may include intermediate rewards, so this win rate is only approximate.
    """
    def __init__(self, verbose=0):
        super(WinRateCallback, self).__init__(verbose)
        self.win_rates = []

    def _on_step(self) -> bool:
        return True

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
    """
    Evaluate the model's performance on a fresh Connect4Env over num_episodes.
    If opponent_params is provided, use that for the opponent; otherwise, the opponent is random.
    Returns the win rate (fraction of wins) for the agent.
    A win is indicated by a final reward >= 1.
    """
    wins = 0
    env = Connect4Env()
    env.set_opponent_params(opponent_params)
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if reward >= 1:
            wins += 1
    env.close()
    win_rate = wins / num_episodes
    return win_rate

def train_self_play(total_timesteps=100000, update_interval=2000, eval_threshold=0.55, max_hours=1, model=None):
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # Start a new model from scratch if not provided.
    if model is None:
        # Use a small entropy coefficient for extra exploration.
        model = PPO("MlpPolicy", env, verbose=1, device="cuda", ent_coef=0.01)
    else:
        model.set_env(env)
    
    # Initially, no opponent parameters are set.
    env.set_attr('opponent_params', None)
    
    win_rate_callback = WinRateCallback(verbose=1)
    
    timesteps_so_far = 0
    games_since_update = 0  # Count of evaluation games since last ensemble update
    ensemble = []  # List to hold up to 4 old opponent models (their CPU parameters)
    
    start_time = time.time()
    max_seconds = max_hours * 3600
    
    while timesteps_so_far < total_timesteps and (time.time() - start_time) < max_seconds:
        # If we have an ensemble, set the environment's opponent to a random member.
        if ensemble:
            chosen = np.random.choice(ensemble)
            env.env_method('set_opponent_params', chosen)
        else:
            # Otherwise, keep opponent random.
            env.env_method('set_opponent_params', None)
        
        model.learn(total_timesteps=update_interval, callback=win_rate_callback)
        timesteps_so_far += update_interval
        print(f"Trained {timesteps_so_far} timesteps")
        
        # Evaluate the current model over 200 games against the current ensemble opponent.
        # If ensemble is nonempty, average evaluations over each opponent in the ensemble.
        if ensemble:
            win_rates = []
            for opp in ensemble:
                wr = evaluate_agent(model, opp, num_episodes=100)
                win_rates.append(wr)
            avg_win_rate = np.mean(win_rates)
            print(f"Average evaluation win rate against ensemble: {avg_win_rate*100:.2f}%")
        else:
            avg_win_rate = evaluate_agent(model, None, num_episodes=200)
            print(f"Evaluation win rate over 200 games (random opponent): {avg_win_rate*100:.2f}%")
        
        games_since_update += 200
        print(f"Games since last ensemble update: {games_since_update}")
        
        # If we've played at least 200 evaluation games since the last update and the new model
        # beats the current ensemble (average win rate >= eval_threshold), update the ensemble.
        if games_since_update >= 200 and avg_win_rate >= eval_threshold:
            new_params = convert_to_cpu(model.get_parameters())
            # Append new snapshot to ensemble.
            ensemble.append(new_params)
            # Keep ensemble size to 4.
            if len(ensemble) > 4:
                ensemble.pop(0)
            print("Ensemble updated with new opponent model. Ensemble size:", len(ensemble))
            games_since_update = 0  # Reset the counter
        else:
            print("Ensemble not updated.")
    
    model.save("connect4_ensemble_master")
    return model

if __name__ == "__main__":
    # Option 1: Train a new model from scratch.
    #trained_model = train_self_play(total_timesteps=12000000, update_interval=600, eval_threshold=0.55, max_hours=20)
    
    # Option 2: Continue training from a pre-trained model.
    # ------------------------------------------------------
    from stable_baselines3 import PPO
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    trained_model = PPO.load(r"C:\Users\lndnc\OneDrive\Desktop\AI\connect4\connect4_ensemble_master.zip", env=env, device="cuda")
    trained_model = train_self_play(total_timesteps=99000000, update_interval=666, eval_threshold=0.51, max_hours=21, model=trained_model)
    
    import game
    game.run_game(trained_model)

