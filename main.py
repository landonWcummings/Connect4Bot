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

def evaluate_agent(model, num_episodes=200):
    """Evaluate the model's performance on a fresh Connect4Env over num_episodes.
       Returns the win rate (fraction of wins) for the agent.
       A win is indicated by a final reward > 0.5."""
    wins = 0
    env = Connect4Env()
    if env.opponent_params is None:
        env.set_opponent_params(None)
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

def train_self_play(total_timesteps=100000, update_interval=2000, eval_threshold=0.55, model=None):
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # If no model is provided, create a new one. Otherwise, continue training the given model.
    if model is None:
        model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    else:
        # Ensure the loaded model uses the new environment.
        model.set_env(env)
    
    env.set_attr('opponent_params', None)
    
    win_rate_callback = WinRateCallback(verbose=1)
    
    timesteps_so_far = 0
    games_since_update = 0  # Count of evaluation games since last opponent update
    
    while timesteps_so_far < total_timesteps:
        model.learn(total_timesteps=update_interval, callback=win_rate_callback)
        timesteps_so_far += update_interval
        print(f"Trained {timesteps_so_far} timesteps")
        
        # Evaluate the current model over 200 games.
        win_rate = evaluate_agent(model, num_episodes=350)
        games_since_update += 200
        print(f"Evaluation win rate over 200 games: {win_rate*100:.2f}%")
        print(f"Games since last update: {games_since_update}")
        
        # Update the opponent only if at least 200 evaluation games have been played
        # and the win rate is at or above the threshold (or if no opponent is set yet).
        if (games_since_update >= 200) and (win_rate >= eval_threshold or env.get_attr('opponent_params')[0] is None):
            opponent_params = convert_to_cpu(model.get_parameters())
            env.env_method('set_opponent_params', opponent_params)
            print("Opponent parameters updated.")
            games_since_update = 0  # Reset the counter after an update
        else:
            print("Opponent not updated.")
    
    model.save("ppo_connect4_final")
    return model

if __name__ == "__main__":
    # Uncomment one of the following options:
    
    # Option 1: Train from scratch.
    # trained_model = train_self_play(total_timesteps=120000, update_interval=2000, eval_threshold=0.55)
    
    # Option 2: Continue training from a pre-trained model.
    # ------------------------------------------------------
    # from stable_baselines3 import PPO
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # # Load the model from a zip file. Ensure the environment is passed.
    trained_model = PPO.load(r"C:\Users\lndnc\OneDrive\Desktop\AI\connect4\ppo_connect4_final.zip", env=env, device="cuda")
    # # Continue training with self-play
    trained_model = train_self_play(total_timesteps=9900000, update_interval=10000, eval_threshold=0.55, model=trained_model)
    # ------------------------------------------------------
    
    # For now, using Option 1:
    #trained_model = train_self_play(total_timesteps=120000, update_interval=2000, eval_threshold=0.55)
    
    import game
    game.run_game(trained_model)
