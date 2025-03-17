import time
import gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from connect4_env import Connect4Env
from torch.utils.tensorboard import SummaryWriter

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

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir="./tensorboard_logs/")
        self.step_counter = 0

    def _on_step(self) -> bool:
        self.step_counter += 1
        # Every 100,000 steps, evaluate against random mover
        if self.step_counter % 100000 == 0:
            win_rate = evaluate_agent(self.model, opponent_params=None, num_episodes=100)
            self.writer.add_scalar("WinRate/RandomOpponent", win_rate, self.step_counter)
            print(f"[Tensorboard] Win rate against random mover at {self.step_counter} steps: {win_rate*100:.2f}%")
        return True

    def _on_training_end(self):
        self.writer.close()

def evaluate_agent(model, opponent_params, num_episodes=200):
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

def train_self_play(total_timesteps=100000, update_interval=2000, eval_threshold=0.55, max_hours=12, model_path="connect4_ensemble_master"):
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # Define the large MLP policy architecture
    large_policy_kwargs = dict(
        net_arch=dict(
            pi=[2048, 2048, 1024, 512],  # Policy network
            vf=[2048, 2048, 1024, 512]   # Value network
        )
    )
    
    # Load model if it exists, otherwise create a new one
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}.zip")
        model = PPO.load(model_path, env=env, device="cuda")
        # Optionally reset the learning rate or other hyperparameters if needed
        model.learning_rate = 3e-4  # Default PPO learning rate, adjust if desired
    else:
        print("Starting new model training")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            ent_coef=0.01,
            policy_kwargs=large_policy_kwargs,
            tensorboard_log="./tensorboard_logs/"
        )
    
    env.set_attr('opponent_params', None)
    
    # Combine callbacks
    win_rate_callback = WinRateCallback(verbose=1)
    tensorboard_callback = TensorboardCallback(verbose=1)
    callbacks = CallbackList([win_rate_callback, tensorboard_callback])
    
    timesteps_so_far = 0
    games_since_update = 0
    ensemble = []
    
    start_time = time.time()
    max_seconds = max_hours * 3600
    
    while timesteps_so_far < total_timesteps and (time.time() - start_time) < max_seconds:
        if ensemble:
            chosen = np.random.choice(ensemble)
            env.env_method('set_opponent_params', chosen)
        else:
            env.env_method('set_opponent_params', None)
        
        model.learn(total_timesteps=update_interval, callback=callbacks, reset_num_timesteps=False)
        timesteps_so_far += update_interval
        print(f"Trained {timesteps_so_far} timesteps")
        
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
        
        if games_since_update >= 200 and avg_win_rate >= eval_threshold:
            new_params = convert_to_cpu(model.get_parameters())
            ensemble.append(new_params)
            if len(ensemble) > 4:
                ensemble.pop(0)
            print("Ensemble updated with new opponent model. Ensemble size:", len(ensemble))
            games_since_update = 0
        else:
            print("Ensemble not updated.")
        
        # Save the model after each update interval
        model.save(model_path)
    
    model.save(model_path)
    return model

if __name__ == "__main__":
    # Resume or start training for 12 hours
    trained_model = train_self_play(
        total_timesteps=100000000,
        update_interval=2000,
        eval_threshold=0.53,
        max_hours=12,
        model_path="connect4_ensemble_master"
    )
    
    import game
    game.run_game(trained_model)