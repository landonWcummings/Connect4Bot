import numpy as np
from stable_baselines3 import PPO
from connect4_env import Connect4Env
from stable_baselines3.common.env_util import DummyVecEnv
import time
import os

N_GAMES = 200
WIN_THRESHOLD = 0.85
EVAL_FREQ = 1000
MODEL_SAVE_DIR = "models_sequential"
FINAL_TRAIN_DURATION_SECS = 2 * 60 * 60  # 2 hours

# === Flatten observation ===
def flatten_obs(board):
    obs = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.int8)
    obs[0] = (board == 1).astype(np.int8)
    obs[1] = (board == -1).astype(np.int8)
    obs[2] = (board == 0).astype(np.int8)
    return obs.flatten().astype(np.float32)

# === Opponent policies ===
def random_opponent(board, env):
    return int(np.random.choice(env.get_valid_moves(board)))

def block4_opponent(board, env):
    valid = env.get_valid_moves(board)
    for col in valid:
        temp = env.drop_piece(board, col, 1)
        if env.check_win(1, temp):
            return col
    return int(np.random.choice(valid))

def win_block_opponent(board, env):
    valid = env.get_valid_moves(board)
    for col in valid:
        temp = env.drop_piece(board, col, -1)
        if env.check_win(-1, temp):
            return col
    return block4_opponent(board, env)

def three_win_block_opponent(board, env):
    valid = env.get_valid_moves(board)
    for col in valid:
        temp = env.drop_piece(board, col, -1)
        if env.check_win(-1, temp):
            return col
    return block4_opponent(board, env)

# === Evaluation function ===
def evaluate(model, opponent_fn):
    wins = 0
    env = Connect4Env()
    for _ in range(N_GAMES):
        board = np.zeros((env.rows, env.cols), dtype=np.int8)
        done = False
        current = 1
        while not done:
            if current == 1:
                obs = flatten_obs(board)
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = opponent_fn(board, env)
            board = env.drop_piece(board, int(action), current)
            if env.check_win(current, board):
                winner = current
                done = True
            elif not env.get_valid_moves(board):
                winner = 0
                done = True
            current *= -1
        if winner == 1:
            wins += 1
    return wins / N_GAMES

# === Trainable environment wrapper ===
class Connect4TrainEnv(Connect4Env):
    def __init__(self, opponent_fn):
        super().__init__()
        self.opponent_fn = opponent_fn
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        return flatten_obs(self.board), {}  # Return tuple (obs, info)

    def step(self, action):
        done = False
        reward = 0.0
        truncated = False  # Set this to False by default, or True if there's a reason to truncate

        # Agent move
        if action in self.get_valid_moves(self.board):
            self.board = self.drop_piece(self.board, int(action), 1)
        else:
            return flatten_obs(self.board), -1.0, True, False, {}

        if self.check_win(1, self.board):
            return flatten_obs(self.board), 1.0, True, False, {}

        if not self.get_valid_moves(self.board):
            return flatten_obs(self.board), 0.0, True, False, {}

        # Opponent move
        opp_action = self.opponent_fn(self.board, self)
        self.board = self.drop_piece(self.board, int(opp_action), -1)

        if self.check_win(-1, self.board):
            return flatten_obs(self.board), -1.0, True, False, {}

        if not self.get_valid_moves(self.board):
            return flatten_obs(self.board), 0.0, True, False, {}

        return flatten_obs(self.board), 0.0, False, truncated, {}

# === Main training loop ===
if __name__ == "__main__":
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    opponents = [
        ("Random", random_opponent),
        ("Block4", block4_opponent),
        ("WinOrBlock4", win_block_opponent),
        ("ThreeWinBlock4", three_win_block_opponent)
    ]

    model = PPO("MlpPolicy", DummyVecEnv([lambda: Connect4TrainEnv(opponents[0][1])]), verbose=1)

    for i, (name, opponent_fn) in enumerate(opponents):
        print(f"\n=== Training vs {name} opponent ===")
        model.set_env(DummyVecEnv([lambda: Connect4TrainEnv(opponent_fn)]))
        win_rate = 0.0
        total_timesteps = 0

        while win_rate < WIN_THRESHOLD:
            model.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)
            win_rate = evaluate(model, opponent_fn)
            total_timesteps += EVAL_FREQ
            print(f"  -> {total_timesteps} timesteps, win rate vs {name}: {win_rate*100:.1f}%")

        model.save(f"{MODEL_SAVE_DIR}/ppo_vs_{name.lower()}.zip")
        print(f"✓ Reached {win_rate*100:.1f}% win rate vs {name}. Model saved.\n")

    # === Final 2-hour training after all levels passed ===
    print("=== Final Training vs Top Opponent (2 hours) ===")
    start_time = time.time()
    top_opponent_fn = opponents[-1][1]
    model.set_env(DummyVecEnv([lambda: Connect4TrainEnv(top_opponent_fn)]))

    while time.time() - start_time < FINAL_TRAIN_DURATION_SECS:
        model.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)
        elapsed = (time.time() - start_time) / 60
        print(f"  -> Trained for {elapsed:.1f} minutes...")

    model.save(f"{MODEL_SAVE_DIR}/ppo_final_2h.zip")
    print("✓ Final training complete. Model saved.")
