import numpy as np
from stable_baselines3 import PPO
from connect4_env import Connect4Env
from stable_baselines3.common.env_util import DummyVecEnv
import os
import math
import argparse
from collections import OrderedDict

N_GAMES = 200
EVAL_FREQ = 1000
MODEL_SAVE_DIR = "models_sequential"
CACHE_SIZE = 100000  # Limit cache size to prevent memory bloat

# === Bounded cache for board evaluations ===
class BoundedCache:
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

board_cache = BoundedCache(CACHE_SIZE)

def hash_board(board):
    """Generate a unique hash for a board state."""
    return board.tobytes()  # Faster than MD5 for simple caching

# === Flatten observation ===
def flatten_obs(board):
    """Optimized observation flattening."""
    obs = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.int8)
    obs[0] = (board == 1)
    obs[1] = (board == -1)
    obs[2] = (board == 0)
    return obs.ravel().astype(np.float32)

# === Precompute winning lines for a 6x7 board ===
winning_lines = []
for row in range(6):
    for col in range(4):
        winning_lines.append([(row, col), (row, col+1), (row, col+2), (row, col+3)])
for row in range(3):
    for col in range(7):
        winning_lines.append([(row, col), (row+1, col), (row+2, col), (row+3, col)])
for row in range(3, 6):
    for col in range(4):
        winning_lines.append([(row, col), (row-1, col+1), (row-2, col+2), (row-3, col+3)])
for row in range(3):
    for col in range(4):
        winning_lines.append([(row, col), (row+1, col+1), (row+2, col+2), (row+3, col+3)])

# === Board evaluation for Minimax ===
def evaluate_board(board, player):
    """Optimized board evaluation with caching."""
    board_hash = hash_board(board)
    if board_hash in board_cache:
        return board_cache[board_hash]
    
    score = 0
    opp = -player
    for line in winning_lines:
        values = np.array([board[r, c] for r, c in line])
        my_pieces = np.sum(values == player)
        opp_pieces = np.sum(values == opp)
        empty = np.sum(values == 0)
        if my_pieces == 4:
            score += 1000000
        elif opp_pieces == 4:
            score -= 1000000
        elif my_pieces == 3 and empty == 1:
            score += 100
        elif opp_pieces == 3 and empty == 1:
            score -= 100
        elif my_pieces == 2 and empty == 2:
            score += 10
        elif opp_pieces == 2 and empty == 2:
            score -= 10
    # Center control bonus
    center_col = 3
    score += 5 * np.sum(board[:, center_col] == player)
    score -= 5 * np.sum(board[:, center_col] == opp)
    
    board_cache[board_hash] = score
    return score

# === Threat detection for move ordering ===
def get_threats(board, env, player):
    """Vectorized threat detection for move ordering."""
    threats = []
    valid_moves = env.get_valid_moves(board)
    for col in valid_moves:
        temp = env.drop_piece(board, col, player)
        score = 0
        for line in winning_lines:
            values = np.array([temp[r, c] for r, c in line])
            if np.sum(values == player) == 3 and np.sum(values == 0) == 1:
                score = 100  # Threat creation
                break
            elif np.sum(values == -player) == 3 and np.sum(values == 0) == 1:
                score = 75  # Threat blocking
        threats.append((col, score))
    return threats

# === Alpha-Beta Pruning Minimax algorithm ===
def minimax(board, depth, is_maximizing, env, alpha, beta):
    if env.check_win(1, board):
        return 1000000 if is_maximizing else -1000000
    if env.check_win(-1, board):
        return -1000000 if is_maximizing else 1000000
    if not env.get_valid_moves(board):
        return 0
    if depth == 0:
        return evaluate_board(board, 1)
    
    valid_moves = env.get_valid_moves(board)
    # Move ordering: prioritize threats, center, others
    threats = get_threats(board, env, 1 if is_maximizing else -1)
    move_priority = sorted(
        [(move, score + (50 if move == 3 and score == 0 else 0)) for move, score in threats],
        key=lambda x: x[1],
        reverse=True
    )
    move_priority = [move for move, _ in move_priority]
    
    if is_maximizing:
        max_eval = -np.inf
        for move in move_priority:
            new_board = env.drop_piece(board, move, 1)
            eval = minimax(new_board, depth - 1, False, env, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in move_priority:
            new_board = env.drop_piece(board, move, -1)
            eval = minimax(new_board, depth - 1, True, env, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# === Choose move for Minimax opponent ===
def choose_move(board, env, depth):
    if depth <= 0:
        raise ValueError("Depth must be at least 1")
    valid_moves = env.get_valid_moves(board)
    # Move ordering: prioritize threats, center, others
    threats = get_threats(board, env, -1)
    move_priority = sorted(
        [(move, score + (50 if move == 3 and score == 0 else 0)) for move, score in threats],
        key=lambda x: x[1],
        reverse=True
    )
    move_priority = [move for move, _ in move_priority]
    
    best_score = np.inf
    best_moves = []
    for move in move_priority:
        new_board = env.drop_piece(board, move, -1)
        if env.check_win(-1, new_board):
            return move  # Immediate win
        score = minimax(new_board, depth - 1, True, env, -np.inf, np.inf)
        if score < best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)
    
    return np.random.choice(best_moves) if best_moves else None

# === Get depth based on multiplier ===
def get_depth(multiplier):
    k = math.floor(multiplier)
    f = multiplier - k
    if np.random.rand() < 1 - f:
        return k
    else:
        return k + 1

# === Get Minimax opponent function ===
def get_minimax_opponent(multiplier):
    def minimax_opponent(board, env):
        depth = get_depth(multiplier)
        return choose_move(board, env, depth)
    return minimax_opponent

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
        return flatten_obs(self.board), {}

    def step(self, action):
        done = False
        reward = 0.0
        truncated = False

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
    # Parse command-line argument for model path
    parser = argparse.ArgumentParser(description="Continue training a PPO model against Minimax opponent.")
    parser.add_argument("model_path", type=str, help="Path to the PPO model file (e.g., ppo_vs_minimax_m1.80.zip)")
    args = parser.parse_args()

    # Ensure model save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    path = r"C:\Users\lando\Desktop\AI\Connect4Bot\models_sequential\ppo_vs_minimax_m2.60.zip"
    # Load the specified model
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(path, env=DummyVecEnv([lambda: Connect4TrainEnv(get_minimax_opponent(1.0))]), verbose=1, device='cpu')

    # Train against Minimax with increasing multiplier
    print("=== Training vs Minimax opponent with increasing multiplier ===")
    multiplier = 1.0
    while True:
        opponent_fn = get_minimax_opponent(multiplier)
        model.set_env(DummyVecEnv([lambda: Connect4TrainEnv(opponent_fn)]))
        model.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)
        win_rate = evaluate(model, opponent_fn)
        print(f"  -> Multiplier: {multiplier:.2f}, win rate: {win_rate*100:.1f}%")
        if win_rate >= 0.72:
            model.save(f"{MODEL_SAVE_DIR}/ppo_vs_minimax_m{multiplier:.2f}.zip")
            multiplier += 0.05
            print(f"  -> Win rate >=72%, increased multiplier to {multiplier:.2f}")
        else:
            print(f"  -> Win rate <72%, continuing training")