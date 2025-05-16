import numpy as np
from stable_baselines3 import PPO
from connect4_env import Connect4Env
from stable_baselines3.common.env_util import DummyVecEnv
import os
import math
from collections import OrderedDict
import argparse

N_GAMES = 200
EVAL_FREQ = 1000
MODEL_SAVE_DIR = "models_sequential"
CACHE_SIZE = 100000  # Limit cache size for board evaluations
MPATH = r"C:\Users\lando\Desktop\AI\Connect4Bot\models_sequential\ppo_vs_minimax_m3.25.zip"

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
    return board.tobytes()

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
    if not valid_moves:
        return None
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
        move = choose_move(board, env, depth)
        if move is None:
            valid_moves = env.get_valid_moves(board)
            return int(np.random.choice(valid_moves)) if valid_moves else None
        return move
    return minimax_opponent

# === Opponent policies ===
def random_opponent(board, env):
    valid_moves = env.get_valid_moves(board)
    return int(np.random.choice(valid_moves)) if valid_moves else None

def block4_opponent(board, env):
    valid = env.get_valid_moves(board)
    for col in valid:
        temp = env.drop_piece(board, col, 1)
        if env.check_win(1, temp):
            return col
    return int(np.random.choice(valid)) if valid else None

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
def evaluate(model, opponent_fn, opponent_name="Opponent"):
    wins = 0
    env = Connect4Env()
    for i in range(N_GAMES):
        board = np.zeros((env.rows, env.cols), dtype=np.int8)
        done = False
        current = 1
        while not done:
            if current == 1:
                obs = flatten_obs(board)
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = opponent_fn(board, env)
                if action is None:
                    print(f"Warning: {opponent_name} returned None action at game {i+1}. Valid moves: {env.get_valid_moves(board)}")
                    valid_moves = env.get_valid_moves(board)
                    action = int(np.random.choice(valid_moves)) if valid_moves else None
            if action is None:
                print(f"Game {i+1} terminated early due to no valid moves.")
                winner = 0
                done = True
                break
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
    win_rate = wins / N_GAMES
    print(f"Evaluated {N_GAMES} games against {opponent_name}: {wins} wins, {win_rate*100:.1f}% win rate")
    return win_rate

# === Single-opponent training environment ===
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
        if opp_action is None:
            valid_moves = self.get_valid_moves(self.board)
            opp_action = int(np.random.choice(valid_moves)) if valid_moves else None
        if opp_action is None:
            return flatten_obs(self.board), -1.0, True, False, {}
        self.board = self.drop_piece(self.board, int(opp_action), -1)

        if self.check_win(-1, self.board):
            return flatten_obs(self.board), -1.0, True, False, {}

        if not self.get_valid_moves(self.board):
            return flatten_obs(self.board), 0.0, True, False, {}

        return flatten_obs(self.board), 0.0, False, truncated, {}

# === Mixed training environment ===
class Connect4MixedTrainEnv(Connect4Env):
    def __init__(self, opponent_fns, weights=None):
        super().__init__()
        self.opponent_fns = opponent_fns  # List of opponent functions
        self.weights = weights if weights else [1.0 / len(opponent_fns)] * len(opponent_fns)  # Equal weights
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.current_opponent = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.current_opponent = np.random.choice(self.opponent_fns, p=self.weights)
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
        opp_action = self.current_opponent(self.board, self)
        if opp_action is None:
            valid_moves = self.get_valid_moves(self.board)
            opp_action = int(np.random.choice(valid_moves)) if valid_moves else None
        if opp_action is None:
            return flatten_obs(self.board), -1.0, True, False, {}
        self.board = self.drop_piece(self.board, int(opp_action), -1)

        if self.check_win(-1, self.board):
            return flatten_obs(self.board), -1.0, True, False, {}

        if not self.get_valid_moves(self.board):
            return flatten_obs(self.board), 0.0, True, False, {}

        # Select new opponent for next step
        self.current_opponent = np.random.choice(self.opponent_fns, p=self.weights)
        return flatten_obs(self.board), 0.0, False, truncated, {}

# === Main training loop ===
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a PPO model for Connect4, optionally loading a pre-trained model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pre-trained PPO model (e.g., ppo_vs_minimax_m1.00.zip). If provided, skips heuristic phase and starts at Minimax stage.")
    args = parser.parse_args()

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Heuristic opponents
    opponents = [
        ("Random", random_opponent),
        ("Block4", block4_opponent),
        ("WinOrBlock4", win_block_opponent),
        ("ThreeWinBlock4", three_win_block_opponent)
    ]
    heuristic_phase_thresholds = [0.85, 0.85, 0.85, 0.90]  # 90% for ThreeWinBlock4
    minimax_stage_heuristic_threshold = 0.77  # 77% for WinOrBlock4 and ThreeWinBlock4 in Minimax stage
    minimax_threshold = 0.70  # 70% for Minimax

    # Initialize or load model
    if args.model_path:
        print(f"Loading pre-trained model from {MPATH}...")
        model = PPO.load(MPATH, env=DummyVecEnv([lambda: Connect4MixedTrainEnv(
            [get_minimax_opponent(1.0), win_block_opponent, three_win_block_opponent], [1/3, 1/3, 1/3]
        )]), verbose=1, device='cpu')
        start_minimax = True
    else:
        print("Initializing new model...")
        model = PPO("MlpPolicy", DummyVecEnv([lambda: Connect4TrainEnv(opponents[0][1])]), verbose=1, device='cpu')
        start_minimax = False

    # Train against heuristic opponents (if no model path provided)
    if not start_minimax:
        for i, (name, opponent_fn) in enumerate(opponents):
            print(f"\n=== Training vs {name} opponent ===")
            model.set_env(DummyVecEnv([lambda: Connect4TrainEnv(opponent_fn)]))
            win_rate = 0.0
            total_timesteps = 0
            threshold = heuristic_phase_thresholds[i]

            while win_rate < threshold:
                model.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)
                win_rate = evaluate(model, opponent_fn, name)
                total_timesteps += EVAL_FREQ
                print(f"  -> {total_timesteps} timesteps, win rate vs {name}: {win_rate*100:.1f}%")

            model.save(f"{MODEL_SAVE_DIR}/ppo_vs_{name.lower()}.zip")
            print(f"✓ Reached {win_rate*100:.1f}% win rate vs {name}. Model saved.\n")

    # Minimax stage: Train against mix of Minimax, WinOrBlock4, and ThreeWinBlock4
    print("=== Minimax Stage: Training vs Minimax, WinOrBlock4, and ThreeWinBlock4 ===")
    multiplier = 3.2
    heuristic_opponents = [win_block_opponent, three_win_block_opponent]
    heuristic_names = ["WinOrBlock4", "ThreeWinBlock4"]

    while True:
        # Initialize mixed training with equal weights
        opponent_fns = [get_minimax_opponent(multiplier), win_block_opponent, three_win_block_opponent]
        opponent_names = ["Minimax", "WinOrBlock4", "ThreeWinBlock4"]
        weights = [1/3, 1/3, 1/3]  # Equal probability for each opponent
        model.set_env(DummyVecEnv([lambda: Connect4MixedTrainEnv(opponent_fns, weights)]))
        
        while True:
            # Train for EVAL_FREQ timesteps
            model.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)
            
            # Evaluate against all opponents
            minimax_win_rate = evaluate(model, opponent_fns[0], opponent_names[0])
            heuristic_win_rates = [evaluate(model, opp, name) for opp, name in zip(heuristic_opponents, heuristic_names)]
            
            print(f"  -> Multiplier: {multiplier:.2f}")
            print(f"     Win rate vs {opponent_names[0]}: {minimax_win_rate*100:.1f}%")
            for name, win_rate in zip(heuristic_names, heuristic_win_rates):
                print(f"     Win rate vs {name}: {win_rate*100:.1f}%")
            
            # Check thresholds
            heuristics_pass = all(win_rate >= minimax_stage_heuristic_threshold for win_rate in heuristic_win_rates)
            minimax_pass = minimax_win_rate >= minimax_threshold
            
            if heuristics_pass and minimax_pass:
                # All thresholds met: save model and increase multiplier
                model.save(f"{MODEL_SAVE_DIR}/ppo_vs_minimax_m{multiplier:.2f}.zip")
                multiplier += 0.05
                print(f"  -> All thresholds met (Heuristics >=77%, Minimax >=70%). Increased multiplier to {multiplier:.2f}")
                break
            else:
                # Some thresholds failed: train only against failing opponents
                failing_opponents = []
                failing_weights = []
                failing_names = []
                if not minimax_pass:
                    failing_opponents.append(opponent_fns[0])
                    failing_weights.append(1.0)
                    failing_names.append(opponent_names[0])
                if not heuristics_pass:
                    for opp, win_rate, name in zip(heuristic_opponents, heuristic_win_rates, heuristic_names):
                        if win_rate < minimax_stage_heuristic_threshold:
                            failing_opponents.append(opp)
                            failing_weights.append(1.0)
                            failing_names.append(name)
                
                if not failing_opponents:
                    print("Warning: No failing opponents detected despite threshold failure. Forcing Minimax training.")
                    failing_opponents = [opponent_fns[0]]
                    failing_weights = [1.0]
                    failing_names = [opponent_names[0]]
                
                # Normalize weights
                total_weight = sum(failing_weights)
                failing_weights = [w / total_weight for w in failing_weights]
                
                print(f"  -> Training against failing opponents: {failing_names}")
                model.set_env(DummyVecEnv([lambda: Connect4MixedTrainEnv(failing_opponents, failing_weights)]))