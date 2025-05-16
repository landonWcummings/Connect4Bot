import numpy as np
from stable_baselines3 import PPO
from connect4_env import Connect4Env
import matplotlib.pyplot as plt

# Number of games per opponent
N_GAMES = 500
MODEL_PATH = r"C:\Users\lando\Desktop\AI\Connect4Bot\models_sequential\ppo_final_2h.zip"  # Path to your trained MLP model

# Load trained model (MLP expects flat obs)
model = PPO.load(MODEL_PATH, device="cpu")

def flatten_obs(board):
    # Convert 2D board to 3-channel flattened obs
    obs = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.int8)
    obs[0] = (board == 1).astype(np.int8)
    obs[1] = (board == -1).astype(np.int8)
    obs[2] = (board == 0).astype(np.int8)
    return obs.flatten().astype(np.float32)

# Opponent policies

def random_opponent(board, env):
    return int(np.random.choice(env.get_valid_moves(board)))


def block4_opponent(board, env):
    # Blocks any immediate opponent win (player 1)
    valid = env.get_valid_moves(board)
    for col in valid:
        # simulate agent winning next
        temp = env.drop_piece(board, col, 1)
        if env.check_win(1, temp):
            return col
    return int(np.random.choice(valid))


def win_block_opponent(board, env):
    valid = env.get_valid_moves(board)
    # win if able
    for col in valid:
        temp = env.drop_piece(board, col, -1)
        if env.check_win(-1, temp):
            return col
    # block agent
    return block4_opponent(board, env)


def three_win_block_opponent(board, env):
    valid = env.get_valid_moves(board)
    # win by extending 3-in-a-row
    for col in valid:
        temp = env.drop_piece(board, col, -1)
        # count if creates 4
        if env.check_win(-1, temp):
            return col
    # block agent potential 4
    return block4_opponent(board, env)

# Evaluate against each opponent
def evaluate(opponent_fn):
    wins = 0
    env = Connect4Env()
    for _ in range(N_GAMES):
        board = np.zeros((env.rows, env.cols), dtype=np.int8)
        done = False
        current = 1  # 1=agent, -1=opponent
        while not done:
            if current == 1:
                obs = flatten_obs(board)
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = opponent_fn(board, env)

            board = env.drop_piece(board, action, current)
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

if __name__ == "__main__":
    opponents = [
        ("Random", random_opponent),
        ("Block4", block4_opponent),
        ("WinOrBlock4", win_block_opponent),
        ("ThreeWinBlock4", three_win_block_opponent)
    ]

    results = []
    for name, fn in opponents:
        print(f"Evaluating vs {name}...")
        rate = evaluate(fn)
        print(f"Win rate vs {name}: {rate*100:.1f}%")
        results.append(rate)

    # Plot results
    labels = [name for name, _ in opponents]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, results)
    plt.ylabel("Win rate")
    plt.ylim(0, 1)
    plt.title("Agent Win Rate vs Different Opponents")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
