import numpy as np
import math
import random
from collections import defaultdict
from stable_baselines3 import PPO
from connect4_env import Connect4Env
import matplotlib.pyplot as plt

# Number of games per opponent per role
N_GAMES_PER_ROLE = 250
MODEL_PATH = r"C:\Users\lando\Desktop\AI\Connect4Bot\models_sequential\ppo_vs_minimax_m1.45.zip"
BOT_PATHS = [
    r"C:\Users\lando\Desktop\AI\Connect4Bot\models_sequential\ppo_vs_block4.zip",
    r"C:\Users\lando\Desktop\AI\Connect4Bot\models_sequential\ppo_vs_threewinblock4.zip"
]
BOT_NAMES = ["PPOBot_Block4", "PPOBot_ThreeWinBlock4"]
MCTS_SIMULATIONS = 500  # Number of MCTS playouts per move

# Load main model and bot models
main_model = PPO.load(MODEL_PATH, device="cpu")
bot_models = [PPO.load(path, device="cpu") for path in BOT_PATHS]

# MCTS Node definition
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state.copy()
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = {}

    def is_fully_expanded(self, valid_moves):
        return set(valid_moves) <= set(self.children.keys())

    def best_child(self, c_param=1.4):
        choices = []
        for move, child in self.children.items():
            uct = (child.wins / child.visits) + c_param * math.sqrt(
                math.log(self.visits) / child.visits
            )
            choices.append((uct, move, child))
        return max(choices, key=lambda x: x[0])[2]
# Helper: flatten observation

def flatten_obs(board):
    # Convert 2D board to 3-channel flattened obs
    obs = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.int8)
    obs[0] = (board == 1).astype(np.int8)
    obs[1] = (board == -1).astype(np.int8)
    obs[2] = (board == 0).astype(np.int8)
    return obs.flatten().astype(np.float32)

# MCTS implementation
def mcts_select(node, env):
    while True:
        valid_moves = env.get_valid_moves(node.state)
        if not valid_moves:
            return node
        if not node.is_fully_expanded(valid_moves):
            return mcts_expand(node, env)
        else:
            node = node.best_child()


def mcts_expand(node, env):
    valid_moves = env.get_valid_moves(node.state)
    untried = [m for m in valid_moves if m not in node.children]
    move = random.choice(untried)
    new_state = env.drop_piece(node.state, move, -1)
    child = MCTSNode(new_state, parent=node, move=move)
    node.children[move] = child
    return child


def mcts_simulate(node, env):
    sim_state = node.state.copy()
    current = 1  # opponent just moved, so simulating agent's turn
    while True:
        valid_moves = env.get_valid_moves(sim_state)
        if not valid_moves:
            return 0  # draw
        move = random.choice(valid_moves)
        sim_state = env.drop_piece(sim_state, move, current)
        if env.check_win(current, sim_state):
            return 1 if current == -1 else -1
        current *= -1


def mcts_backpropagate(node, result):
    while node is not None:
        node.visits += 1
        if node.parent is None or node.move is not None:
            # result is from perspective of MCTS player (-1)
            node.wins += (1 if result == 1 else 0)
        node = node.parent


def mcts_opponent(board, env, simulations=MCTS_SIMULATIONS):
    root = MCTSNode(board)
    for _ in range(simulations):
        node = mcts_select(root, env)
        result = mcts_simulate(node, env)
        mcts_backpropagate(node, result)
    # pick move with highest visit count
    moves_visits = [(child.visits, move) for move, child in root.children.items()]
    if not moves_visits:
        return int(random.choice(env.get_valid_moves(board)))
    _, best_move = max(moves_visits)
    return best_move

# Existing opponent policies and wrappers
def random_opponent(board, env):
    return int(random.choice(env.get_valid_moves(board)))

def block4_opponent(board, env):
    valid = env.get_valid_moves(board)
    for col in valid:
        temp = env.drop_piece(board, col, 1)
        if env.check_win(1, temp):
            return col
    return int(random.choice(valid))

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

def ppo_bot_opponent(board, env, bot_model):
    obs = flatten_obs(board)
    action, _ = bot_model.predict(obs, deterministic=True)
    action = int(action)
    valid = env.get_valid_moves(board)
    if action not in valid:
        action = int(random.choice(valid))
    return action

# Evaluate against each opponent

def evaluate(opponent_fn, agent_first=True):
    wins = 0
    env = Connect4Env()
    for _ in range(N_GAMES_PER_ROLE):
        board = np.zeros((env.rows, env.cols), dtype=np.int8)
        done = False
        current = 1 if agent_first else -1
        while not done:
            if current == 1:
                obs = flatten_obs(board)
                action, _ = main_model.predict(obs, deterministic=True)
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
    return wins / N_GAMES_PER_ROLE

if __name__ == "__main__":
    opponents = [
        ("Random", random_opponent),
        ("Block4", block4_opponent),
        ("WinOrBlock4", win_block_opponent),
        ("ThreeWinBlock4", three_win_block_opponent),
        (BOT_NAMES[0], lambda b, e: ppo_bot_opponent(b, e, bot_models[0])),
        (BOT_NAMES[1], lambda b, e: ppo_bot_opponent(b, e, bot_models[1])),
        ("MCTS_Opponent", lambda b, e: mcts_opponent(b, e, MCTS_SIMULATIONS))
    ]

    results_first, results_second, results_avg = [], [], []
    for name, fn in opponents:
        print(f"Evaluating vs {name}...")
        rate_first = evaluate(fn, agent_first=True)
        rate_second = evaluate(fn, agent_first=False)
        rate_avg = (rate_first + rate_second) / 2
        print(f"Win rate vs {name} (Agent First): {rate_first*100:.1f}%")
        print(f"Win rate vs {name} (Agent Second): {rate_second*100:.1f}%")
        print(f"Average win rate vs {name}: {rate_avg*100:.1f}%")
        results_first.append(rate_first)
        results_second.append(rate_second)
        results_avg.append(rate_avg)

    # Plot results
    labels = [name for name, _ in opponents]
    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, results_first, width, label='Agent First')
    bars2 = ax.bar(x, results_second, width, label='Agent Second')
    bars3 = ax.bar(x + width, results_avg, width, label='Average')

    ax.set_ylabel('Win Rate')
    ax.set_title('PPO_Final Win Rate vs Different Opponents including MCTS')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('connect4_mcts_win_rates.png', dpi=300)
    plt.show()
