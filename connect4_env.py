import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

class Connect4Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(Connect4Env, self).__init__()
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.action_space = spaces.Discrete(self.cols)
        # MLP-compatible observation space: flattened to (42,)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.rows * self.cols,), dtype=np.int8)
        self.done = False
        self.opponent_params = None
        self._temp_opponent_model = None

    def set_opponent_params(self, params):
        self.opponent_params = params
        self._temp_opponent_model = None

    def reset(self, **kwargs):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.done = False
        # Flatten the board for MLP
        return self.board.flatten(), {}

    def step(self, action):
        if self.done:
            return self.board.flatten(), 0, True, False, {}

        # Agent's turn (player = 1)
        if not self.is_valid_move(action):
            valid_moves = self.get_valid_moves(self.board)
            if valid_moves:
                action = np.random.choice(valid_moves)
                penalty = -0.5
            else:
                self.done = True
                return self.board.flatten(), -1, True, False, {"invalid_move": True}
        else:
            penalty = 0

        opp_pot_before = self.count_potential_wins(-1, self.board)
        agent_setup_before = self.count_setup_potential(1, self.board)
        opp_setup_before = self.count_setup_potential(-1, self.board)

        self.board = self.drop_piece(self.board, action, 1)

        reward = penalty
        if self.check_win(1, self.board):
            self.done = True
            return self.board.flatten(), reward + 1, True, False, {}

        opp_pot_after = self.count_potential_wins(-1, self.board)
        blocking_reward = (opp_pot_before - opp_pot_after) * 0.1

        agent_setup_after = self.count_setup_potential(1, self.board)
        opp_setup_after = self.count_setup_potential(-1, self.board)
        setup_reward = (agent_setup_after - agent_setup_before) * 0.1 - (opp_setup_after - opp_setup_before) * 0.1

        reward += blocking_reward + setup_reward

        if not self.get_valid_moves(self.board):
            self.done = True
            return self.board.flatten(), reward, True, False, {}

        # Opponent's turn (player = -1)
        opp_valid_moves = self.get_valid_moves(self.board)
        if opp_valid_moves:
            if self.opponent_params is not None:
                if self._temp_opponent_model is None:
                    # Use MlpPolicy with large architecture
                    self._temp_opponent_model = PPO(
                        "MlpPolicy",
                        self,
                        verbose=0,
                        policy_kwargs=dict(
                            net_arch=dict(
                                pi=[2048, 2048, 1024, 512],
                                vf=[2048, 2048, 1024, 512]
                            )
                        )
                    )
                self._temp_opponent_model.set_parameters(self.opponent_params)
                opp_action, _ = self._temp_opponent_model.predict(self.board.flatten())
                if opp_action not in opp_valid_moves:
                    opp_action = np.random.choice(opp_valid_moves)
            else:
                opp_action = np.random.choice(opp_valid_moves)
            self.board = self.drop_piece(self.board, opp_action, -1)
            if self.check_win(-1, self.board):
                self.done = True
                return self.board.flatten(), reward - 1, True, False, {}

        if not self.get_valid_moves(self.board):
            self.done = True

        return self.board.flatten(), reward, self.done, False, {}

    def is_valid_move(self, col):
        return self.board[0, col] == 0

    def get_valid_moves(self, board):
        return [c for c in range(self.cols) if board[0, c] == 0]

    def drop_piece(self, board, col, player):
        new_board = board.copy()
        for row in range(self.rows - 1, -1, -1):
            if new_board[row, col] == 0:
                new_board[row, col] = player
                break
        return new_board

    def check_win(self, player, board):
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if np.all(board[r, c:c+4] == player):
                    return True
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if np.all(board[r:r+4, c] == player):
                    return True
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(board[r+i, c+i] == player for i in range(4)):
                    return True
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(board[r-i, c+i] == player for i in range(4)):
                    return True
        return False

    def count_potential_wins(self, player, board):
        count = 0
        for move in self.get_valid_moves(board):
            temp_board = self.drop_piece(board, move, player)
            if self.check_win(player, temp_board):
                count += 1
        return count

    def count_setup_potential(self, player, board):
        count = 0
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = board[r, c:c+4]
                if np.count_nonzero(window == -player) == 0 and np.count_nonzero(window == player) > 0:
                    count += 1
        for c in range(self.cols):
            for r in range(self.rows - 3):
                window = board[r:r+4, c]
                if np.count_nonzero(window == -player) == 0 and np.count_nonzero(window == player) > 0:
                    count += 1
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = np.array([board[r+i, c+i] for i in range(4)])
                if np.count_nonzero(window == -player) == 0 and np.count_nonzero(window == player) > 0:
                    count += 1
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = np.array([board[r-i, c+i] for i in range(4)])
                if np.count_nonzero(window == -player) == 0 and np.count_nonzero(window == player) > 0:
                    count += 1
        return count

    def render(self, mode="human"):
        print(np.flip(self.board, 0))