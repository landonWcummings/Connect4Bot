import gymnasium as gym
import numpy as np
from gymnasium import spaces

class Connect4Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(Connect4Env, self).__init__()
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.action_space = spaces.Discrete(self.cols)
        # Flattened observation space for MLP
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.rows * self.cols * 3,), dtype=np.int8)
        self.done = False
        self.opponent_params = None
        self._temp_opponent_model = None

    def set_opponent_params(self, params):
        self.opponent_params = params
        self._temp_opponent_model = None

    def reset(self, **kwargs):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.done = False
        return self.get_observation(), {}

    def get_observation(self):
        obs = np.zeros((3, self.rows, self.cols), dtype=np.int8)
        obs[0] = (self.board == 1).astype(np.int8)
        obs[1] = (self.board == -1).astype(np.int8)
        obs[2] = (self.board == 0).astype(np.int8)
        return obs.flatten()

    def is_valid_move(self, col):
        return 0 <= col < self.cols and self.board[0, col] == 0

    def get_valid_moves(self, board):
        return [col for col in range(self.cols) if board[0, col] == 0]

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
                if all(board[r, c + i] == player for i in range(4)):
                    return True
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if all(board[r + i, c] == player for i in range(4)):
                    return True
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(board[r + i, c + i] == player for i in range(4)):
                    return True
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(board[r - i, c + i] == player for i in range(4)):
                    return True
        return False

    def count_potential_wins(self, player, board):
        count = 0
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = board[r, c:c + 4]
                if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                    count += 1
        for r in range(self.rows - 3):
            for c in range(self.cols):
                window = board[r:r + 4, c]
                if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                    count += 1
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [board[r + i, c + i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    count += 1
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = [board[r - i, c + i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    count += 1
        return count

    def count_almost_wins(self, player, board):
        return self.count_potential_wins(player, board)

    def count_setup_potential(self, player, board):
        count = 0
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = board[r, c:c + 4]
                if np.sum(window == player) == 2 and np.sum(window == 0) == 2:
                    count += 1
        return count

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        if not self.is_valid_move(action):
            valid_moves = self.get_valid_moves(self.board)
            if valid_moves:
                action = np.random.choice(valid_moves)
                penalty = -0.5
            else:
                self.done = True
                return self.get_observation(), -1, True, False, {"invalid_move": True}
        else:
            penalty = 0

        opp_pot_before = self.count_potential_wins(-1, self.board)
        agent_setup_before = self.count_setup_potential(1, self.board)
        opp_setup_before = self.count_setup_potential(-1, self.board)
        agent_almost_wins_before = self.count_almost_wins(1, self.board)
        opp_almost_wins_before = self.count_almost_wins(-1, self.board)

        self.board = self.drop_piece(self.board, action, 1)

        reward = penalty
        if self.check_win(1, self.board):
            self.done = True
            return self.get_observation(), 10, True, False, {}

        opp_pot_after = self.count_potential_wins(-1, self.board)
        blocking_reward = (opp_pot_before - opp_pot_after) * 0.5

        agent_setup_after = self.count_setup_potential(1, self.board)
        opp_setup_after = self.count_setup_potential(-1, self.board)
        setup_reward = (agent_setup_after - agent_setup_before) * 0.5 - (opp_setup_after - opp_setup_before) * 0.3

        agent_almost_wins_after = self.count_almost_wins(1, self.board)
        proximity_reward = (agent_almost_wins_after - agent_almost_wins_before) * 0.2

        opp_almost_wins_after = self.count_almost_wins(-1, self.board)
        opp_progress_penalty = (opp_almost_wins_after - opp_almost_wins_before) * -0.2

        reward += blocking_reward + setup_reward + proximity_reward + opp_progress_penalty

        if not self.get_valid_moves(self.board):
            self.done = True
            return self.get_observation(), reward, True, False, {}

        opp_valid_moves = self.get_valid_moves(self.board)
        if opp_valid_moves:
            if self.opponent_params is not None and isinstance(self.opponent_params, dict):
                if self._temp_opponent_model is None:
                    from stable_baselines3 import PPO
                    self._temp_opponent_model = PPO(
                        "MlpPolicy", self, verbose=0,
                        policy_kwargs=dict(net_arch=[128, 64])
                    )
                self._temp_opponent_model.set_parameters(self.opponent_params)
                opp_action, _ = self._temp_opponent_model.predict(self.get_observation())
                if opp_action not in opp_valid_moves:
                    opp_action = self.heuristic_opponent_move() if self.opponent_params == "heuristic" else np.random.choice(opp_valid_moves)
            elif self.opponent_params == "heuristic":
                opp_action = self.heuristic_opponent_move()
            else:
                opp_action = np.random.choice(opp_valid_moves)
            self.board = self.drop_piece(self.board, opp_action, -1)
            if self.check_win(-1, self.board):
                self.done = True
                return self.get_observation(), reward - 1, True, False, {}

        if not self.get_valid_moves(self.board):
            self.done = True

        return self.get_observation(), reward, self.done, False, {}

    def heuristic_opponent_move(self):
        valid_moves = self.get_valid_moves(self.board)
        for col in valid_moves:
            temp_board = self.drop_piece(self.board, col, -1)
            if self.check_win(-1, temp_board):
                return col
            temp_board = self.drop_piece(self.board, col, 1)
            if self.check_win(1, temp_board):
                return col
        if 3 in valid_moves:
            return 3
        return np.random.choice(valid_moves)
