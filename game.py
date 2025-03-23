import numpy as np
import pygame
import sys
from connect4_env import Connect4Env

# PyGame parameters
SQUARE_SIZE = 100
COLS = 7
ROWS = 6
WIDTH = COLS * SQUARE_SIZE
HEIGHT = ROWS * SQUARE_SIZE
RADIUS = int(SQUARE_SIZE / 2 - 5)

# Colors (RGB)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)      # Agent (player 1)
YELLOW = (255, 255, 0) # Human (player -1)

def observation_to_board(obs):
    """Convert a (3, 6, 7) observation to a (6, 7) board."""
    if obs.ndim == 3:  # (3, 6, 7)
        board = np.zeros((ROWS, COLS), dtype=np.int8)
        board[obs[0] == 1] = 1    # Player 1
        board[obs[1] == 1] = -1   # Player -1
        # Empty cells (obs[2] == 1) remain 0
        return board
    return obs  # If already 2D, return as-is

def board_to_observation(board):
    """Convert a (6, 7) board to a (3, 6, 7) observation for the model."""
    obs = np.zeros((3, ROWS, COLS), dtype=np.int8)
    obs[0] = (board == 1).astype(np.int8)  # Player 1
    obs[1] = (board == -1).astype(np.int8) # Player -1
    obs[2] = (board == 0).astype(np.int8)  # Empty
    return obs

def draw_board(board, screen):
    # Convert to 2D if necessary
    board = observation_to_board(board)
    
    # Draw board background and empty slots
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                               RADIUS)
    
    # Draw the pieces
    for c in range(COLS):
        for r in range(ROWS):
            if board[r, c] == 1:
                pygame.draw.circle(screen, RED,
                                   (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                                   RADIUS)
            elif board[r, c] == -1:
                pygame.draw.circle(screen, YELLOW,
                                   (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                                   RADIUS)
    pygame.display.update()

def drop_piece(board, col, player):
    board = observation_to_board(board)
    new_board = board.copy()
    for row in range(ROWS - 1, -1, -1):
        if new_board[row, col] == 0:
            new_board[row, col] = player
            break
    return new_board  # Keep as 2D for game logic

def check_win(player, board):
    board = observation_to_board(board)
    
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if np.all(board[r, c:c+4] == player):
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if np.all(board[r:r+4, c] == player):
                return True
    # Diagonal (positive slope)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i, c+i] == player for i in range(4)):
                return True
    # Diagonal (negative slope)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r-i, c+i] == player for i in range(4)):
                return True
    return False

def is_valid_move(board, col):
    board = observation_to_board(board)
    return board[0, col] == 0

def get_valid_moves(board):
    board = observation_to_board(board)
    return [c for c in range(COLS) if board[0, c] == 0]

def run_game(model, starting_player=-1):
    env = Connect4Env()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4: Play Against PPO Agent")
    clock = pygame.time.Clock()

    obs, _ = env.reset()  # Get initial (3, 6, 7) observation
    board = observation_to_board(obs)  # Convert to 2D for game logic
    game_over = False
    current_player = starting_player  # -1 for human, 1 for agent

    draw_board(board, screen)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Human's turn
            if current_player == -1 and event.type == pygame.MOUSEBUTTONDOWN:
                posx = event.pos[0]
                col = posx // SQUARE_SIZE
                if is_valid_move(board, col):
                    board = drop_piece(board, col, -1)
                    if check_win(-1, board):
                        draw_board(board, screen)
                        print("Human wins!")
                        game_over = True
                    current_player = 1
                    draw_board(board, screen)
        
        # Agent's turn
        if current_player == 1 and not game_over:
            # Convert 2D board to 3D observation for model
            obs = board_to_observation(board)
            action, _ = model.predict(obs)  # Model expects (3, 6, 7)
            valid_moves = get_valid_moves(board)
            if action not in valid_moves:
                action = np.random.choice(valid_moves)
            board = drop_piece(board, action, 1)
            if check_win(1, board):
                draw_board(board, screen)
                print("Agent wins!")
                game_over = True
            current_player = -1
            draw_board(board, screen)
        
        # Check for draw
        if len(get_valid_moves(board)) == 0 and not game_over:
            print("It's a draw!")
            game_over = True
        
        clock.tick(30)
    
    pygame.time.wait(3000)
    pygame.quit()

if __name__ == "__main__":
    from stable_baselines3 import PPO
    model = PPO.load(r"C:\Users\lndnc\OneDrive\Desktop\AI\connect4\connect4_cnn_master.zip")
    run_game(model, starting_player=1)