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

def draw_board(board, screen):
    # If board is a tuple, extract the observation
    if isinstance(board, tuple):
        board = board[0]
    
    # If board is 1D (flattened), reshape to 2D
    if board.ndim == 1:
        board = board.reshape((ROWS, COLS))
    
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
    # If board is 1D, reshape to 2D
    if board.ndim == 1:
        board = board.reshape((ROWS, COLS))
    
    new_board = board.copy()
    for row in range(ROWS - 1, -1, -1):
        if new_board[row, col] == 0:
            new_board[row, col] = player
            break
    return new_board.flatten()  # Return flattened for model compatibility

def check_win(player, board):
    # If board is 1D, reshape to 2D
    if board.ndim == 1:
        board = board.reshape((ROWS, COLS))
    
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
    # If board is 1D, reshape to 2D
    if board.ndim == 1:
        board = board.reshape((ROWS, COLS))
    return board[0, col] == 0

def get_valid_moves(board):
    # If board is 1D, reshape to 2D
    if board.ndim == 1:
        board = board.reshape((ROWS, COLS))
    return [c for c in range(COLS) if board[0, c] == 0]

def run_game(model, starting_player=-1):
    env = Connect4Env()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4: Play Against PPO Agent")
    clock = pygame.time.Clock()

    board, _ = env.reset()  # Unpack observation and info (board is already flattened)
    game_over = False
    current_player = starting_player  # -1 for human, 1 for agent

    draw_board(board, screen)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Human's turn: on mouse click, determine column
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
            action, _ = model.predict(board)  # Model expects flattened input
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
    # Load a pre-trained model
    from stable_baselines3 import PPO
    model = PPO.load(r"C:\Users\lndnc\OneDrive\Desktop\AI\connect4\connect4_ensemble_master.zip")
    run_game(model, starting_player=1)