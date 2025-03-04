import numpy as np
import pygame
import sys
from connect4_env import Connect4Env

# PyGame parameters.
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
    
    # Draw board background and empty slots.
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                               RADIUS)
    
    # Draw the pieces.
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

def run_game(model, starting_player=-1):
    env = Connect4Env()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4: Play Against PPO Agent")
    clock = pygame.time.Clock()

    board, _ = env.reset()  # Unpack observation and info
    game_over = False
    current_player = starting_player  # Starting player: -1 for human, 1 for agent

    draw_board(board, screen)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Human's turn: on mouse click, determine column.
            if current_player == -1 and event.type == pygame.MOUSEBUTTONDOWN:
                posx = event.pos[0]
                col = posx // SQUARE_SIZE
                if env.is_valid_move(col):
                    board = env.drop_piece(board, col, -1)
                    if env.check_win(-1, board):
                        draw_board(board, screen)
                        print("Human wins!")
                        game_over = True
                    current_player = 1
                    draw_board(board, screen)
        
        # Agent's turn.
        if current_player == 1 and not game_over:
            action, _ = model.predict(board)
            valid_moves = env.get_valid_moves(board)
            if action not in valid_moves:
                action = np.random.choice(valid_moves)
            board = env.drop_piece(board, action, 1)
            if env.check_win(1, board):
                draw_board(board, screen)
                print("Agent wins!")
                game_over = True
            current_player = -1
            draw_board(board, screen)
        
        # Check for draw.
        if len(env.get_valid_moves(board)) == 0 and not game_over:
            print("It's a draw!")
            game_over = True
        
        clock.tick(30)
    
    pygame.time.wait(3000)
    pygame.quit()

if __name__ == "__main__":
    # Option 1: Pass the model directly (if already defined in your training script)
    norm = False

    if norm:
        run_game(model)
    else:
    # Option 2: Load a pre-trained model from a file.
    # Uncomment the lines below to load a model and play.
        from stable_baselines3 import PPO
        model = PPO.load(r"C:\Users\lndnc\OneDrive\Desktop\AI\connect4\connect4_ensemble_master.zip")
        run_game(model, starting_player=1)
    
    # For now, if no model is passed, raise an error:
    # raise ValueError("No model provided. Please pass a model or uncomment the model loading code.")
