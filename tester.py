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
RED = (255, 0, 0)      # Agent's pieces (player 1)
YELLOW = (255, 255, 0) # Opponent's pieces (player -1)

def draw_board(board, screen):
    # Draw board background and empty slots.
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                               RADIUS)
    # Draw pieces based on board state.
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

def run_tester():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4 Tester")
    clock = pygame.time.Clock()

    env = Connect4Env()
    state = env.reset()
    draw_board(state, screen)
    
    print("Starting tester. Click a column to make a move.")
    done = False
    move_count = 0
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # When user clicks, interpret the x-coordinate as the column.
            if event.type == pygame.MOUSEBUTTONDOWN:
                posx = event.pos[0]
                col = posx // SQUARE_SIZE

                # If the move is valid, take a step in the environment.
                if env.is_valid_move(col):
                    state, reward, done, info = env.step(col)
                    move_count += 1
                    total_reward += reward
                    print(f"Move {move_count}: Chose column {col}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}")
                    draw_board(state, screen)
                else:
                    print(f"Clicked column {col} is invalid. Try a different column.")
                    
        clock.tick(30)
    
    print("Game over. Final Total Reward:", total_reward)
    pygame.time.wait(3000)
    pygame.quit()

if __name__ == "__main__":
    run_tester()
