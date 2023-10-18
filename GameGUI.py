import pygame
import numpy as np
from games.Gomoku import Gomoku

class GameGUI():
    def __init__(self, game, mcts):
        self.game = game
        self.mcts = mcts

        self.player = 1
        self.cell_size = 40  # Adjust this to control the size of each cell on the game board
        self.grid_color = (0, 0, 0)
        self.player_colors = {-1: (255, 0, 0), 0: (255, 255, 255), 1: (0, 0, 255)}
        self.screen = None
        self.font = None
        self.initialize_gui()

    def initialize_gui(self):
        pygame.init()
        window_size = (self.game.column_count * self.cell_size, self.game.row_count * self.cell_size)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(self.game.__repr__())

        # Initialize font for displaying game over message
        self.font = pygame.font.Font(None, 36)

        # Draw the initial empty board
        self.draw_board(self.game.get_initial_state())
        pygame.display.flip()  # Update the display

    def draw_board(self, state):
        self.screen.fill((255, 255, 255))  # Fill the screen with white
        for row in range(self.game.row_count):
            for col in range(self.game.column_count):
                pygame.draw.rect(
                    self.screen,
                    self.grid_color,
                    (
                        col * self.cell_size,
                        row * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                    1,
                )
                player = state[row, col]
                if player != 0:
                    pygame.draw.circle(
                        self.screen,
                        self.player_colors[player],
                        (
                            col * self.cell_size + self.cell_size // 2,
                            row * self.cell_size + self.cell_size // 2,
                        ),
                        self.cell_size // 2 - 2,
                    )

    def draw_game_over(self, result):
        if result == 1:
            message = "Player 1 wins!"
        elif result == -1:
            message = "Player -1 wins!"
        else:
            message = "It's a draw!"
        text = self.font.render(message, True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        self.screen.blit(text, text_rect)

    def run_game(self):
        state = self.game.get_initial_state()

        running = True
        game_over = False
        winner = 0

        while running and not game_over:
            action = None
            if self.player == 1:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        col = event.pos[0] // self.cell_size
                        row = event.pos[1] // self.cell_size
                        print("col: ", col, " row: ", row)
                        action = row * self.game.column_count + col
                        print("action: ", action)
                        print("valid actions: ", self.game.get_valid_moves(state))

                        print(self.game.get_valid_moves(state))
                        if self.game.get_valid_moves(state)[action] == 0:
                            action = None
                
            else:
                neutral_state = self.game.change_perspective(state, self.player)
                mcts_probs = self.mcts.search(neutral_state)
                action = np.argmax(mcts_probs)

            if action != 0 and action is None:
                continue

            state = self.game.get_next_state(state, action, self.player)
            print("next state: ", state)
            self.draw_board(state)
            pygame.display.flip()
            winner, game_over = self.game.get_value_and_terminated(state, action)

            self.player = self.game.get_opponent(self.player)
            if game_over:
                print(winner, " wins!")
                self.draw_game_over(winner)
                pygame.display.flip()


             

        pygame.quit()



if __name__ == "__main__":
    game = GomokuGUI(Gomoku())
    game.run_game()
