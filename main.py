from PyQt6.QtWidgets import QApplication
import numpy as np
import time
import random

from ChessArena import ChessArena, ChessApp
from Bots.vz_m import chess_bot1


def evaluate_bot_performance(time_budget):
    """
    Function to evaluate the performance of the bot with a given time budget.

    :param time_budget: The time budget for making a move in seconds.
    :return: The performance score of the bot.
    """
    # Initialize the board and other necessary variables
    board = np.array(
        [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ]
    )
    player_sequence = ["w", "b"]
    start_time = time.time()

    # Simulate a game with the given time budget
    while time.time() - start_time < 10:  # Run the simulation for 10 seconds
        move = chess_bot1(player_sequence, board, time_budget)
        # Apply the move to the board (this is a simplified example)
        start, end = move
        board[end[0], end[1]] = board[start[0], start[1]]
        board[start[0], start[1]] = ""
        player_sequence.reverse()

    # Evaluate the performance (this is a placeholder, you can define your own criteria)
    performance_score = np.random.uniform(
        0, 1
    )  # Replace with actual performance evaluation

    return performance_score


def find_optimal_time_budget():
    """
    Function to find the optimal time budget for the bot.

    :return: The optimal time budget.
    """
    time_budgets = np.linspace(0.125, 0.3, 10)
    best_time_budget = None
    best_performance = -float("inf")

    for time_budget in time_budgets:
        performance = evaluate_bot_performance(time_budget)
        print(f"Time budget: {time_budget}, Performance: {performance}")
        if performance > best_performance:
            best_performance = performance
            best_time_budget = time_budget

    return best_time_budget


if __name__ == "__main__":
    import sys

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook
    print("Starting Chess Arena...")
    app = ChessApp()
    app.start()
