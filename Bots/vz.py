import numpy as np
import time
import random
from PyQt6 import QtCore

# Importing the chess bot registration utility.
from Bots.ChessBotList import register_chess_bot

import os

# File to log the chess bot's activities.
file_name = "log_p2.txt"
file = open(file_name, "a+")


# Function to print logs to the console and write them to a file.
def print_log(text):
    print(text)
    file.write(text + "\n")
    file.flush()


# Dictionary for memoization of previously computed board states to optimize performance.
transposition_table = {}


# Main chess bot function that calculates the best move based on the board state.
def chess_bot(player_sequence, board, time_budget, **kwargs):
    """
    Main function for the chess bot that calculates the best move based on a given board state.

    :param player_sequence: Sequence of players, indicating whose turn it is.
    :param board: Current state of the chess board as a numpy array.
    :param time_budget: Time allocated for the bot to calculate its move.
    :param kwargs: Additional parameters for flexibility.
    :return: The best move as a tuple of start and end positions.
    """
    start_time = time.time()
    currentPlayerColor = player_sequence[1]  # Current player's color.

    # Assigning values to chess pieces for evaluation purposes.
    piece_values = {
        "r": 500,  # Rook
        "n": 300,  # Knight
        "b": 300,  # Bishop
        "q": 900,  # Queen
        "k": 10000,  # King
        "p": 50,  # Pawn
    }

    # Function to compute a hash value for the current board state for memoization.
    def get_board_hash(board):
        return hash(board.tobytes())

    # Static evaluation tables for piece positioning based on chess strategy.
    pawnEval = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
        [0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
        [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    knightEval = [
        [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
        [-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0],
        [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0],
        [-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0],
        [-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0],
        [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0],
        [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0],
        [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
    ]

    bishopEval = [
        [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0],
        [-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0],
        [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0],
        [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0],
        [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
    ]

    rookEval = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
        [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
        [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
        [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
        [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
        [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    ]

    evalQueen = [
        [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
        [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
        [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
        [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
        [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
        [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
    ]

    # Function to get the value of a piece based on its type and position on the board.
    def get_piece_value(piece, color, x, y):
        """
        Calculate the value of a piece considering its position and type.

        :param piece: The type of the piece (e.g., 'p' for pawn, 'k' for king).
        :param color: The color of the piece ('w' or 'b').
        :param x: Row position of the piece on the board.
        :param y: Column position of the piece on the board.
        :return: Computed value of the piece.
        """
        value = piece_values.get(piece, 0)  # Base value of the piece.
        matrix = []
        if piece == "p":
            matrix = pawnEval
        elif piece == "r":
            matrix = rookEval
        elif piece == "n":
            matrix = knightEval
        elif piece == "b":
            matrix = bishopEval
        elif piece == "k":
            matrix = knightEval
        elif piece == "q":
            matrix = evalQueen
        if color != currentPlayerColor:
            matrix = matrix[::-1]  # Reverse the evaluation matrix for the opponent.
        return value + matrix[x][y]

    # Function to generate all legal moves for a specific piece on the board.
    def generate_move(x, y, color, board):
        """
        Generate all legal moves for a piece at position (x, y).

        :param x: Row index.
        :param y: Column index.
        :param color: Color of the piece.
        :param board: Current board state.
        :return: List of possible moves as tuples ((start_x, start_y), (end_x, end_y)).
        """
        deplacements = []
        piece = board[x, y]

        if piece == "" or piece[1] != color:  # No piece or opponent's piece.
            return []
        else:
            if piece == "p":
                moves = pion(x, y, board, color)  # Generate moves for pawns.
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == "n":
                moves = cavalier(x, y, board, color)  # Generate knight moves.
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == "k":
                moves = roi(x, y, board, color)  # Generate king moves.
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == "r":
                moves = tour(x, y, board, color)  # Generate rook moves.
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == "b":
                moves = fou(x, y, board, color)  # Generate bishop moves.
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == "q":
                moves = reine(x, y, board, color)  # Generate queen moves.
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
        return deplacements

    # Function to evaluate the development of pieces for a specific color.
    def evaluate_development(board, color):
        """
        Evaluate the development of a player's pieces based on their positions.

        :param board: Current board state.
        :param color: Color of the pieces to evaluate.
        :return: Development score for the color.
        """
        score = 0
        back_rank = 0 if color == currentPlayerColor else 7  # Determine back rank.
        for y in range(8):
            piece = board[back_rank, y]
            if piece != "" and piece[1] == color:
                if piece[0] in ["n", "b", "r"]:
                    score -= 100  # Penalize undeveloped knights, bishops, and rooks.
                elif piece[0] == "q":
                    score -= 50  # Smaller penalty for an undeveloped queen.
        return score

        return score

    # Function to evaluate threats posed by a player's pieces.
    def evaluate_threats(board, color):
        """
        Calculate the threat score based on the moves that attack opponent pieces.

        :param board: Current board state.
        :param color: Color of the player to evaluate threats for.
        :return: Threat score.
        """
        threat_score = 0
        opponent_color = "b" if color == "w" else "w"

        for x in range(8):
            for y in range(8):
                if board[x, y] != "" and board[x, y][1] == color:
                    moves = generate_move(x, y, color, board)
                    for move in moves:
                        target_x, target_y = move[1]
                        target_piece = board[target_x, target_y]
                        if target_piece != "" and target_piece[1] == opponent_color:
                            threat_score += (
                                get_piece_value(target_piece[0], target_piece[1], x, y)
                                * 0.1
                            )

        return threat_score

    # Function to evaluate potential captures.
    def evaluate_captures(board, color):
        """
        Evaluate the board for potential captures a player can make.

        :param board: Current board state.
        :param color: Color of the player to evaluate.
        :return: Capture bonus score.
        """
        bonus = 0
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] != "" and board[x, y][1] != color:
                    attackers = count_attackers(board, x, y, color)
                    if attackers > 0:
                        piece_value = piece_values.get(board[x, y][0], 0)
                        bonus += piece_value * 0.1 * attackers
        return bonus

    # Function to count the number of attackers on a specific square.
    def count_attackers(board, x, y, color):
        """
        Count how many pieces of the given color can attack a square.

        :param board: Current board state.
        :param x: Row index of the target square.
        :param y: Column index of the target square.
        :param color: Color of the attacking pieces to evaluate.
        :return: Number of attackers on the square.
        """
        return sum(1 for move in generate_moves(board, color) if move[0] == (x, y))

    # Helper function to evaluate the safety of the king.
    def is_king_safe(board, color):
        """
        Check if the king of the given color is in a safe position.

        :param board: Current board state.
        :param color: The color of the king to check.
        :return: True if the king is safe, False otherwise.
        """
        king_position = None
        for x in range(8):
            for y in range(8):
                if board[x, y] == "k" + color:  # Find the king.
                    king_position = (x, y)
                    break
        if not king_position:
            return False  # King is missing (edge case).

        return not is_king_in_check(board, color, king_position)

    # Function to check if the king is in check.
    def is_king_in_check(board, color, king_position):
        """
        Determine if the king is currently in check.

        :param board: Current board state.
        :param color: Color of the king to check.
        :param king_position: Coordinates of the king.
        :return: True if the king is in check, False otherwise.
        """
        opponent_moves = generate_moves(board, "w" if color == "b" else "b")
        for start, end in opponent_moves:
            if end[0] == king_position[0] and end[1] == king_position[1]:
                return True
        return False

    # Function to determine if the player is in checkmate.
    def is_checkmate(board, color):
        """
        Check if the player of the given color is in checkmate.

        :param board: Current board state.
        :param color: Color of the player to check.
        :return: True if the player is in checkmate, False otherwise.
        """
        return not is_king_safe(board, "w" if color == "b" else "b")

    # Function to evaluate the board state for a player.
    def evaluate_board(board, color):
        """
        Calculate a score for the board state based on material and positional factors.

        :param board: Current board state.
        :param color: Color of the player for evaluation.
        :return: Evaluation score.
        """
        MATE_SCORE = 100000
        score = 0

        # Evaluate material and positional values.
        for x in range(8):
            for y in range(8):
                piece = board[x, y]
                if piece != "":
                    piece_value = get_piece_value(piece[0], piece[1], x, y)
                    if piece[1] == color:
                        score += piece_value
                    else:
                        score -= piece_value

        # Checkmate conditions.
        if is_checkmate(board, color):
            return -MATE_SCORE
        if is_checkmate(board, "w" if color == "b" else "b"):
            return MATE_SCORE

        # Add positional bonuses.
        center_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for x, y in center_positions:
            piece = board[x, y]
            if piece == "":
                continue
            if piece[1] == color:
                score += 50
            else:
                score -= 50

        score += 3 * evaluate_captures(board, color)
        score += 2 * len(generate_moves(board, color))
        threat_score = 2 * evaluate_threats(board, color)
        score += threat_score
        score += evaluate_development(board, color)

        if is_king_safe(board, color):
            score += 400

        return score

    def evaluate_captures(board, color):
        """
        Evaluate the board for potential captures a player can make.

        :param board: Current board state as a numpy array.
        :param color: The color of the player evaluating captures ('w' or 'b').
        :return: A score representing the potential value of captures.
        """
        capture_score = 0
        for x in range(8):
            for y in range(8):
                piece = board[x, y]
                if piece != "" and piece[1] != color:
                    attackers = count_attackers(board, x, y, color)
                    if attackers > 0:
                        capture_score += (
                            get_piece_value(piece[0], piece[1], x, y) * 1 * attackers
                        )
        return capture_score

    # Function to sort moves by heuristic importance.
    def sort_moves(board, moves, color):
        """
        Sort moves by their heuristic importance to prioritize the best moves.

        :param board: Current board state.
        :param moves: List of moves to sort.
        :param color: Color of the player.
        :return: Sorted list of moves.
        """

        def move_score(move):
            start, end = move
            piece = board[start[0], start[1]]
            target = board[end[0], end[1]]
            score = 0

            # Priorité aux captures
            if target != "":
                score += (
                    1000
                    + get_piece_value(target[0], target[1], end[0], end[1])
                    - get_piece_value(piece[0], piece[1], start[0], start[1])
                )

                # Bonus pour les mouvements vers le centre
            if end[0] in [3, 4] and end[1] in [3, 4]:
                score += 75  # Augmenter le bonus pour le contrôle du centre

            # Bonus pour le développement des pièces
            if piece[0] in ["n", "b", "r"] and start[0] in [0, 7]:
                score += 50  # Bonus pour développer les pièces de la rangée arrière

            return score

        return sorted(moves, key=move_score, reverse=True)

    # Function to make a move on the board and return a new board state.
    def make_move(board, move):
        """
        Apply a move to the board and return the resulting board state.

        :param board: Current board state.
        :param move: Move to apply as a tuple ((start_x, start_y), (end_x, end_y)).
        :return: New board state after the move.
        """
        start, end = move
        from copy import deepcopy

        new_board = deepcopy(board)
        new_board[start[0], start[1]] = new_board[end[0], end[1]]
        return new_board

    # Function to perform iterative deepening search using minimax.
    def minimaxRoot(depth, board, isMaximizing, maximizing_color):
        """
        Root function for the minimax algorithm with alpha-beta pruning.

        :param depth: Depth of the search tree.
        :param board: Current board state.
        :param isMaximizing: Whether this is a maximizing player's turn.
        :param maximizing_color: Color of the maximizing player.
        :return: Best score and corresponding move.
        """
        newGameMoves = generate_moves(board, maximizing_color)
        print(newGameMoves, "moves__")
        bestScore = float("-inf")
        bestMove = None

        for move in newGameMoves:
            new_board = make_move(board, move)
            try:

                score = minimax(
                    new_board,
                    depth - 1,
                    float("-inf"),
                    float("inf"),
                    isMaximizing,
                    maximizing_color,
                )
                if score >= bestScore:
                    bestScore = score
                    bestMove = move
            except Exception as e:
                print(str(e))
                break
        print_log(f"[Chess Bot] Best move: {bestMove}, Score: {bestScore}")
        return bestScore, bestMove

    # Minimax function with alpha-beta pruning for evaluating moves.
    def minimax(board, depth, alpha, beta, maximizing_player, maximizing_color):
        """
        Minimax algorithm with alpha-beta pruning to evaluate board states.

        :param board: Current board state.
        :param depth: Depth of the search tree.
        :param alpha: Alpha value for pruning.
        :param beta: Beta value for pruning.
        :param maximizing_player: True if it's the maximizing player's turn.
        :param maximizing_color: Color of the maximizing player.
        :return: Evaluation score for the current board state.
        """
        if (
            time.time() - start_time > time_budget * 0.95
        ):  # Utilisez 95% du temps alloué
            raise TimeoutError("Time limit exceeded")

        if depth == 0 or time.time() - start_time > time_budget * 0.9:
            score = evaluate_board(board, maximizing_color)
            print_log(f"Leaf evaluation at depth {depth}: {score}")
            return score

        current_color = (
            maximizing_color
            if maximizing_player
            else ("b" if maximizing_color == "w" else "w")
        )
        moves = generate_moves(board, current_color)

        best_score = float("-inf") if maximizing_player else float("inf")

        for move in moves:
            new_board = make_move(board, move)
            score = minimax(
                new_board,
                depth - 1,
                alpha,
                beta,
                not maximizing_player,
                maximizing_color,
            )

            if maximizing_player:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                print_log(
                    f"Max node at depth {depth}, move {move}: {score} (best: {best_score})"
                )
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                print_log(
                    f"Min node at depth {depth}, move {move}: {score} (best: {best_score})"
                )

            if beta <= alpha:
                print_log(f"Pruning at depth {depth}")
                break
        return best_score

    # Helper function to generate all possible moves for all pieces of a given color.
    def generate_moves(board, color):
        """
        Generate all possible moves for the given color.

        :param board: Current board state.
        :param color: The color of the player ('w' or 'b').
        :return: List of possible moves for the color.
        """
        deplacements = []
        for x in range(8):
            for y in range(8):
                if board[x, y] == "":
                    continue
                piece = board[x, y][0]
                clr = board[x, y][1]
                if color == clr:
                    if piece == "p":  # Pion
                        moves = pion(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])
                    elif piece == "n":  # Cavalier
                        moves = cavalier(x, y, board, color)
                        deplacements.extend(
                            [
                                (
                                    (
                                        x,
                                        y,
                                    ),
                                    (nx, ny),
                                )
                                for nx, ny in moves
                            ]
                        )
                    elif piece == "k":  # Roi
                        moves = roi(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])
                    elif piece == "r":  # Tour
                        moves = tour(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])
                    elif piece == "b":  # Fou
                        moves = fou(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])
                    elif piece == "q":  # Reine
                        moves = reine(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])

        return sort_moves(board, deplacements, color)

    # Movement generation functions for specific pieces.
    def cavalier(pos_x, pos_y, board, color):
        """
        Generate legal moves for a knight.

        :param pos_x: Row index of the knight.
        :param pos_y: Column index of the knight.
        :param board: Current board state.
        :param color: Color of the knight.
        :return: List of legal moves.
        """
        mouvements = [
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ]
        deplacements = []
        for dx, dy in mouvements:
            nx, ny = pos_x + dx, pos_y + dy
            if 0 <= nx <= 7 and 0 <= ny <= 7:
                piece = board[nx, ny]
                if piece == "" or (piece[1] != color):  # Vide ou pièce ennemie
                    deplacements.append((nx, ny))
        return deplacements

    def pion(pos_x, pos_y, board, color):
        """
        Generate legal moves for a pawn.

        :param pos_x: Row index of the pawn.
        :param pos_y: Column index of the pawn.
        :param board: Current board state.
        :param color: Color of the pawn.
        :return: List of legal moves.
        """
        deplacements = []
        direction = 1 if color == currentPlayerColor else -1

        # Avance simple
        nx = pos_x + direction
        if 0 <= nx <= 7 and board[nx, pos_y] == "":  # La case devant est vide
            deplacements.append((nx, pos_y))

        # Captures diagonales
        for dy in [-1, 1]:
            nx, ny = pos_x + direction, pos_y + dy
            if 0 <= nx <= 7 and 0 <= ny <= 7:
                if board[nx, ny] != "" and board[nx, ny][1] != color:
                    deplacements.append((nx, ny))
        return deplacements

    def roi(pos_x, pos_y, board, color):
        """
        Generate legal moves for a king.

        :param pos_x: Row index of the king.
        :param pos_y: Column index of the king.
        :param board: Current board state.
        :param color: Color of the king.
        :return: List of legal moves.
        """
        mouvements = [
            (-1, -1),
            (-1, 0),
            (-1, 1),  # Haut gauche, haut, haut droite
            (0, -1),
            (0, 1),  # Gauche, droite
            (1, -1),
            (1, 0),
            (1, 1),  # Bas gauche, bas, bas droite
        ]
        deplacements = []

        for dx, dy in mouvements:
            nx, ny = pos_x + dx, pos_y + dy
            if (
                0 <= nx <= 7 and 0 <= ny <= 7
            ):  # Vérifie que la position reste sur le plateau
                piece = board[nx, ny]
                if (
                    piece == "" or piece[1] != color
                ):  # Case vide ou occupée par une pièce ennemie
                    deplacements.append((nx, ny))

        return deplacements

    def tour(pos_x, pos_y, board, color):
        """
        Generate legal moves for a rook.

        :param pos_x: Row index of the rook.
        :param pos_y: Column index of the rook.
        :param board: Current board state.
        :param color: Color of the rook.
        :return: List of legal moves.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Haut, Bas, Gauche, Droite
        return get_moves_directions(board, pos_x, pos_y, color, directions)

    def fou(pos_x, pos_y, board, color):
        """
        Generate legal moves for a bishop.

        :param pos_x: Row index of the bishop.
        :param pos_y: Column index of the bishop.
        :param board: Current board state.
        :param color: Color of the bishop.
        :return: List of legal moves.
        """
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonales
        return get_moves_directions(board, pos_x, pos_y, color, directions)

    def reine(pos_x, pos_y, board, color):
        """
        Generate legal moves for a queen.

        :param pos_x: Row index of the queen.
        :param pos_y: Column index of the queen.
        :param board: Current board state.
        :param color: Color of the queen.
        :return: List of legal moves.
        """
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),  # Directions de la tour
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),  # Directions du fou
        ]
        return get_moves_directions(board, pos_x, pos_y, color, directions)

    def get_moves_directions(board, pos_x, pos_y, color, directions):
        """
        Generate moves for a piece that moves in specific directions (e.g., rook, bishop, queen).

        :param board: Current board state.
        :param pos_x: Row index of the piece.
        :param pos_y: Column index of the piece.
        :param color: Color of the piece.
        :param directions: List of (dx, dy) tuples indicating move directions.
        :return: List of legal moves for the piece.
        """
        moves = []
        for dx, dy in directions:
            nx, ny = pos_x + dx, pos_y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                piece = board[nx, ny]
                if piece == "":  # Case vide
                    moves.append((nx, ny))
                elif piece[1] != color:  # Pièce ennemie
                    moves.append((nx, ny))
                    break
                else:  # Pièce alliée
                    break
                nx += dx
                ny += dy
        return moves

    # Fonction d'évaluation modifiée

    # Main evaluation loop for iterative deepening search.
    best_score = float("-inf")
    best_move = None
    for depth in range(2, 8):  # Incrementally increase depth to refine the best move.
        if time.time() - start_time > time_budget * 0.8:  # Stop if time is running out.
            break
        current_score, current_best_move = minimaxRoot(
            depth, board, True, currentPlayerColor
        )
        if current_score > best_score:
            best_move = current_best_move
            best_score = current_score

    # Return the best move found or a default move if none is found.
    return best_move if best_move != None else ((0, 0), (0, 0))


# Register the chess bot with its unique name.
register_chess_bot("vz", chess_bot)
