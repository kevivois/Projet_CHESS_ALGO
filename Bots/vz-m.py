import numpy as np
import time
import random
from PyQt6 import QtCore

#   Be careful with modules to import from the root (don't forget the Bots.)
from Bots.ChessBotList import register_chess_bot


def chess_bot1(player_sequence, board, time_budget, **kwargs):
    """
    Main function for the chess bot that calculates the best move based on a given board state.

    :param player_sequence: Sequence of players, indicating whose turn it is.
    :param board: Current state of the chess board as a numpy array.
    :param time_budget: Time allocated for the bot to calculate its move.
    :param kwargs: Additional parameters for flexibility.
    :return: The best move as a tuple of start and end positions.
    """
    start_time = time.time()
    possible_move = []
    color = player_sequence[1]
    piece_values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 200}

    def evaluate_move(start, end, board, path, color):
        """
        Evaluate a move based on various criteria such as capturing pieces,
        avoiding repetitive moves, and ensuring the king is not in check after the move.

        :param start: The starting position of the move as a tuple (x, y).
        :param end: The ending position of the move as a tuple (x, y).
        :param board: The current state of the chess board as a numpy array.
        :param path: The sequence of moves leading to this position.
        :param color: The color of the player ('w' for white, 'b' for black).
        :return: The score of the move as a float.
        """
        x, y = end
        score = 0
        if board[x, y] == "":
            score = 1
        else:
            piece = board[x, y][0]
            if piece[-1] != color:
                score = 10 + piece_values.get(piece[0], 0)

        # Penalty for repetitive moves
        if len(path) > 2 and path[-2] == [end, start]:
            score -= 2

        # Check if the king is in check after the move
        temp_board = board.copy()
        temp_board[end[0], end[1]] = temp_board[start[0], start[1]]
        temp_board[start[0], start[1]] = ""
        if is_king_in_check(temp_board, color):
            score -= 50  # High penalty if the king is in check

        # Bonus for escaping a check
        if is_king_in_check(board, color) and not is_king_in_check(temp_board, color):
            score += 30  # Bonus for escaping check

        # Bonus for controlling the center
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for x, y in center_squares:
            if board[x, y] != "" and board[x, y][-1] == color:
                score += 0.5

        if board[x, y] != "" and board[x, y][1] == color:
            moves = generate_move(x, y, color, board)
            score += len(moves) * 0.1

        return score + random.uniform(-0.1, 0.1)

    def findPath(path, board, depth, alpha, beta, maximizing_player):
        """
        Recursive function to perform alpha-beta pruning and find the best move.

        :param path: The sequence of moves considered so far.
        :param board: Current state of the chess board as a numpy array.
        :param depth: Depth of the search tree to explore.
        :param alpha: Alpha value for pruning in the alpha-beta algorithm.
        :param beta: Beta value for pruning in the alpha-beta algorithm.
        :param maximizing_player: Boolean indicating if the current player is maximizing.
        :return: The best score and the corresponding move as a tuple.
        """
        if depth == 0 or time.time() - start_time > time_budget * 0.9:
            start, end = path[-1] if path else ((0, 0), (0, 0))
            score = evaluate_move(
                start,
                end,
                board,
                path,
                color if maximizing_player else ("b" if color == "w" else "w"),
            )
            return score, path[-1] if path else ((0, 0), (0, 0))

        if maximizing_player:
            max_score = float("-inf")
            best_move = None

            for move in possible_move:
                start, end = move

                new_depth = depth - 1
                if board[end[0], end[1]] != "":
                    new_depth += 1

                original_start = board[start[0]][start[1]]
                original_end = board[end[0]][end[1]]
                board[end[0]][end[1]] = original_start
                board[start[0]][start[1]] = ""

                score, _ = findPath(path + [move], board, new_depth, alpha, beta, False)

                board[start[0]][start[1]] = original_start
                board[end[0]][end[1]] = original_end

                if score > max_score:
                    max_score = score
                    best_move = move

                alpha = max(alpha, max_score)
                if beta <= alpha:
                    break

            return max_score, best_move
        else:
            min_score = float("inf")
            best_move = None
            for move in possible_move:
                max_score = float("-inf")
            best_move = None

            for move in possible_move:
                start, end = move

                new_depth = depth - 1
                if board[end[0], end[1]] != "":
                    new_depth += 1

                original_start = board[start[0]][start[1]]
                original_end = board[end[0]][end[1]]
                board[end[0]][end[1]] = original_start
                board[start[0]][start[1]] = ""

                score, _ = findPath(path + [move], board, new_depth, alpha, beta, False)

                board[start[0]][start[1]] = original_start
                board[end[0]][end[1]] = original_end

                if score > max_score:
                    max_score = score
                    best_move = move

                beta = min(beta, max_score)
                if beta <= alpha:
                    break

            return min_score, best_move

    def is_within_board(pos, board):
        """
        Check if a position is within the boundaries of the board.

        :param pos: Position to check as a tuple (x, y).
        :param board: Current state of the chess board as a numpy array.
        :return: True if the position is valid, False otherwise.
        """
        return 0 <= pos[0] < board.shape[0] and 0 <= pos[1] < board.shape[1]

    def generate_move(x, y, color, board):
        """
        Generate all possible moves for a piece located at (x, y).

        :param x: The x-coordinate of the piece.
        :param y: The y-coordinate of the piece.
        :param color: The color of the player ('w' for white, 'b' for black).
        :param board: Current state of the chess board as a numpy array.
        :return: A list of possible moves as tuples of start and end positions.
        """
        moves = []
        piece = board[x, y][0]
        if piece == "p":  # Pawn
            if is_within_board([x + 1, y], board) and board[x + 1, y] == "":
                moves.append([[x, y], [x + 1, y]])
                # Captures
                for dy in [-1, 1]:
                    if (
                        is_within_board([x + 1, y + dy], board)
                        and board[x + 1, y + dy] != ""
                        and board[x + 1, y + dy][-1] != color
                    ):
                        moves.append([[x, y], [x + 1, y + dy]])
        elif piece == "k":  # King
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if is_within_board([new_x, new_y], board):
                        if (
                            board[new_x, new_y] == ""
                            or board[new_x, new_y][-1] != color
                        ):
                            moves.append([[x, y], [new_x, new_y]])
        elif piece == "n":  # Knight
            moves = []
            knight_moves = [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ]
            for move in knight_moves:
                new_pos = [x + move[0], y + move[1]]
                if is_within_board(new_pos, board) and (
                    board[new_pos[0], new_pos[1]] == ""
                    or board[new_pos[0], new_pos[1]][-1] != color
                ):
                    moves.append([[x, y], new_pos])
        elif piece in ["b", "r", "q"]:  # Bishop, Rook, Queen
            directions = []
            if piece in ["b", "q"]:  # Bishop/Queen moves
                directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            if piece in ["r", "q"]:  # Rook/Queen moves
                directions.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                while is_within_board((new_x, new_y), board):
                    if board[new_x, new_y] == "":
                        moves.append([[x, y], [new_x, new_y]])
                    elif board[new_x, new_y][1] != color:
                        moves.append([[x, y], [new_x, new_y]])
                        break
                    else:
                        break
                    new_x, new_y = new_x + dx, new_y + dy

        return moves

    def is_king_in_check(board, color):
        """
        Determine if the king of the specified color is in check.

        :param board: Current state of the chess board as a numpy array.
        :param color: The color of the king ('w' for white, 'b' for black).
        :return: True if the king is in check, False otherwise.
        """
        # Find the king's position
        king_pos = None
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] == f"k{color}":
                    king_pos = (x, y)
                    break
            if king_pos:
                break

        # Check if any opponent piece can attack the king
        opponent_color = "b" if color == "w" else "w"
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] != "" and board[x, y][-1] == opponent_color:
                    moves = generate_move(x, y, opponent_color, board)
                    if any(move[1] == king_pos for move in moves):
                        return True
        return False

    for x in range(board.shape[0] - 1):
        for y in range(board.shape[1]):
            if board[x, y] != "" and board[x, y][-1] == color:
                possible_move.extend(generate_move(x, y, color, board))

    if not possible_move:
        return (0, 0), (0, 0)

    random.shuffle(possible_move)

    piece_count = sum(
        1
        for x in range(board.shape[0])
        for y in range(board.shape[1])
        if board[x, y] != ""
    )

    if piece_count > 20 or is_king_in_check(
        board, color
    ):  # Opening/middlegame or King in check
        depth = 3
    elif piece_count > 10:  # Late middlegame
        depth = 4
    else:  # Endgame
        depth = 5

    _, best_move = findPath([], board, depth, float("-inf"), float("inf"), True)
    if best_move:
        return best_move

    return (0, 0), (0, 0)


#   Example how to register the function
register_chess_bot("vz-m", chess_bot1)
