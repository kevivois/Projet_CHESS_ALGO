import numpy as np
from typing import Tuple, List, Dict, Optional
import time
from collections import defaultdict
from Bots.ChessBotList import register_chess_bot
from Bots.piece_values import (
    pawntable,
    knightstable,
    bishopstable,
    rookstable,
    queenstable,
    kingstable,
    kings_end_game_table,
    individual_piece_values,
)


class ChessBot:
    def __init__(self):
        self.piece_values = individual_piece_values
        self.position_weights = {
            "p": np.array(pawntable).reshape(8, 8),
            "n": np.array(knightstable).reshape(8, 8),
            "b": np.array(bishopstable).reshape(8, 8),
            "r": np.array(rookstable).reshape(8, 8),
            "q": np.array(queenstable).reshape(8, 8),
            "k": np.array(kingstable).reshape(8, 8),
        }
        self.kings_end_game_table = np.array(kings_end_game_table).reshape(8, 8)
        self.opening_moves = [
            ((6, 4), (5, 4)),
            ((6, 3), (5, 3)),
            ((7, 1), (5, 2)),
            ((7, 6), (5, 5)),
            ((6, 2), (5, 2)),
            ((6, 5), (5, 5)),
        ]
        self.move_history = defaultdict(lambda: {"count": 0, "success": 0})

    def is_square_threatened(
        self, board: np.ndarray, x: int, y: int, color: str
    ) -> bool:
        opponent_color = "b" if color == "w" else "w"
        for i in range(8):
            for j in range(8):
                if board[i, j] != "" and board[i, j][1] == opponent_color:
                    moves = self.get_piece_moves(board, i, j, opponent_color)
                    if (x, y) in moves:
                        return True
        return False

    def get_piece_moves(
        self, board: np.ndarray, x: int, y: int, color: str
    ) -> List[Tuple[int, int]]:
        piece = board[x, y]
        if not piece or piece[1] != color:
            return []
        piece_type = piece[0]
        moves = []

        if piece_type == "p":
            direction = 1
            new_x = x + direction
            if 0 <= new_x < 8 and board[new_x, y] == "":
                moves.append((new_x, y))
            for dy in [-1, 1]:
                new_y = y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    if board[new_x, new_y] != "" and board[new_x, new_y][1] != color:
                        moves.append((new_x, new_y))

        elif piece_type in ["r", "b", "q"]:
            directions = []
            if piece_type in ["r", "q"]:
                directions += [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if piece_type in ["b", "q"]:
                directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                while 0 <= new_x < 8 and 0 <= new_y < 8:
                    if board[new_x, new_y] == "":
                        moves.append((new_x, new_y))
                    elif board[new_x, new_y][1] != color:
                        moves.append((new_x, new_y))
                        break
                    else:
                        break
                    new_x, new_y = new_x + dx, new_y + dy

        elif piece_type == "n":
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
            for dx, dy in knight_moves:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    if board[new_x, new_y] == "" or board[new_x, new_y][1] != color:
                        moves.append((new_x, new_y))

        elif piece_type == "k":
            king_moves = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
            for dx, dy in king_moves:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    if board[new_x, new_y] == "" or board[new_x, new_y][1] != color:
                        moves.append((new_x, new_y))

        return moves

    def evaluate_position(self, board: np.ndarray, color: str) -> float:
        score = 0.0
        piece_count = 0
        for x in range(8):
            for y in range(8):
                piece = board[x, y]
                if piece:
                    piece_count += 1
                    piece_type, piece_color = piece[0], piece[1]
                    base_value = self.piece_values[piece_type]
                    if piece_type in self.position_weights:
                        position_bonus = self.position_weights[piece_type][x, y]
                        if piece_type == "k" and piece_count <= 10:
                            position_bonus = self.kings_end_game_table[x, y]
                        base_value += position_bonus
                    multiplier = 1 if piece_color == color else -1
                    score += base_value * multiplier

        # Bonus pour le contrôle du centre
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for x, y in center_squares:
            if board[x, y] != "" and board[x, y][1] == color:
                score *= 1.1

        # Bonus pour la structure des pions
        pawn_columns = defaultdict(int)
        for x in range(8):
            for y in range(8):
                if board[x, y].startswith("p") and board[x, y][1] == color:
                    pawn_columns[y] += 1
                    if (color == "w" and x == 1) or (
                        color == "b" and x == 6
                    ):  # Bonus pour les pions près de la promotion
                        score *= 1.2

        for y in range(8):
            if pawn_columns[y] > 0 and (
                pawn_columns[y - 1] > 0
                if y > 0
                else False or pawn_columns[y + 1] > 0 if y < 7 else False
            ):
                score *= 1.1

        return score

    def minimax(
        self,
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        color: str,
        start_time: float,
        time_budget: float,
    ) -> Tuple[float, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        # if time.time() - start_time > time_budget * 0.95:
        #    raise TimeoutError

        if depth == 0 or time.time() - start_time > time_budget * 0.95:
            return self.evaluate_position(board, color), None

        moves = []
        current_color = color if maximizing else ("b" if color == "w" else "w")
        for x in range(8):
            for y in range(8):
                if board[x, y] != "" and board[x, y][1] == current_color:
                    piece_moves = self.get_piece_moves(board, x, y, current_color)
                    moves.extend([((x, y), move) for move in piece_moves])

        if maximizing:
            best_value = float("-inf")
            best_move = None
            for move in moves:
                start, end = move
                temp_board = board.copy()
                temp_board[end[0], end[1]] = temp_board[start[0], start[1]]
                temp_board[start[0], start[1]] = ""
                value, _ = self.minimax(
                    temp_board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    color,
                    start_time,
                    time_budget,
                )
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_value, best_move
        else:
            best_value = float("inf")
            best_move = None
            for move in moves:
                start, end = move
                temp_board = board.copy()
                temp_board[end[0], end[1]] = temp_board[start[0], start[1]]
                temp_board[start[0], start[1]] = ""
                value, _ = self.minimax(
                    temp_board,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    color,
                    start_time,
                    time_budget,
                )
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_value, best_move


def chess_bot(
    player_sequence: List[str], board: np.ndarray, time_budget: float, **kwargs
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    bot = ChessBot()
    color = player_sequence[1]
    start_time = time.time()
    best_move = None
    depth = 1

    try:
        while time.time() - start_time < time_budget * 0.9:
            _, current_best_move = bot.minimax(
                board,
                depth,
                float("-inf"),
                float("inf"),
                True,
                color,
                start_time,
                time_budget,
            )
            if current_best_move:
                best_move = current_best_move
                move_key = str(best_move)
                bot.move_history[move_key]["count"] += 1
            depth += 1
    except TimeoutError:
        pass

    if not best_move:
        direction = -1 if color == "w" else 1
        for x in range(8):
            for y in range(8):
                if (
                    board[x, y] == f"p{color}"
                    and 0 <= x + direction < 8
                    and board[x + direction, y] == ""
                ):
                    return (x, y), (x + direction, y)

    return best_move or ((0, 0), (0, 0))


register_chess_bot("test", chess_bot)
