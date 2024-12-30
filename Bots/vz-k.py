#
#   Example function to be implemented for
#       Single important function is next_best
#           color: a single character str indicating the color represented by this bot ('w' for white)
#           board: a 2d matrix containing strings as a descriptors of the board '' means empty location "XC" means a piece represented by X of the color C is present there
#           budget: time budget allowed for this turn, the function must return a pair (xs,ys) --> (xd,yd) to indicate a piece at xs, ys moving to xd, yd
#
import numpy as np
import time
import random
from PyQt6 import QtCore

#   Be careful with modules to import from the root (don't forget the Bots.)
from Bots.ChessBotList import register_chess_bot


# Valeurs des pièces
PIECE_VALUES = {
    "p": 100,  # Pion
    "n": 350,  # Cavalier
    "b": 350,  # Fou
    "r": 525,  # Tour
    "q": 5000,  # Dame
    "k": 10000,  # Roi (non évalué directement)
}


def is_valid_position(x, y, board):
    return 0 <= x < board.shape[0] and 0 <= y < board.shape[1]


def is_empty_or_enemy(board, x, y, color):
    return board[x, y] == "" or board[x, y][1] != color


def get_material_value(board, color):
    result = 0
    for x in range(board.shape[0] - 1):
        for y in range(board.shape[1]):
            if board[x, y] == "":
                continue
            piece = board[x, y][0]
            piece_color = board[x, y][1]
            value = PIECE_VALUES.get(piece, 0)
            if piece_color == color:
                result += value
            else:
                result -= value
    return result


def evaluate_space_control(board, color):
    CENTER_WEIGHTS = {
        (3, 3): 10,
        (3, 4): 10,
        (4, 3): 10,
        (4, 4): 10,  # Cases centrales
        (2, 3): 5,
        (2, 4): 5,
        (3, 2): 5,
        (3, 5): 5,  # Cases autour du centre
        (4, 2): 5,
        (4, 5): 5,
        (5, 3): 5,
        (5, 4): 5,
    }
    score = 0
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            piece = board[x, y]
            if piece and piece[1] == color:
                piece_moves = get_legal_moves_for_piece(board, (x, y), color)
                for move in piece_moves:
                    _, (new_x, new_y) = move
                    # Ajoutez le poids de la case contrôlée
                    score += CENTER_WEIGHTS.get((new_x, new_y), 1)

    return score


def get_mobility_score(board, color):
    mobility_myColor = len(get_legal_moves(board, color))  # Mobilité du joueur
    mobility_opponent = len(
        get_legal_moves(board, "w" if color == "b" else "b")
    )  # Mobilité de l'adversaire

    return mobility_myColor - mobility_opponent


def evaluate_board(board, color):
    # Valeur du matériel done get_material_value(board,color)
    # Contrôle de l'espace evaluate_space_control
    # Mobilité
    # Structure des pions
    # Sécurité du roi
    # Couples de pièces
    # Avantage positionnel
    # Phase de jeu (début, milieu, fin)
    # Menaces et tactiques
    return (
        get_material_value(board, color)
        + evaluate_space_control(board, color)
        + get_mobility_score(board, color)
    )


def evaluate_king_safety(board, color):
    score = 0
    king_pos = next(
        ((i, j) for i in range(8) for j in range(8) if board[i, j] == f"k{color}"), None
    )
    if king_pos:
        x, y = king_pos
        # Bonus for pawns in front of the king
        for dx in [-1, 0, 1]:
            if 0 <= y + dx < 8:
                if color == "W" and x > 0 and board[x - 1, y + dx] == f"p{color}":
                    score += 10
                elif color == "B" and x < 7 and board[x + 1, y + dx] == f"p{color}":
                    score += 10
    return score


def evaluate_mobility(board, color):
    return len(get_legal_moves(board, color))


def get_capture_moves(board, color):
    captures = []
    for move in get_legal_moves(board, color):
        start, end = move
        if board[end] != "":
            captures.append(move)
    return captures


def quiescence(board, alpha, beta, color, start_time, time_budget):
    """Performs quiescence search to stabilize tactical evaluations."""
    stand_pat = evaluate_board(board, color)  # Static evaluation of the position

    # Alpha-beta pruning
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    if time.time() - start_time > time_budget * 0.9:
        return stand_pat

    # Generate all capture moves
    capture_moves = get_capture_moves(board, color)

    for move in capture_moves:
        s, e, _ = make_move(board, move)
        score = -quiescence(
            board, -beta, -alpha, "w" if color == "b" else "b", start_time, time_budget
        )
        undo_move(board, move, e, s)

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def find_king(color, board):
    for x in range(board.shape[0] - 1):
        for y in range(board.shape[1]):
            if board[x, y] == "":
                continue
            if board[x, y][0] == "k" and board[x, y][1] == color:
                return (x, y)
    return (-1, -1)


def is_check(board, color):
    # Détermine si le joueur est en échec
    king_pos = find_king(color, board)
    if king_pos[0] == -1:
        return True
    opponent = "b" if color == "w" else "w"

    # Vérifiez si une pièce ennemie peut attaquer le roi
    for x in range(board.shape[0] - 1):
        for y in range(board.shape[1]):
            piece = board[x, y]
            if piece != "" and piece.startswith(opponent):
                # Vérifiez si une pièce ennemie peut attaquer la position du roi
                legal_moves = get_legal_moves_for_piece(board, (x, y), opponent)
                if king_pos in [t[1] for t in legal_moves]:
                    return True
    return False


def negamax(board, depth, alpha, beta, color, myColor, start_time, time_budget):
    """Negamax with quiescence search."""
    if depth <= 0 or time.time() - start_time > time_budget * 0.9:
        sc = (1 if color == myColor else -1) * evaluate_board(board, color)
        print("[Evaluation] score :", sc, " with board : \n ", board)
        return sc

    # if is_check(board,color):
    #   return (float("-inf") if color == myColor else float("inf"))

    max_eval = float("-inf")
    legal_moves = get_legal_moves(board, color)
    if not legal_moves:  # No valid moves
        return evaluate_board(board, myColor)  # Evaluate the current board state

    for move in legal_moves:
        s, e, _ = make_move(board, move)
        eval = -negamax(
            board,
            depth - 1,
            -beta,
            -alpha,
            "w" if color == "b" else "b",
            myColor,
            start_time,
            time_budget,
        )
        undo_move(board, move, e, s)

        max_eval = max(max_eval, eval)
        alpha = max(alpha, max_eval)

        if alpha >= beta:
            break  # Beta cut-off

    return max_eval


def get_legal_moves(board, color):
    moves = set()
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            piece = board[x, y]
            if piece and piece[1] == color:
                piece_type = piece[0].lower()
                if piece_type == "p":
                    moves.update(get_pawn_moves(board, x, y, color))
                elif piece_type == "r":
                    moves.update(get_rook_moves(board, x, y, color))
                elif piece_type == "n":
                    moves.update(get_knight_moves(board, x, y, color))
                elif piece_type == "b":
                    moves.update(get_bishop_moves(board, x, y, color))
                elif piece_type == "q":
                    moves.update(get_queen_moves(board, x, y, color))
                elif piece_type == "k":
                    moves.update(get_king_moves(board, x, y, color))
    return moves


def get_legal_moves_for_piece(board, position, color):
    """
    Retourne les mouvements légaux pour une pièce à une position donnée.
    """
    x, y = position
    piece = board[x, y]
    if not piece or piece[1] != color:
        return []

    piece_type = piece[0].lower()
    if piece_type == "p":
        return get_pawn_moves(board, x, y, color)
    elif piece_type == "n":
        return get_knight_moves(board, x, y, color)
    elif piece_type == "b":
        return get_bishop_moves(board, x, y, color)
    elif piece_type == "r":
        return get_rook_moves(board, x, y, color)
    elif piece_type == "q":
        return get_queen_moves(board, x, y, color)
    elif piece_type == "k":
        return get_king_moves(board, x, y, color)
    return []


def make_move(board, move):
    start, end = move
    endvalue = board[end]
    startvalue = board[start]
    board[end] = board[start]
    board[start] = ""
    return startvalue, endvalue, board


def undo_move(board, move, extrav, endv):
    start, end = move
    board[start] = endv
    board[end] = extrav
    return board


def get_pawn_moves(board, x, y, color):
    moves = set()
    direction = 1

    # Mouvement simple
    if is_valid_position(x + direction, y, board) and board[x + direction, y] == "":
        moves.add(((x, y), (x + direction, y)))

        """# Double mouvement depuis la position initiale
        if (color == 'w' and x == 6) or (color == 'b' and x == 1):
            if board[x + 2*direction, y] == '':
                moves.add(((x, y), (x + 2*direction, y)))
        """

    # Captures en diagonale
    for dy in [-1, 1]:
        new_x, new_y = x + direction, y + dy
        if (
            is_valid_position(new_x, new_y, board)
            and board[new_x, new_y] != ""
            and board[new_x, new_y][1] != color
        ):
            moves.add(((x, y), (new_x, new_y)))

    return moves


def get_rook_moves(board, x, y, color):
    moves = set()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        for i in range(1, 8):
            new_x, new_y = x + i * dx, y + i * dy
            if not is_valid_position(new_x, new_y, board):
                break
            if board[new_x, new_y] == "":
                moves.add(((x, y), (new_x, new_y)))
            elif board[new_x, new_y][1] != color:
                moves.add(((x, y), (new_x, new_y)))
                break
            else:
                break
    return moves


def get_knight_moves(board, x, y, color):
    moves = set()
    knight_moves = [
        (2, 1),
        (1, 2),
        (-1, 2),
        (-2, 1),
        (-2, -1),
        (-1, -2),
        (1, -2),
        (2, -1),
    ]
    for dx, dy in knight_moves:
        new_x, new_y = x + dx, y + dy
        if is_valid_position(new_x, new_y, board) and (
            board[new_x, new_y] == "" or board[new_x, new_y][1] != color
        ):
            moves.add(((x, y), (new_x, new_y)))
    return moves


def get_bishop_moves(board, x, y, color):
    moves = set()
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in directions:
        for i in range(1, 8):
            new_x, new_y = x + i * dx, y + i * dy
            if not is_valid_position(new_x, new_y, board):
                break
            if board[new_x, new_y] == "":
                moves.add(((x, y), (new_x, new_y)))
            elif board[new_x, new_y][1] != color:
                moves.add(((x, y), (new_x, new_y)))
                break
            else:
                break
    return moves


#   Simply move the pawns forward and tries to capture as soon as possible


def get_queen_moves(board, x, y, color):
    a = get_rook_moves(board, x, y, color)
    a.update(get_bishop_moves(board, x, y, color))
    return a


def get_king_moves(board, x, y, color):
    moves = set()
    king_moves = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    for dx, dy in king_moves:
        new_x, new_y = x + dx, y + dy
        if is_valid_position(new_x, new_y, board) and (
            board[new_x, new_y] == "" or board[new_x, new_y][1] != color
        ):
            moves.add(((x, y), (new_x, new_y)))
    # Note: Cette implémentation ne prend pas en compte le roque
    return moves


def chess_bot2(player_sequence, board, time_budget, **kwargs):
    start_time = time.time()
    color = player_sequence[1]
    depth = 3  # Profondeur initiale
    print(
        f"[Chess Bot] Starting with color {color} and time budget {time_budget} seconds."
    )

    best_move = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    best_moves = []
    best_scores = []
    legal_moves = get_legal_moves(board, color)
    while time.time() - start_time < time_budget * 0.95:
        for move in legal_moves:
            s, e, _ = make_move(board, move)
            eval = -negamax(
                board,
                depth - 1,
                float("-inf"),
                float("inf"),
                "w" if color == "b" else "b",
                color,
                start_time,
                time_budget,
            )
            undo_move(board, move, e, s)
            if eval >= best_score:
                best_score = eval
                best_move = move
                best_moves.append(move)
                best_scores.append(eval)
        depth += 1

    for i in range(0, len(best_scores)):
        print("move : ", best_moves[i], " score: ", best_scores[i], "\n")

    print(f"[Chess Bot] Best move: {best_move}, Score: {best_score}")
    return best_move if best_move else ((0, 0), (0, 0))


# Enregistrement du bot
register_chess_bot("AdvancedChessBot", chess_bot2)
