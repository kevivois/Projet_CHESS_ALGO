import numpy as np
import time
import random
from Bots.ChessBotList import register_chess_bot

# Valeurs des pièces
PIECE_VALUES = {
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000
}

# Bonus de position pour chaque type de pièce
POSITION_BONUS = {
    'p': np.array([
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ]),
    # Bonus pour d'autres pièces (facultatif)
}

def is_valid_position(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def is_empty_or_enemy(board, x, y, color):
    return board[x, y] == '' or board[x, y][1] != color

# Fonction pour évaluer le développement des pions
def evaluate_pawn_development(board, color):
    score = 0
    pawn_columns = set()
    enemy_color = 'B' if color == 'W' else 'W'
    
    for x in range(8):
        for y in range(8):
            if board[x, y] == f'p{color}':
                if 2 <= y <= 5:
                    score += 10 * (x if color == 'W' else 7 - x)
                if y in pawn_columns:
                    score -= 20
                pawn_columns.add(y)
                if all(board[i, y] != f'p{enemy_color}' for i in range(8)):
                    score += 50
                if y > 0 and y < 7:
                    if all(board[i, y-1] != f'p{color}' and board[i, y+1] != f'p{color}' for i in range(8)):
                        score -= 30
    
    center_control = sum(1 for x in [3, 4] for y in [3, 4] if board[x, y] == f'p{color}')
    score += center_control * 20
    
    return score

def evaluate_pawn_structure(board, color):
    score = 0
    for y in range(8):
        pawn_chain = 0
        for x in range(8):
            if board[x, y] == f'p{color}':
                if pawn_chain > 0:
                    score += 10
                pawn_chain += 1
            else:
                pawn_chain = 0
    return score

def evaluate_board(board, color, move=None):
    score = 0
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            piece = board[x, y]
            if piece:
                piece_type = piece[0].lower()
                piece_color = piece[1]
                value = PIECE_VALUES.get(piece_type, 0)
                if piece_color == color:
                    score += value
                    score += POSITION_BONUS.get(piece_type, np.zeros((8, 8)))[x][y]
                else:
                    score -= value
                    score -= POSITION_BONUS.get(piece_type, np.zeros((8, 8)))[7-x][y]
    
    score += evaluate_pawn_development(board, color)
    score += evaluate_pawn_structure(board, color)
    
    print(f"[Evaluation] Score for color {color}: {score}")
    return score

def minimax(board, depth, alpha, beta, maximizing_player, color):
    print(f"[Minimax] Called at depth {depth}, alpha={alpha}, beta={beta}, maximizing={maximizing_player}")
    if depth <= 0:
        evaluation = evaluate_board(board, color)
        print(f"[Minimax] Depth 0 reached. Evaluation: {evaluation}")
        return evaluation

    if maximizing_player:
        max_eval = float('-inf')
        for move in get_legal_moves(board, color):
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, alpha, beta, False, color)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            print(f"[Maximizing] Move: {move}, Eval: {eval}, Alpha: {alpha}, Beta: {beta}")
            if beta <= alpha:
                print(f"[Pruning] Beta <= Alpha (Maximizing). Breaking.",alpha,beta)
                break
        print(f"[Returning] max_eval : ",max_eval)
        return max_eval
    else:
        min_eval = float('inf')
        opponent_color = 'b' if color == 'w' else 'w'
        for move in get_legal_moves(board, opponent_color):
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, alpha, beta, True, color)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            print(f"[Minimizing] Move: {move}, Eval: {eval}, Alpha: {alpha}, Beta: {beta}")
            if beta <= alpha:
                print(f"[Pruning] Beta <= Alpha (Minimizing). Breaking.",alpha,beta)
                break
        print(f"[Returning] min_eval : ",min_eval)
        return min_eval
    
def get_legal_moves(board, color):
    moves = set()
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            piece = board[x, y]
            if piece and piece[1] == color:
                piece_type = piece[0].lower()
                if piece_type == 'p':
                    moves.update(get_pawn_moves(board, x, y, color))
                elif piece_type == 'r':
                    moves.update(get_rook_moves(board, x, y, color))
                elif piece_type == 'n':
                    moves.update(get_knight_moves(board, x, y, color))
                elif piece_type == 'b':
                    moves.update(get_bishop_moves(board, x, y, color))
                elif piece_type == 'q':
                    moves.update(get_queen_moves(board, x, y, color))
                elif piece_type == 'k':
                    moves.update(get_king_moves(board, x, y, color))
    return list(moves)

def make_move(board, move):
    new_board = board.copy()
    start, end = move
    new_board[end] = new_board[start]
    new_board[start] = ''
    return new_board
def get_pawn_moves(board, x, y, color):
    moves = []
    direction = -1 if color == 'W' else 1
    
    # Mouvement simple
    if is_valid_position(x + direction, y) and board[x + direction, y] == '':
        moves.append(((x, y), (x + direction, y)))
        
        # Double mouvement depuis la position initiale
        if (color == 'W' and x == 6) or (color == 'B' and x == 1):
            if board[x + 2*direction, y] == '':
                moves.append(((x, y), (x + 2*direction, y)))
    
    # Captures en diagonale
    for dy in [-1, 1]:
        new_x, new_y = x + direction, y + dy
        if is_valid_position(new_x, new_y) and board[new_x, new_y] != '' and board[new_x, new_y][1] != color:
            moves.append(((x, y), (new_x, new_y)))
    
    return moves

def get_rook_moves(board, x, y, color):
    moves = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        for i in range(1, 8):
            new_x, new_y = x + i*dx, y + i*dy
            if not is_valid_position(new_x, new_y):
                break
            if board[new_x, new_y] == '':
                moves.append(((x, y), (new_x, new_y)))
            elif board[new_x, new_y][1] != color:
                moves.append(((x, y), (new_x, new_y)))
                break
            else:
                break
    return moves

def get_knight_moves(board, x, y, color):
    moves = []
    knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    for dx, dy in knight_moves:
        new_x, new_y = x + dx, y + dy
        if is_valid_position(new_x, new_y) and (board[new_x, new_y] == '' or board[new_x, new_y][1] != color):
            moves.append(((x, y), (new_x, new_y)))
    return moves

def get_bishop_moves(board, x, y, color):
    moves = []
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in directions:
        for i in range(1, 8):
            new_x, new_y = x + i*dx, y + i*dy
            if not is_valid_position(new_x, new_y):
                break
            if board[new_x, new_y] == '':
                moves.append(((x, y), (new_x, new_y)))
            elif board[new_x, new_y][1] != color:
                moves.append(((x, y), (new_x, new_y)))
                break
            else:
                break
    return moves

def get_queen_moves(board, x, y, color):
    return get_rook_moves(board, x, y, color) + get_bishop_moves(board, x, y, color)

def get_king_moves(board, x, y, color):
    moves = []
    king_moves = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    for dx, dy in king_moves:
        new_x, new_y = x + dx, y + dy
        if is_valid_position(new_x, new_y) and (board[new_x, new_y] == '' or board[new_x, new_y][1] != color):
            moves.append(((x, y), (new_x, new_y)))
    # Note: Cette implémentation ne prend pas en compte le roque
    return moves

def chess_bot(player_sequence, board, time_budget, **kwargs):
    start_time = time.time()
    color = player_sequence[1]
    depth = 3  # Profondeur initiale
    print(f"[Chess Bot] Starting with color {color} and time budget {time_budget} seconds.")
    
    best_move = None
    best_score = float('-inf')
    legal_moves = get_legal_moves(board, color)
    while time.time() - start_time < time_budget * 0.95:
        for move in legal_moves:
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, float('-inf'), float('inf'), False, color)
            if eval > best_score:
                best_score = eval
                best_move = move
        depth += 1
        if depth >= 5:
            break
        print("sah2")

    print(f"[Chess Bot] Best move: {best_move}, Score: {best_score}")
    return best_move if best_move else ((0, 0), (0, 0))

# Enregistrement du bot
register_chess_bot("AdvancedChessBot", chess_bot)
