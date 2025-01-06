import numpy as np
import time
import random
from PyQt6 import QtCore

#   Be careful with modules to import from the root (don't forget the Bots.)
from Bots.ChessBotList import register_chess_bot

import os
file_name = "log_p2.txt"
file = open(file_name,"a+")

def print_log(text):
    print(text)
    file.write(text + "\n")
    file.flush()

transposition_table = {}

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
    # possible_move = []
    currentPlayerColor = player_sequence[1]
    piece_values = {
            'r': 500,    # Tour
            'n': 300,    # Cavalier
            'b': 300,    # Fou
            'q': 900,    # Reine
            'k': 10000,  # Roi
            'p': 50     # Pion
        }

    def get_board_hash(board):
        return hash(board.tobytes())

    pawnEval = [
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0],
        [1.0,  1.0,  2.0,  3.0,  3.0,  2.0,  1.0,  1.0],
        [0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
        [0.0,  0.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0],
        [0.5, -0.5, -1.0,  0.0,  0.0, -1.0, -0.5,  0.5],
        [0.5,  1.0,  1.0, -2.0, -2.0,  1.0,  1.0,  0.5],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
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

    kingEval = [
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
        [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
        [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0],
    ]

    
    def get_piece_value(piece,color,x,y):
        value = piece_values.get(piece,0)
        matrix = []
        if piece == "p":
            matrix= pawnEval
        elif piece == "r":
            matrix= rookEval
        elif piece == "n":
            matrix= knightEval
        elif piece == 'b':
            matrix= bishopEval
        elif piece == "k":
            matrix= knightEval
        elif piece == "q":
            matrix= evalQueen
        if color != currentPlayerColor:
            matrix = matrix[::-1]
        return  value + matrix[x][y]
            



    def generate_move(x,y,color,board):
        deplacements = []
        piece = board[x, y]

        if piece == "" or piece[1] != color:
            return []
        else:
            if piece == 'p':  # Pion
                    moves = pion(x, y, board, color)
                    deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == 'n':  # Cavalier
                moves = cavalier(x, y, board, color)
                deplacements = [((x, y, ),(nx, ny)) for nx, ny in moves]
            elif piece == 'k':  # Roi
                moves = roi(x, y, board, color)
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == 'r':  # Tour
                moves = tour(x, y, board, color)
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == 'b':  # Fou
                moves = fou(x, y, board, color)
                deplacements = [((x, y), (nx, ny)) for nx, ny in moves]
            elif piece == 'q':  # Reine
                moves = reine(x, y, board, color)
                deplacements = [((x, y),( nx, ny)) for nx, ny in moves]
        return deplacements

    
    def evaluate_development(board, color):
        score = 0
        back_rank = 0 if color == currentPlayerColor else 7
        for y in range(8):
            piece = board[back_rank, y]
            if piece != "" and piece[1] == color:
                if piece[0] == 'n':  # Cavalier
                    score -= 100
                elif piece[0] == 'b':  # Fou
                    score -= 100
                elif piece[0] == 'r':  # Tour
                    score -= 100
                elif piece[0] == 'q':  # Dame
                    score -= 50
    
        return score



        


    def evaluate_threats(board, color):
        threat_score = 0
        opponent_color = 'b' if color == 'w' else 'w'
        
        for x in range(8):
            for y in range(8):
                if board[x, y] != "" and board[x, y][1] == color:
                    moves = generate_move(x, y, color, board)
                    for move in moves:
                        target_x, target_y = move[1]
                        target_piece = board[target_x, target_y]
                        if target_piece != "" and target_piece[1] == opponent_color:
                            threat_score += get_piece_value(target_piece[0],target_piece[1],x,y) * 0.1
        
        return threat_score


    def evaluate_captures(board, color):
        bonus = 0
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] != "" and board[x, y][1] != color:
                    attackers = count_attackers(board, x, y, color)
                    if attackers > 0:
                        piece_value = piece_values.get(board[x, y][0], 0)
                        bonus += piece_value * 0.1 * attackers
        return bonus
    
    def count_attackers(board, x, y, color):
        return sum(1 for move in generate_moves(board, color) if move[0] == (x, y))


    

    def is_king_safe(board,color):
        """
        Vérifie si le roi est en sécurité.
        :param color_sign: +1 pour les blancs, -1 pour les noirs.
        :param Bo: Plateau.
        :return: True si le roi est en sécurité, False sinon.
        """
        king_position = None
        for x in range(8):
            for y in range(8):
                if board[x,y] == 'k'+color:
                    king_position = (x,y)
                    break
        if not king_position:
            return False  # Roi inexistant (cas rare)

        return not is_king_in_check(board,color,king_position)
    
    def is_king_in_check(board,color,king_position):
        opponent_moves = generate_moves(board,"w" if color == "b" else "b")
        for start,end in opponent_moves:
            if end[0] == king_position[0] and end[1] == king_position[1]:
                return True
        return False

    def is_checkmate(board,color): 
        return not is_king_safe(board,"w" if color == "b" else "b")
    
    def evaluate_board(board, color):
        MATE_SCORE = 100000
        score = 0
        # Évaluation existante des pièces
        for x in range(8):
            for y in range(8):
                piece = board[x, y]
                if piece != "":
                    piece_value = get_piece_value(piece[0],piece[1],x,y)
                    if piece[1] == color:
                        score += piece_value
                    else:
                        score -= piece_value


        if is_checkmate(board,color):  # Mat contre l'adversaire
            return MATE_SCORE
        if is_checkmate(board,"w" if color == 'b' else 'b'):  # Mat contre nous
                return -MATE_SCORE
            
        center_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for x, y in center_positions:
            piece = board[x,y]
            if piece == "":
                continue
            if piece[1] == color:
                score += 50  # Bonus pour contrôler le centre
            else:  # Pièce ennemie au centre
                score -= 50  # Malus si l'adversaire contrôle le centre
        
        # Bonus pour les captures potentielles
        score += 3*evaluate_captures(board, color)

        score += 2 * len(generate_moves(board,color))
        threat_score = 2*evaluate_threats(board, color)

        score += threat_score

        score += evaluate_development(board, color)

                          
        if is_king_safe(board,color):
            score += 400  # Bonus pour un roi en sécurité)
        
        return score
    def evaluate_captures(board, color):
        capture_score = 0
        for x in range(8):
            for y in range(8):
                piece = board[x, y]
                if piece != "" and piece[1] != color:
                    attackers = count_attackers(board, x, y, color)
                    if attackers > 0:
                        capture_score += get_piece_value(piece[0],piece[1],x,y) * 1 * attackers
        return capture_score
    

    def sort_moves(board, moves, color):
        def move_score(move):
            start, end = move
            piece = board[start[0], start[1]]
            target = board[end[0], end[1]]
            score = 0
            
            # Priorité aux captures
            if target != "":
                score += 1000 + get_piece_value(target[0],target[1],end[0],end[1]) - get_piece_value(piece[0],piece[1],start[0],start[1])

            
                # Bonus pour les mouvements vers le centre
            if end[0] in [3, 4] and end[1] in [3, 4]:
                score += 75  # Augmenter le bonus pour le contrôle du centre
            
            # Bonus pour le développement des pièces
            if piece[0] in ['n', 'b', 'r'] and start[0] in [0, 7]:
                score += 50  # Bonus pour développer les pièces de la rangée arrière
            
            return score

    
        return sorted(moves, key=move_score, reverse=True)



    

    def is_pawn_protected(x, y, board,color):
        """
        Vérifie si un pion est protégé par un autre pion.
        :param x: Coordonnée x du pion.
        :param y: Coordonnée y du pion.
        :param Bo: Plateau.
        :param color_sign: +1 pour les blancs, -1 pour les noirs.
        :return: True si le pion est protégé, False sinon.
        """
        for dx, dy in [(1, -1), (1, 1)] if color == currentPlayerColor else [(-1, -1), (-1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8 and board[nx,ny] == 'p'+color:
                return True
        return False
    
    def is_pawn_isolated(x, y, board):
        """
        Vérifie si un pion est isolé.
        :param x: Coordonnée x du pion.
        :param y: Coordonnée y du pion.
        :param Bo: Plateau.
        :return: True si le pion est isolé, False sinon.
        """
        for dy in [-1, 1]:  # Colonnes adjacentes
            ny = y + dy
            if 0 <= ny < 8:
                for nx in range(8):  # Vérifie toute la colonne adjacente
                    if board[nx,ny] == board[x,y]:
                        return False
        return True
    


    def is_pawn_passed(x, y, board,color):
        """
        Vérifie si un pion est passé.
        :param x: Coordonnée x du pion.
        :param y: Coordonnée y du pion.
        :param Bo: Plateau.
        :param color_sign: +1 pour les blancs, -1 pour les noirs.
        :return: True si le pion est passé, False sinon.
        """
        direction = 1 if color == currentPlayerColor else -1
        for nx in range(x + direction, 8 if color == currentPlayerColor == 1 else -1, direction):
            for dy in [-1, 0, 1]:
                ny = y + dy
                if board[nx,nx] != "" and board[nx,ny][1] != color and board[nx,ny][0] == 'p':
                    return False  # Pion adverse peut bloquer ou capturer
        return True
        

    
    def make_move(board,move):
        start,end = move
        from copy import deepcopy
        new_board = deepcopy(board)
        new_board[start[0],start[1]] = new_board[end[0],end[1]]
        return new_board
        

    def minimaxRoot(depth, board, isMaximizing,maximizing_color):
        newGameMoves = generate_moves(
            board, maximizing_color
        )
        print(newGameMoves,"moves__")
        bestScore = float("-inf")
        bestMove = None

        for move in newGameMoves:
            new_board = make_move(board,move)
            try:

                score = minimax(
                    new_board, depth - 1, float("-inf"), float("inf"),isMaximizing,maximizing_color
                )
                if score >= bestScore:
                    bestScore = score
                    bestMove = move
            except Exception as e:
                print(str(e))
                break
        print_log(f"[Chess Bot] Best move: {bestMove}, Score: {bestScore}")
        return bestScore,bestMove

    def minimax(board, depth, alpha, beta, maximizing_player, maximizing_color):
        if time.time() - start_time > time_budget * 0.95:  # Utilisez 95% du temps alloué
            raise TimeoutError("Time limit exceeded")

        if depth == 0 or time.time() - start_time > time_budget * 0.9:
            score = evaluate_board(board, maximizing_color)
            print_log(f"Leaf evaluation at depth {depth}: {score}")
            return score

        current_color = maximizing_color if maximizing_player else ('b' if maximizing_color == 'w' else 'w')
        moves = generate_moves(board, current_color)
        
        best_score = float('-inf') if maximizing_player else float('inf')
        
        for move in moves:
            new_board = make_move(board, move)
            score = minimax(new_board, depth - 1, alpha, beta, not maximizing_player, maximizing_color)
            
            if maximizing_player:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                print_log(f"Max node at depth {depth}, move {move}: {score} (best: {best_score})")
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                print_log(f"Min node at depth {depth}, move {move}: {score} (best: {best_score})")
                
            if beta <= alpha:
                print_log(f"Pruning at depth {depth}")
                break
        return best_score
    
    def evaluate_pawn_threats(board,color):
        score = 0
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] != "" and board[x, y][0] == 'p':
                    clr = board[x, y][1]
                    direction = -1 if clr != color else 1
                    for dy in [-1, 1]:
                        threat_x, threat_y = x + direction, y + dy
                        if is_within_board((threat_x, threat_y), board):
                            target = board[threat_x, threat_y]
                            if target != "" and target[1] != clr:
                                score += 5 if clr == color else -5
        return score


    def is_within_board(pos, board):
        """
        Check if a position is within the boundaries of the board.

        :param pos: Position to check as a tuple (x, y).
        :param board: Current state of the chess board as a numpy array.
        :return: True if the position is valid, False otherwise.
        """
        return 0 <= pos[0] < board.shape[0] and 0 <= pos[1] < board.shape[1]

    def generate_moves(board, color):
        """
        Génère tous les déplacements possibles pour un joueur donné.
        :param Bo: Plateau
        :param color_sign: +1 pour les blancs, -1 pour les noirs
        :return: Liste des déplacements possibles
        """
        deplacements = []
        for x in range(8):
            for y in range(8):
                if board[x,y] == "":
                    continue
                piece = board[x,y][0]
                clr = board[x,y][1]
                if color == clr:
                    if piece == 'p':  # Pion
                        moves = pion(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])
                    elif piece == 'n':  # Cavalier
                        moves = cavalier(x, y, board, color)
                        deplacements.extend([((x, y, ),(nx, ny)) for nx, ny in moves])
                    elif piece == 'k':  # Roi
                        moves = roi(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])
                    elif piece == 'r':  # Tour
                        moves = tour(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])
                    elif piece == 'b':  # Fou
                        moves = fou(x, y, board, color)
                        deplacements.extend([((x, y), (nx, ny)) for nx, ny in moves])
                    elif piece == 'q':  # Reine
                        moves = reine(x, y, board, color)
                        deplacements.extend([((x, y),( nx, ny)) for nx, ny in moves])

        return sort_moves(board,deplacements,color)
    

    def cavalier(pos_x, pos_y, board, color):
        mouvements = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        deplacements = []
        for dx, dy in mouvements:
            nx, ny = pos_x + dx, pos_y + dy
            if 0 <= nx <= 7 and 0 <= ny <= 7:
                piece = board[nx,ny]
                if piece == '' or (piece[1] != color):  # Vide ou pièce ennemie
                    deplacements.append((nx, ny))
        return deplacements

    def pion(pos_x, pos_y, board,color):
        deplacements = []
        direction = 1 if color == currentPlayerColor else -1

        # Avance simple
        nx = pos_x + direction
        if 0 <= nx <= 7 and board[nx,pos_y] == "":  # La case devant est vide
            deplacements.append((nx, pos_y))

        # Captures diagonales
        for dy in [-1, 1]:
            nx, ny = pos_x + direction, pos_y + dy
            if 0 <= nx <= 7 and 0 <= ny <= 7:
                if board[nx,ny] != "" and board[nx,ny][1] != color:
                    deplacements.append((nx, ny))
        return deplacements

    def roi(pos_x, pos_y, board, color):
        mouvements = [
            (-1, -1), (-1, 0), (-1, 1),  # Haut gauche, haut, haut droite
            (0, -1),          (0, 1),   # Gauche, droite
            (1, -1), (1, 0), (1, 1)     # Bas gauche, bas, bas droite
        ]
        deplacements = []

        for dx, dy in mouvements:
            nx, ny = pos_x + dx, pos_y + dy
            if 0 <= nx <= 7 and 0 <= ny <= 7:  # Vérifie que la position reste sur le plateau
                piece = board[nx,ny]
                if piece == "" or piece[1] != color:  # Case vide ou occupée par une pièce ennemie
                    deplacements.append((nx, ny))

        return deplacements

    def tour(pos_x, pos_y, board, color):
        """
        Génère les déplacements possibles pour une tour.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Haut, Bas, Gauche, Droite
        return get_moves_directions(board, pos_x, pos_y, color, directions)

    def fou(pos_x, pos_y, board, color):
        """
        Génère les déplacements possibles pour un fou.
        """
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonales
        return get_moves_directions(board, pos_x, pos_y, color, directions)

    def reine(pos_x, pos_y, board, color):
        """
        Génère les déplacements possibles pour une reine.
        """
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Directions de la tour
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Directions du fou
        ]
        return get_moves_directions(board, pos_x, pos_y, color, directions)

    def get_moves_directions(board, pos_x, pos_y, color, directions):
        """
        Génère les déplacements possibles dans les directions spécifiées.
        """
        moves = []
        for dx, dy in directions:
            nx, ny = pos_x + dx, pos_y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                piece = board[nx,ny]
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
    


    piece_count = sum(1 for x in range(board.shape[0]) for y in range(board.shape[1]) if board[x, y] != "")



    best_score = float("-inf")
    best_move = None
    for depth in range(2,8):
        if time.time() - start_time > time_budget * 0.8:
            break
        current_score,current_best_move = minimaxRoot(depth, board, True,currentPlayerColor)
        if current_score > best_score:
            best_move= current_best_move
            best_score = current_score


    return best_move if best_move != None else ((0, 0), (0, 0))


#   Example how to register the function
register_chess_bot("vz-m", chess_bot)
