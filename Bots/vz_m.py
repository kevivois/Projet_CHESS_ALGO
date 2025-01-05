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
        "p": 100,  
        "n": 320,  
        "b": 330,  
        "r": 500,
        "q": 900,
        "k": 20000,  
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

    
    def evaluate_center_control(board,color):
        score = 0
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for x, y in center_squares:
            piece = board[x, y]
            if piece != "":
                if piece[1] == color:
                    score += 5
                else:
                    score -= 5
        return score
    


    def get_move_count(board):
        sum(1 for row in board for piece in row if piece != '')

    def evaluate_board(board, color):
        score = 0
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] != "":
                    piece = board[x,y]
                    piece_type = piece[0]
                    piece_color = piece[1]
                
                    
                    # Valeur de base + valeur positionnelle
                    position_score = 0
                    if piece_type == 'p':
                        position_score = pawnEval[x][y]
                    elif piece_type == 'n':
                        position_score = knightEval[x][y]
                    elif piece_type == 'b':
                        position_score = bishopEval[x][y]
                    elif piece_type == 'r':
                        position_score = rookEval[x][y]
                    elif piece_type == 'q':
                        position_score = evalQueen[x][y]
                    elif piece_type == 'k':
                        position_score = kingEval[x][y]
                    
                    # Valeur totale de la pièce
                    value = piece_values.get(piece_type, 0) + position_score

                    capture_bonus = get_capture_bonus(board, x, y, color)
                    value += capture_bonus
                    
                    # Crucial : -1 pour les pièces adverses
                    multiplier = 1 if piece_color == color else -1
                    score += value * multiplier

        score += evaluate_center_control(board,color) * 5

        score += evaluate_pawn_threats(board,color) * 5

        mobility = len(generate_all_moves(board, color))

        score += mobility * 0.1
    
        return score
    


    def get_capture_bonus(board, x, y, color):
        bonus = 0
        piece = board[x, y]
        if piece[1] != color:  # Si c'est une pièce adverse
            attackers = count_attackers(board, x, y, color)
            if attackers > 0:
                piece_value = piece_values.get(piece[0], 0)
                bonus = piece_value * 2 * attackers  # 10% de bonus par attaquant
                if piece[0] == 'p':
                    bonus *= 2  # Double bonus pour la capture de pions
        return bonus
    
    def count_attackers(board, x, y, color):
        return sum(1 for move in generate_all_moves(board, color) if move[0] == (x, y))
        

    
    def make_move(board,move):
        start,end = move
        from copy import deepcopy
        new_board = deepcopy(board)
        new_board[start[0],start[1]] = new_board[end[0],end[1]]
        return new_board
        

    def minimaxRoot(depth, board, isMaximizing,maximizing_color):
        newGameMoves = generate_all_moves(
            board, maximizing_color
        )
        bestScore = float("-inf")
        bestMove = None

        for move in newGameMoves:
            new_board = make_move(board,move)
            try:

                score = minimax(
                    new_board, depth - 1, float("-inf"), float("inf"),isMaximizing,currentPlayerColor
                )
                if score >= bestScore:
                    bestScore = score
                    bestMove = move
            except:
                break
        print_log(f"[Chess Bot] Best move: {bestMove}, Score: {bestScore}")
        return bestScore,bestMove

    def minimax(board, depth, alpha, beta, maximizing_player, maximizing_color):
        if time.time() - start_time > time_budget * 0.95:  # Utilisez 95% du temps alloué
            raise TimeoutError("Time limit exceeded")
        board_hash = get_board_hash(board)
        if board_hash in transposition_table:
            entry = transposition_table[board_hash]
            if entry['depth'] >= depth:
                if entry['flag'] == 'EXACT':
                    return entry['score']
                elif entry['flag'] == 'LOWERBOUND' and entry['score'] >= beta:
                    return entry['score']
                elif entry['flag'] == 'UPPERBOUND' and entry['score'] <= alpha:
                    return entry['score']

        if depth == 0 or time.time() - start_time > time_budget * 0.9:
            score = evaluate_board(board, maximizing_color)
            transposition_table.update({
            board_hash: {
                'score': score,
                'depth': depth,
                'flag': 'EXACT'
            }
        })
            print_log(f"Leaf evaluation at depth {depth}: {score}")
            return score

        current_color = maximizing_color if maximizing_player else ('b' if maximizing_color == 'w' else 'w')
        moves = generate_all_moves(board, current_color)
        
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

            flag = 'EXACT'
        if best_score <= alpha:
            flag = 'UPPERBOUND'
        elif best_score >= beta:
            flag = 'LOWERBOUND'

        transposition_table.update({
            board_hash: {
                'score': best_score,
                'depth': depth,
                'flag': flag
            }
        })
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

    def generate_move(x, y, color, board):
        moves = []
        piece = board[x, y][0]
        
        if piece == "p":  # Pawn
            direction = 1 if color == currentPlayerColor else -1
            
            # Avancée simple
            new_x = x + direction
            if is_within_board([new_x, y], board) and board[new_x, y] == "":
                moves.append(((x, y), (new_x, y)))
                
            
            # Captures en diagonale
            for dy in [-1, 1]:
                capture_x = x + direction
                capture_y = y + dy
                if is_within_board([capture_x, capture_y], board):
                    target = board[capture_x, capture_y]
                    if target != "" and target[1] != color:
                        moves.append(((x, y), (capture_x, capture_y)))
                        
        elif piece in ["b", "r", "q"]:  # Bishop, Rook, Queen
            directions = []
            if piece in ["b", "q"]:  # Bishop/Queen moves
                directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            if piece in ["r", "q"]:  # Rook/Queen moves
                directions.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                while is_within_board((new_x, new_y), board):
                    target = board[new_x, new_y]
                    if target == "":
                        moves.append(((x, y), (new_x, new_y)))
                    elif target != "" and target[1] != color:
                        moves.append(((x, y), (new_x, new_y)))
                        break
                    else:  # Pièce alliée
                        break
                    new_x, new_y = new_x + dx, new_y + dy
                    
        elif piece == "n":  # Knight
            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                        (1, 2), (1, -2), (-1, 2), (-1, -2)]
            for dx, dy in knight_moves:
                new_x, new_y = x + dx, y + dy
                if is_within_board((new_x, new_y), board):
                    target = board[new_x, new_y]
                    if target == "" or (target != "" and target[1] != color):
                        moves.append(((x, y), (new_x, new_y)))
                        
        elif piece == "k":  # King
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if is_within_board((new_x, new_y), board):
                        target = board[new_x, new_y]
                        if target == "" or (target != "" and target[1] != color):
                            moves.append(((x, y), (new_x, new_y)))
                            
        # Debug des mouvements générés
        if moves:
            print_log(f"Generated {len(moves)} moves for {piece}{color} at {x},{y}")
            
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

    def generate_all_moves(board, color):
        result = []
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] != "" and board[x, y][-1] == color:
                    result.extend(generate_move(x, y, color, board))
        return result
    


    piece_count = sum(1 for x in range(board.shape[0]) for y in range(board.shape[1]) if board[x, y] != "")



    best_move = None
    best_score = float("-inf")
    for depth in range(2,10):
        if time.time() - start_time > time_budget * 0.8:
            break
        current_score,current_best_move = minimaxRoot(depth, board, True,currentPlayerColor)
        if current_score > best_score:
            best_move= current_best_move




    return best_move if best_move else ((0, 0), (0, 0))


#   Example how to register the function
register_chess_bot("vz-m", chess_bot)
