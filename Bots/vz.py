#
#   Example function to be implemented for
#       Single important function is next_best
#           color: a single character str indicating the color represented by this bot ('w' for white)
#           board: a 2d matrix containing strings as a descriptors of the board '' means empty location "XC" means a piece represented by X of the color C is present there
#           budget: time budget allowed for this turn, the function must return a pair (xs,ys) --> (xd,yd) to indicate a piece at xs, ys moving to xd, yd
#
import time
import random
from PyQt6 import QtCore

#   Be careful with modules to import from the root (don't forget the Bots.)
from Bots.ChessBotList import register_chess_bot

#   Simply move the pawns forward and tries to capture as soon as possible
def chess_bot(player_sequence, board, time_budget, **kwargs):
    start_time = time.time()
    possible_move = []
    color = player_sequence[1]

    def evaluate_move(start, end, board, path):
        x, y = end
        score = 0
        if board[x, y] == '':
            return 1
        else:
            piece = board[x, y][0]
            if piece[-1] != color:
                piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 100}
                score = 10 + piece_values.get(piece[0], 0)
        if len(path) > 2 and path[-2] == [end, start]:
            score -= 2
        return score + random.uniform(-0.1, 0.1)

    def findPath(path, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or time.time() - start_time > time_budget * 0.95:
            start, end = path[-1] if path else ((0,0), (0,0))
            return evaluate_move(start, end, board, path), path[-1] if path else ((0,0), (0,0))

        if maximizing_player:
            max_score = float('-inf')
            best_move = None

            for move in possible_move:
                start, end = move

                new_depth = depth - 1
                if board[end[0], end[1]] != '':
                    new_depth += 1

                original_start = board[start[0]][start[1]]
                original_end = board[end[0]][end[1]]
                board[end[0]][end[1]] = original_start
                board[start[0]][start[1]] = ''

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
            min_score = float('inf')
            best_move = None
            for move in possible_move:
                max_score = float('-inf')
            best_move = None

            for move in possible_move:
                start, end = move

                new_depth = depth - 1
                if board[end[0], end[1]] != '':
                    new_depth += 1

                original_start = board[start[0]][start[1]]
                original_end = board[end[0]][end[1]]
                board[end[0]][end[1]] = original_start
                board[start[0]][start[1]] = ''

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
        return 0 <= pos[0] < board.shape[0] and 0 <= pos[1] < board.shape[1]

    def generate_move(x, y, color, board):
        moves = []
        piece = board[x,y][0]
        if piece == "p": # Pawn
            if is_within_board([x + 1, y], board):
                if board[x + 1, y] == '':
                    moves.append([[x, y], [x + 1, y]])
                # Captures
                for dy in [-1, 1]:
                    if is_within_board([x + 1, y + dy], board) and board[x + 1, y + dy] != '' and board[x + 1, y + dy][-1] != color:
                        moves.append([[x, y], [x + 1, y + dy]])
        elif piece == "k": # King
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if is_within_board([new_x, new_y], board):
                        if board[new_x, new_y] == '' or board[new_x, new_y][-1] != color:
                            moves.append([[x, y], [new_x, new_y]])
        elif piece == "n":  # Knight
            moves = []
            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            for move in knight_moves:
                new_pos = [x + move[0], y + move[1]]
                if is_within_board(new_pos, board) and (board[new_pos[0], new_pos[1]] == '' or board[new_pos[0], new_pos[1]][-1] != color):
                    moves.append([[x,y],new_pos])
        elif piece in ["b", "r", "q"]: #bishop, Rook, Queen
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if piece == "b" else \
                        [(1, 0), (-1, 0), (0, 1), (0, -1)] if piece == "r" else \
                        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for direction in directions:
                for i in range(1, max(board.shape)):
                    new_x, new_y = x + direction[0] * i, y + direction[1] * i
                    if not is_within_board([new_x, new_y], board):
                        break
                    if board[new_x, new_y] == '':
                        moves.append([[x, y], [new_x, new_y]])
                    elif board[new_x, new_y][-1] != color:
                        moves.append([[x, y], [new_x, new_y]])
                        break
                    else:
                        break
        return moves

    for x in range(board.shape[0] - 1):
        for y in range(board.shape[1]):
            if board[x, y] != '' and board[x, y][-1] == color:
                possible_move.extend(generate_move(x, y, color, board))

    if not possible_move:
        return (0, 0), (0, 0)
    
    random.shuffle(possible_move)

    _, best_move = findPath([], board, 3, float('-inf'), float('inf'), True)
    if best_move:
        return best_move
    
    return (0, 0), (0, 0)

#   Example how to register the function
register_chess_bot("vz", chess_bot)