
#
#   Example function to be implemented for
#       Single important function is next_best
#           color: a single character str indicating the color represented by this bot ('w' for white)
#           board: a 2d matrix containing strings as a descriptors of the board '' means empty location "XC" means a piece represented by X of the color C is present there
#           budget: time budget allowed for this turn, the function must return a pair (xs,ys) --> (xd,yd) to indicate a piece at xs, ys moving to xd, yd
#

from PyQt6 import QtCore

#   Be careful with modules to import from the root (don't forget the Bots.)
from Bots.ChessBotList import register_chess_bot

#   Simply move the pawns forward and tries to capture as soon as possible
def chess_bot(player_sequence, board, time_budget, **kwargs):

    possible_move = []
    color = player_sequence[1]
    def evaluate_move(start, end, board):
        x,y = end
        
        if board[x,y] == '':
            return 1
        piece = board[x,y][0]
        if piece[-1] != color:
            return 10
        elif piece[-1] != color and piece[0] == "k":
            return 50
        return 0

    def findPath(path,board, depth):
        if depth == 0:
            return evaluate_move(*path[-1], board), path[-1]
        
        max_score = float('-inf')
        best_move = None

        for move in possible_move:
            start, end = move

            board[end[0],end[1]] = board[start[0],start[1]]
            board[start[0], start[1]] = ''

            score, _ = findPath(path + [move], board, depth - 1)

            board[start[0], start[1]] = board[end[0], end[1]]
            board[end[0], end[1]] = ''

            if score > max_score:
                max_score = score
                best_move = move

        return max_score, best_move
    
    def generate_move(x,y,color, board):
        moves = []
        direction = 1 if color == "w" else -1
        if y > 0 and board[x+direction,y-1] != '' and board[x+direction,y-1][-1] != color:
            moves.append([[x,y],[x+1,y-1]])
        if y < board.shape[1] - 1 and board[x+direction,y+1] != '' and board[x+direction,y+1][1] != color:
            moves.append([[x,y],[x+1,y+1]])
        elif board[x+direction,y] == '':
            moves.append([[x,y],[x+1,y]])
        return moves


    for x in range(board.shape[0]-1):
        for y in range(board.shape[1]):
            if board[x,y] == "p"+color:
                possible_move.extend(generate_move(x,y,color,board))
    
    _, best_move = findPath([], board, 3)
    if best_move:
        return best_move
    return (0,0), (0,0)

#   Example how to register the function
register_chess_bot("vz", chess_bot)