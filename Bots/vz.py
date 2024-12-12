
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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ChessRules import *

#   Simply move the pawns forward and tries to capture as soon as possible
def chess_bot(player_sequence, board, time_budget, **kwargs):

    color = player_sequence[1]
    for x in range(board.shape[0]-1):
        for y in range(board.shape[1]):
            if board[x,y][0] == "p":
                if board[x,y] != "p"+color:
                    continue
                if move_is_valid(player_sequence[0:4], [[x,y], [x+1,y-1]], board):
                    return (x,y), (x+1,y-1)
                if move_is_valid(player_sequence[0:4], [[x,y], [x+1,y+1]], board):
                    return (x,y), (x+1,y+1)
                elif board[x+1,y] == '':
                    return (x,y), (x+1,y)
            elif board[x,y][0] == "k":
                if board[x,y] != "k"+color:
                    continue
                if move_is_valid(player_sequence[0:4], [[x,y],[x+1,y-1]], board): #diag bas-gauche
                    return(x,y), (x+1,y-1)
                elif move_is_valid(player_sequence[0:4], [[x,y], [x+1,y]], board): #bas
                    return(x,y), (x+1,y)
                elif move_is_valid(player_sequence[0:4], [[x,y], [x+1,y+1]], board): #diag bas-droite
                    return (x,y), (x+1,y+1)
                elif move_is_valid(player_sequence[0:4], [[x,y], [x,y-1]], board): #gauche
                    return (x,y), (x,y-1)
                elif move_is_valid(player_sequence[0:4], [[x,y], [x,y+1]], board): #droite
                    return (x,y), (x,y+1)
                elif move_is_valid(player_sequence[0:4], [[x,y], [x-1,y-1]], board): #diag haut-gauche
                    return (x,y), (x-1,y-1)
                elif move_is_valid(player_sequence[0:4], [[x,y], [x-1,y]], board): #haut
                    return (x,y), (x-1,y)
                elif move_is_valid(player_sequence[0:4], [[x,y], [x-1,y+1]], board): #diag haut-droite
                    return (x,y), (x-1,y+1)
            elif board[x,y][0] == "r":
                if board[x,y] != "r" + color:
                    continue

                


    return (0,0), (0,0)

#   Example how to register the function
register_chess_bot("vz", chess_bot)