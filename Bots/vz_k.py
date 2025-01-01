import numpy as np
import time
import random
from PyQt6 import QtCore

#   Be careful with modules to import from the root (don't forget the Bots.)
from Bots.ChessBotList import register_chess_bot

CHESS_PIECES_NAMES = {
    "k": "King",
    "q": "Queen",
    "n": "Knight",
    "b": "Bishop",
    "r": "Rook",
    "p": "Pawn",
}


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
    myColor = player_sequence[1]
    piece_values = {
        "p": 10,
        "n": 30,
        "b": 30,
        "r": 50,
        "q": 200,
        "k": 900,
    }

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

    def getPieceEval(piece, x, y):
        if piece == "":
            return 0
        value = piece_values.get(piece[0], 0)
        if piece == "p":
            return value + pawnEval[x][y]
        elif piece == "n":
            return value + knightEval[x][y]
        elif piece == "b":
            return value + bishopEval[x][y]
        elif piece == "r":
            return value + rookEval[x][y]
        elif piece == "q":
            return value + evalQueen[x][y]
        elif piece == "k":
            return value + kingEval[x][y]
        return value

    def evaluate_move(board, path, color):
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
        score = 0
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x,y] != "":
                    t = getPieceEval(board[x,y][0],x,y)
                    score += (1 if board[x,y][1] == color else -1) * t
                    if board[x, y][1] == color:
                        moves = generate_move(x, y, color, board)
                        for start,end in moves:
                            if board[end[0],end[1]] != '':
                                score += 2*getPieceEval(board[end[0],end[1]][0], end[0], end[1])
                        score += len(moves) * 0.1  # Small bonus for each possible move 
        
        # Penalty for repetitive moves
        if len(path) > 2 and path[-2] == path[-1][::-1]:
            score -= 2
        if is_king_in_check(board, color):
            score -= 5000  # High penalty if the king is in check

        # Add a small random factor to the score to introduce slight variability
        return score + random.uniform(-0.1, 0.1)




    def quiescence_search(board, alpha, beta, evaluate, is_terminal, generate_moves):
        """
        Perform quiescence search at the current position.

        Args:
            position: The current game position.
            alpha: The alpha value for alpha-beta pruning.
            beta: The beta value for alpha-beta pruning.
            evaluate: A function to evaluate the position statically.
            is_terminal: A function to check if the position is terminal.
            generate_moves: A function to generate tactical moves (like captures).

        Returns:
            The static evaluation score of the position after quiescence search.
        """
        # Evaluate the current position statically
        stand_pat = evaluate(board)
        
        # If the static evaluation is outside the alpha-beta bounds, return it
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Generate all tactical moves (e.g., captures)
        moves = generate_move(board,myColor)

        for move in moves:
            new_board = make_move(board,move)
            if is_king_in_check(new_board,myColor):
                continue  # Skip terminal positions

            # Recursively evaluate the position after the move
            score = -quiescence_search(new_board, -beta, -alpha, evaluate, is_terminal, generate_moves)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha
    

    def make_move(board,move):
        start,end = move
        new_board = board.copy()
        new_board[end[0],end[1]] = new_board[start[0],start[1]]
        new_board[start[0],start[1]] = ''
        return new_board
    
    def sort_immediate_moves(board,moves,color):
        return sorted(moves,key=lambda m: pre_evaluate_move(board,m,color),reverse=True)
    
    def pre_evaluate_move(board,move,color):
        start,end = move
        piece_start = board[start[0],start[1]]
        piece_end = board[end[0],end[1]]
        if piece_end == "":
            return 0
        value = getPieceEval(piece_start[0],start[0],start[1]) - getPieceEval(piece_end[0],end[0],end[1])
        return (-1 if color == piece_start[1] else 1) * value



    def findPath(board, depth, alpha, beta, maximizing_player,path=None):
        if path == None:
            path = []
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
        actual_color = myColor if maximizing_player else ("b" if myColor == "w" else "w")
        if depth == 0 or time.time() - start_time > time_budget * 0.9:
            score = evaluate_move(
                board,
                path,
                actual_color
            ) 
            b = board[path[-1][-1][0],path[-1][-1][1]]
            print("[Evaluation] moving "+ actual_color + " " + CHESS_PIECES_NAMES.get(b[0] if b != "" else "","none")  + " from " + str(path[-1][0]) + " to " +str(path[-1][1])," | score : ",score)
            return score, path[-1] if path else ((0, 0), (0, 0))
        
        moves = generate_all_moves(board,actual_color)

        moves = sort_immediate_moves(board,moves,actual_color)

        if maximizing_player:

            max_score = float("-inf")
            best_move = None
            for move in moves:
                start, end = move

                new_depth = depth - 1
                if board[end[0], end[1]] != "":
                    new_depth += 1
                new_board = make_move(board,move)
                score, _ = findPath(new_board, new_depth, alpha, beta,not maximizing_player,path + [move])

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

            for move in moves:
                start, end = move

                new_depth = depth - 1
                if board[end[0], end[1]] != "":
                    new_depth += 1

                new_board = make_move(board,move)

                score, _ = findPath(new_board, new_depth, alpha, beta,maximizing_player,path + [move])

                if score < min_score:
                    min_score = score
                    best_move = move

                beta = min(beta, min_score)
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
                    elif board[new_x, new_y] != "" and board[new_x, new_y][1] != color:
                        moves.append([[x, y], [new_x, new_y]])
                        break
                    else:
                        break
                    new_x, new_y = new_x + dx, new_y + dy

        return moves
    
    def generate_all_moves(board,color):
        result = []
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x, y] != "" and board[x, y][-1] == color:
                    result.extend(generate_move(x, y, color, board))
        return result

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
    
    '''if not possible_move:
        return (0, 0), (0, 0)
    '''

    random.shuffle(possible_move)

    piece_count = sum(
        1
        for x in range(board.shape[0])
        for y in range(board.shape[1])
        if board[x, y] != ""
    )

    if piece_count > 20 or is_king_in_check(
        board, myColor
    ):  # Opening/middlegame or King in check
        depth = 3
    elif piece_count > 10:  # Late middlegame
        depth = 4
    else:  # Endgame
        depth = 5
    bestMoveScore = float("-inf")
    bestMove = None
    for m in sort_immediate_moves(board,generate_all_moves(board,myColor),myColor):
        new_board = make_move(board,m)
        score, best_move = findPath(new_board, depth, float("-inf"), float("inf"), True,[m])
        if score > bestMoveScore:
            bestMoveScore = score
            bestMove = best_move
    if bestMove:
        return bestMove

    return (0, 0), (0, 0)


#   Example how to register the function
register_chess_bot("vz-m", chess_bot1)
