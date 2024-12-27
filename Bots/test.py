import numpy as np
from typing import Tuple, List, Dict, Optional
import time
from collections import defaultdict
from Bots.ChessBotList import register_chess_bot

class ChessBot:
    def __init__(self):
        # Piece values
        self.piece_values = {
            'p': 100,   # Pawn
            'r': 500,   # Rook
            'n': 320,   # Knight
            'b': 330,   # Bishop
            'q': 900,   # Queen
            'k': 20000  # King
        }
        
        # Position weights for piece placement
        self.position_weights = {
            'p': np.array([  # Pawn position weights
                [0, 0, 0, 0, 0, 0, 0, 0],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [10, 10, 20, 30, 30, 20, 10, 10],
                [5, 5, 10, 25, 25, 10, 5, 5],
                [0, 0, 0, 20, 20, 0, 0, 0],
                [5, -5, -10, 0, 0, -10, -5, 5],
                [5, 10, 10, -20, -20, 10, 10, 5],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            'n': np.array([  # Knight position weights
                [-50, -40, -30, -30, -30, -30, -40, -50],
                [-40, -20, 0, 0, 0, 0, -20, -40],
                [-30, 0, 10, 15, 15, 10, 0, -30],
                [-30, 5, 15, 20, 20, 15, 5, -30],
                [-30, 0, 15, 20, 20, 15, 0, -30],
                [-30, 5, 10, 15, 15, 10, 5, -30],
                [-40, -20, 0, 5, 5, 0, -20, -40],
                [-50, -40, -30, -30, -30, -30, -40, -50]
            ])
        }
        
        # Opening book (modified for single-square pawn moves)
        self.opening_moves = {
            '': [  # Starting position
                ((1, 4), (2, 4)),  # e3
                ((1, 3), (2, 3))   # d3
            ]
        }
        
        # Move history for learning
        self.move_history = defaultdict(lambda: {'count': 0, 'success': 0})

    def get_piece_moves(self, board: np.ndarray, x: int, y: int) -> List[Tuple[int, int]]:
        """Get all legal moves for a piece."""
        piece = board[x, y]
        if not piece:
            return []
            
        piece_type, color = piece[0], piece[1]
        moves = []
        
        if piece_type == 'p':  # Pawn - Modified for single square movement only
            direction = 1 if color == 'w' else -1
            new_x = x + direction
            
            # Only allow single square forward movement
            if 0 <= new_x < 8 and board[new_x, y] == '':
                moves.append((new_x, y))
            
            # Captures still allowed diagonally
            for dy in [-1, 1]:
                new_y = y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    if board[new_x, new_y] != '' and board[new_x, new_y][1] != color:
                        moves.append((new_x, new_y))
                        
        elif piece_type in ['r', 'b', 'q']:  # Rook, Bishop, Queen
            directions = []
            if piece_type in ['r', 'q']:  # Rook/Queen moves
                directions += [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if piece_type in ['b', 'q']:  # Bishop/Queen moves
                directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                while 0 <= new_x < 8 and 0 <= new_y < 8:
                    if board[new_x, new_y] == '':
                        moves.append((new_x, new_y))
                    elif board[new_x, new_y][1] != color:
                        moves.append((new_x, new_y))
                        break
                    else:
                        break
                    new_x, new_y = new_x + dx, new_y + dy
                    
        elif piece_type == 'n':  # Knight
            knight_moves = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2)
            ]
            for dx, dy in knight_moves:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    if board[new_x, new_y] == '' or board[new_x, new_y][1] != color:
                        moves.append((new_x, new_y))
                        
        elif piece_type == 'k':  # King
            king_moves = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
            for dx, dy in king_moves:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    if board[new_x, new_y] == '' or board[new_x, new_y][1] != color:
                        moves.append((new_x, new_y))
        
        return moves

    def evaluate_position(self, board: np.ndarray, color: str) -> float:
        """Evaluate board position for given color."""
        score = 0.0
        
        # Material evaluation
        for x in range(8):
            for y in range(8):
                piece = board[x, y]
                if piece:
                    piece_type, piece_color = piece[0], piece[1]
                    base_value = self.piece_values[piece_type]
                    
                    # Position bonus
                    if piece_type in self.position_weights:
                        position_bonus = self.position_weights[piece_type][x, y]
                        base_value += position_bonus
                    
                    # Add to score (positive for our pieces, negative for opponent's)
                    multiplier = 1 if piece_color == color else -1
                    score += base_value * multiplier
        
        # Center control
        center_squares = [(3,3), (3,4), (4,3), (4,4)]
        for x, y in center_squares:
            if board[x, y] != '' and board[x, y][1] == color:
                score += 30
        
        # Pawn structure
        for y in range(8):
            # Connected pawns bonus
            pawn_files = [x for x in range(8) if board[x, y].startswith('p')]
            if len(pawn_files) > 1:
                score += 20
                
        return score

    def minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, 
                maximizing: bool, color: str, start_time: float, time_budget: float) -> Tuple[float, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Minimax algorithm with alpha-beta pruning."""
        if time.time() - start_time > time_budget * 0.95:
            raise TimeoutError
            
        if depth == 0:
            return self.evaluate_position(board, color), None
            
        moves = []
        for x in range(8):
            for y in range(8):
                if board[x, y] != '' and board[x, y][1] == (color if maximizing else ('b' if color == 'w' else 'w')):
                    for new_pos in self.get_piece_moves(board, x, y):
                        moves.append(((x, y), new_pos))
        print(move)
        
        if maximizing:
            best_value = float('-inf')
            best_move = None
            for move in moves:
                # Make move
                start, end = move
                temp_board = board.copy()
                temp_board[end[0], end[1]] = temp_board[start[0], start[1]]
                temp_board[start[0], start[1]] = ''
                
                value, _ = self.minimax(temp_board, depth-1, alpha, beta, False, color, start_time, time_budget)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_value, best_move
        else:
            best_value = float('inf')
            best_move = None
            for move in moves:
                start, end = move
                temp_board = board.copy()
                temp_board[end[0], end[1]] = temp_board[start[0], start[1]]
                temp_board[start[0], start[1]] = ''
                
                value, _ = self.minimax(temp_board, depth-1, alpha, beta, True, color, start_time, time_budget)
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_value, best_move

def chess_bot(player_sequence: List[str], board: np.ndarray, time_budget: float, **kwargs) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Main chess bot function."""
    bot = ChessBot()
    color = player_sequence[1]
    
    # Check opening book
    board_key = str(board)
    if board_key in bot.opening_moves:
        return bot.opening_moves[board_key][0]
    
    # Iterative deepening
    start_time = time.time()
    best_move = None
    depth = 1
    
    try:
        while time.time() - start_time < time_budget * 0.9:
            _, current_best_move = bot.minimax(
                board, depth, float('-inf'), float('inf'),
                True, color, start_time, time_budget
            )
            if current_best_move:
                best_move = current_best_move
                # Update move history
                move_key = str(best_move)
                bot.move_history[move_key]['count'] += 1
            depth += 1
    except TimeoutError:
        pass
    
    # Fallback to simple pawn move if needed
    if not best_move:
        for x in range(board.shape[0]-1):
            for y in range(board.shape[1]):
                if board[x,y] == f"p{color}" and board[x+1,y] == '':
                    return (x,y), (x+1,y)
    
    return best_move or ((0,0), (0,0))


register_chess_bot("test",chess_bot)