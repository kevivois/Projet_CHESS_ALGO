from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6 import uic

from Bots.ChessBotList import *
from ChessRules import *
from ParallelPlayer import *
from Bots import *

import numpy as np
import time
import random
from Bots.vz_m import chess_bot1  # Add this line to import vz_m


#   Wrap up for QApplication
class ChessApp(QtWidgets.QApplication):
    def __init__(self):
        super().__init__([])

    def start(self):
        arena = ChessArena()
        arena.show()
        arena.start()

        self.exec()


#   Main window to handle the chess board
CHESS_PIECES = ["k", "q", "n", "b", "r", "p"]
CHESS_COLOR = {
    "w": [QtGui.QColor(255, 255, 255), QtGui.QColor(0, 0, 0)],
    "b": [QtGui.QColor(0, 0, 0), QtGui.QColor(255, 255, 255)],
    "r": [QtGui.QColor(200, 0, 0), QtGui.QColor(50, 255, 255)],
    "y": [QtGui.QColor(200, 200, 0), QtGui.QColor(50, 50, 255)],
}
COLOR_NAMES = {"w": "White", "b": "Black", "r": "Red", "y": "Yellow"}
CHESS_PIECES_NAMES = {
    "k": "King",
    "q": "Queen",
    "n": "Knight",
    "b": "Bishop",
    "r": "Rook",
    "p": "Pawn",
}


class ChessArena(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        uic.loadUi("Data/UI.ui", self)

        #   Render for chess board
        self.chess_scene = QtWidgets.QGraphicsScene()
        self.chess_scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("white")))
        # self.chess_scene.setBackgroundBrush(QBrush(Qt6.lightGray))
        self.chessboardView.setScene(self.chess_scene)

        self.loadBoardButton.clicked.connect(self.select_and_load_board)

        self.launchGameButton.clicked.connect(self.launch_game)

        self.load_assets()

        self.current_player = None

    def add_system_message(self, message):
        msg_widget = QtWidgets.QLabel(message)
        msg_widget.setWordWrap(True)
        print("[SYS]", message)
        self.systemMessagesLayout.addWidget(msg_widget)
        self.systemMessagesBox = QtWidgets.QScrollArea()
        self.systemMessagesBox.verticalScrollBar().setSliderPosition(
            self.systemMessagesBox.verticalScrollBar().maximum()
        )

    #   Called to start the bot simulation
    def launch_game(self):
        self.add_system_message("# Starting new Game #")
        #   Prepare AIs
        self.players_AI = {}
        for cid, color in enumerate(self.players_AI_choice):
            self.players_AI[color] = CHESS_BOT_LIST[
                self.players_AI_choice[color].currentText()
            ]
            self.add_system_message(
                "AI #" + str(cid) + " = " + str(self.players_AI[color].__name__)
            )

        self.nbr_turn_to_play = int(self.maxTurnBudget.text())
        self.max_time_budget = float(self.timeBudgetInput.text())

        self.play_next_turn()

    def play_next_turn(self):
        if self.current_player is not None:
            print("Cannot launch new turn while already processing")
            return

        if self.nbr_turn_to_play == 0:
            print("No more play to do")
            self.end_game(None)
            return

        next_player_color = self.player_order[0:3]

        #   Prepare board view
        rotated_view_board = np.rot90(self.board, int(next_player_color[2]))
        self.current_player = ParallelTurn(
            self.players_AI[next_player_color[1]],
            self.player_order,
            rotated_view_board,
            self.max_time_budget,
        )
        self.current_player.setTerminationEnabled(True)
        self.current_player.start()

        #   Timer to call
        QtCore.QTimer.singleShot(int(self.max_time_budget * 1000 * 1.05), self.end_turn)

    def end_turn(self):
        all_other_defeated = False

        if self.current_player.isRunning():
            self.current_player.terminate()
            self.add_system_message(
                COLOR_NAMES[self.current_player.color] + " did not end his turn"
            )
        else:

            player_color = self.current_player.color

            next_play = self.current_player.next_move

            if not move_is_valid(
                self.player_order, next_play, self.current_player.board
            ):
                self.add_system_message(
                    COLOR_NAMES[player_color]
                    + " invalid move from "
                    + str(next_play[0])
                    + " to "
                    + str(next_play[1])
                )
                return

            self.add_system_message(
                COLOR_NAMES[player_color]
                + " moved "
                + CHESS_PIECES_NAMES[
                    self.current_player.board[next_play[0][0], next_play[0][1]][0]
                ]
                + " from "
                + str(next_play[0])
                + " to "
                + str(next_play[1])
            )

            if self.current_player.board[next_play[1][0], next_play[1][1]] != "":
                self.add_system_message(
                    COLOR_NAMES[player_color]
                    + " captured "
                    + COLOR_NAMES[
                        self.current_player.board[next_play[1][0], next_play[1][1]][1]
                    ]
                    + " "
                    + CHESS_PIECES_NAMES[
                        self.current_player.board[next_play[1][0], next_play[1][1]][0]
                    ]
                )

            #   apply move
            self.current_player.board[next_play[1][0], next_play[1][1]] = (
                self.current_player.board[next_play[0][0], next_play[0][1]]
            )
            self.current_player.board[next_play[0][0], next_play[0][1]] = ""

            #   check for promotion
            if (
                self.current_player.board[next_play[1][0], next_play[1][1]][0] == "p"
                and next_play[1][0] == self.current_player.board.shape[0] - 1
            ):
                self.current_player.board[next_play[1][0], next_play[1][1]] = (
                    "q" + self.current_player.board[next_play[1][0], next_play[1][1]][1]
                )

            all_other_defeated = True
            for row in self.board:
                for elem in row:
                    if len(elem) > 0 and elem[0] == "k":
                        if int(
                            self.player_order[self.player_order.find(elem[1]) - 1]
                        ) != int(self.current_player.team):
                            all_other_defeated = False

        #   Update board state
        self.current_player = None

        self.setup_board()

        #   Current player goes at the end of the play queue
        self.player_order = self.player_order[3:] + self.player_order[0:3]
        self.nbr_turn_to_play -= 1

        if all_other_defeated:
            self.end_game(player_color)
        else:
            self.play_next_turn()

    def end_game(self, winner):
        if winner is None:
            self.add_system_message("# Match ended in a draw")
            optimal_time_budget = self.find_optimal_time_budget()
            self.add_system_message(
                f"The optimal time budget is: {optimal_time_budget}"
            )
        else:
            self.add_system_message("# " + str(COLOR_NAMES[winner]) + " won the match")

    def evaluate_bot_performance(self, time_budget):
        """
        Function to evaluate the performance of the bot with a given time budget.

        :param time_budget: The time budget for making a move in seconds.
        :return: The performance score of the bot.
        """
        # Initialize the board and other necessary variables
        board = np.array(
            [
                ["r", "n", "b", "q", "k", "b", "n", "r"],
                ["p", "p", "p", "p", "p", "p", "p", "p"],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
                ["P", "P", "P", "P", "P", "P", "P", "P"],
                ["R", "N", "B", "Q", "K", "B", "N", "R"],
            ]
        )
        player_sequence = ["w", "b"]
        start_time = time.time()

        # Simulate a game with the given time budget
        while time.time() - start_time < 10:  # Run the simulation for 10 seconds
            move = chess_bot1(player_sequence, board, time_budget)
            # Apply the move to the board (this is a simplified example)
            start, end = move
            board[end[0], end[1]] = board[start[0], start[1]]
            board[start[0], start[1]] = ""
            player_sequence.reverse()

        # Evaluate the performance (this is a placeholder, you can define your own criteria)
        performance_score = np.random.uniform(
            0, 1
        )  # Replace with actual performance evaluation

        return performance_score

    def find_optimal_time_budget(self):
        """
        Function to find the optimal time budget for the bot.

        :return: The optimal time budget.
        """
        time_budgets = np.linspace(0.125, 0.3, 10)
        best_time_budget = None
        best_performance = -float("inf")

        for time_budget in time_budgets:
            performance = self.evaluate_bot_performance(time_budget)
            print(f"Time budget: {time_budget}, Performance: {performance}")
            if performance > best_performance:
                best_performance = performance
                best_time_budget = time_budget

        return best_time_budget

    def select_and_load_board(self):
        path = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select board",
            "C:\\Users\\Louis\\Desktop\\ISChess\\Data\\maps",
            "Board File (*.brd)",
        )

        if path is None:
            return
        path = path[0]

        self.board = self.load_board(path)

        if self.board is None:
            return

        self.setup_board()
        self.setup_players()

    def load_board(self, path):
        try:
            with open(path, "r") as f:
                data = f.read()

                lines = data.split("\n")
                self.player_order = lines[0]
                elems = [l.replace("--", "").split(",") for l in lines[1:]]

                #   Protection against final empty lines
                while len(elems) > 0 and len(elems[-1]) == 0:
                    del elems[-1]

                #   check lines length equals
                for l in elems:
                    if len(l) != len(elems[0]):
                        return None

                return np.array(elems, dtype="O")
        except Exception as e:
            print(e)
            return None

        return None

    def load_assets(self):
        self.white_square = QtGui.QPixmap("Data/assets/light_square.png")
        self.black_square = QtGui.QPixmap("Data/assets/dark_square.png")

        self.pieces_imgs = {}

        for p in CHESS_PIECES:
            image = QtGui.QImage("Data/assets/" + p + ".png")
            self.pieces_imgs[p] = image

    def setup_players(self):
        for i in reversed(range(self.playersList.count())):
            if self.playersList.itemAt(i).widget() is not None:
                self.playersList.itemAt(i).widget().setParent(None)

        self.players_AI_choice = {}
        for color in self.colored_piece_pixmaps:
            l = QtWidgets.QLabel("Color:" + COLOR_NAMES[color])
            self.playersList.addWidget(l)

            choice = QtWidgets.QComboBox()

            for name in CHESS_BOT_LIST:
                choice.addItem(name, CHESS_BOT_LIST[name])
            choice.setCurrentIndex(2)

            self.players_AI_choice[color] = choice
            self.playersList.addWidget(choice)

        self.playersList.addItem(
            QtWidgets.QSpacerItem(
                20,
                40,
                QtWidgets.QSizePolicy.Policy.Minimum,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        )

    def setup_board(self):

        for i in reversed(self.chess_scene.items()):
            self.chess_scene.removeItem((i))

        #   Maintain the pointer towards the items on the board
        self.piece_items = np.array(
            [[None] * self.board.shape[1]] * self.board.shape[0], dtype=object
        )
        #   Maintain list of colored pixmap
        self.colored_piece_pixmaps = {}

        for y in range(self.board.shape[1]):
            for x in range(self.board.shape[0]):

                square_color = (
                    self.white_square if (x + y) % 2 == 0 else self.black_square
                )
                square_item = self.chess_scene.addPixmap(square_color)
                square_item.setPos(
                    QtCore.QPointF(
                        square_color.size().width() * y,
                        square_color.size().height() * x,
                    )
                )

                if self.board[x, y] != "" and self.board[x, y] != "XX":
                    player_piece = self.board[x, y][0]
                    player_color = self.board[x, y][1]

                    if player_color not in self.colored_piece_pixmaps:
                        self.colored_piece_pixmaps[player_color] = {}

                    if player_piece not in self.colored_piece_pixmaps[player_color]:
                        piece_img = self.pieces_imgs[player_piece]
                        copy = piece_img.copy()

                        def mix(Q1, Q2, f, a):
                            return QtGui.QColor(
                                int(Q1.red() * f + Q2.red() * (1 - f)),
                                int(Q1.green() * f + Q2.green() * (1 - f)),
                                int(Q1.blue() * f + Q2.blue() * (1 - f)),
                                a,
                            )

                        for px in range(copy.size().width()):
                            for py in range(copy.size().height()):
                                copy.setPixelColor(
                                    px,
                                    py,
                                    mix(
                                        CHESS_COLOR[player_color][0],
                                        CHESS_COLOR[player_color][1],
                                        copy.pixelColor(px, py).red() / 255.0,
                                        copy.pixelColor(px, py).alpha(),
                                    ),
                                )

                        self.colored_piece_pixmaps[player_color][
                            player_piece
                        ] = QtGui.QPixmap().fromImage(copy)

                    self.piece_items[x, y] = self.chess_scene.addPixmap(
                        self.colored_piece_pixmaps[player_color][player_piece]
                    )
                    self.piece_items[x, y].setPos(
                        QtCore.QPointF(
                            square_color.size().width() * y,
                            square_color.size().height() * x,
                        )
                    )

        self.chessboardView.fitInView(self.chess_scene.sceneRect())

    def start(self):
        self.board = self.load_board("Data/maps/default.brd")
        self.board = self.load_board("Data/maps/default.brd")
        self.setup_board()
        self.setup_players()
        self.chess_scene.update()
