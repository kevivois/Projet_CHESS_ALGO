from PyQt6.QtWidgets import QApplication

from ChessArena import ChessArena, ChessApp


if __name__ == "__main__":

    import sys

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    app = ChessApp()
    app.start()
