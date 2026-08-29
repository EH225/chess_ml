import chess
from ai import get_best_move


class Game:

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()

    def make_player_move(self, uci_move):

        try:
            move = chess.Move.from_uci(uci_move)
        except ValueError:
            return False

        # python-chess checks whether the move is legal
        if move not in self.board.legal_moves:
            return False

        self.board.push(move)

        return True

    def make_engine_move(self):
        # Don't try to move if the game is already over
        if self.board.is_game_over():
            return None

        move = get_best_move(self.board)

        if move is None:
            return None

        self.board.push(move)

        return move.uci()

    def get_fen(self):
        return self.board.fen()

    def get_game_status(self):

        if not self.board.is_game_over():
            return {"game_over": False, "status": None}

        outcome = self.board.outcome()

        if outcome is None:
            return {"game_over": False, "status": None}

        # Checkmate

        if outcome.termination == chess.Termination.CHECKMATE:

            if outcome.winner:
                return {
                    "game_over": True,
                    "status": "Checkmate — You win!"
                }

            else:
                return {
                    "game_over": True,
                    "status": "Checkmate — AI wins!"
                }

        # Stalemate

        if outcome.termination == chess.Termination.STALEMATE:
            return {
                "game_over": True,
                "status": "Draw — Stalemate"
            }

        # Threefold repetition

        if outcome.termination == chess.Termination.THREEFOLD_REPETITION:
            return {
                "game_over": True,
                "status": "Draw — Threefold repetition"
            }

        # Fifty-move rule

        if outcome.termination == chess.Termination.FIFTY_MOVES:
            return {
                "game_over": True,
                "status": "Draw — Fifty-move rule"
            }

        # Insufficient material

        if outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            return {
                "game_over": True,
                "status": "Draw — Insufficient material"
            }

        # Fallback

        return {
            "game_over": True,
            "status": "Game over"
        }

    def get_check_square(self):
        """
        Return the square of the king currently in check.

        Returns None if neither king is in check.
        """

        if not self.board.is_check():
            return None

        king_color = self.board.turn

        king_square = self.board.king(king_color)

        if king_square is None:
            return None

        return chess.square_name(king_square)
