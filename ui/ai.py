import random
import time

def get_best_move(board):
    time.sleep(3)
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    return random.choice(legal_moves)
