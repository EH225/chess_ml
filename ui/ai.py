import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import random
import core.search_algos as sa
from core.torch_models import load_model
from utils.general import read_yaml, create_move_to_idx_map
import chess

### Define the model to play against
model_cfg = ("cnn", sa.monte_carlo_tree_search, {"batch_size": 64, "n_iters": 1500})
# model_cfg = ("cnn", sa.naive_search, {"batch_size": 64})
model = load_model(model_cfg[0])
idx_to_move = {v: k for k, v in create_move_to_idx_map().items()}


def get_best_move(board):
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    best_idx, state_value, action_values, info = model_cfg[1](board.fen(), model, **model_cfg[2])
    return chess.Move.from_uci(idx_to_move[best_idx])
