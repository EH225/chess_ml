# -*- coding: utf-8 -*-
"""
This module contains helper functions for model evaluation.
"""
import sys, os, time

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import asyncio, subprocess

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    STOCKFISH_PATH = rf"{os.path.join(PARENT_DIR, 'stockfish/stockfish-windows-x86-64-avx2.exe')}"
else:
    STOCKFISH_PATH = rf"{os.path.join(PARENT_DIR, 'stockfish/stockfish-ubuntu-x86-64-avx2')}"

import chess
import chess.engine
import numpy as np
import pandas as pd
from typing import Callable, Dict
from utils.general import create_move_to_idx_map
from core.torch_models import load_model


def play_game(model, search_algo, search_kwargs, engine, model_plays_white=True,
              stockfish_elo=1320, idx_to_move=None):
    """
    This function simulates 1 game where the input model plays vs a given engine and returns a summary of the
    result.

    :param model: An input model to play vs the engine.
    :param search_algo: A callable search algorithm used in conjunction with model to select moves to play vs
        the engine.
    :param search_kwargs: Additional kwargs to pass along with the current board FEN and a pointer to model
        when calling search_algo. This can contain e.g. n_iters which governs how the search function behaves
        or how deep it searches.
    :param engine: A stockfish engine instance to play vs the model.
    :param model_plays_white: Determines if the model should play as white.
    :param stockfish_elo: Sets the ELO strength rating of the stockfish engine. The min value is 1320.
    :param idx_to_move: A dictionary mapping integer index values [0, 1967] to uci moves strings.
    :returns: The board result, the final FEN, and a name for what the game outcome was.
    """
    board = chess.Board() # Instantiate a new game board at the starting positions
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    model_color = chess.WHITE if model_plays_white else chess.BLACK

    while not board.is_game_over(): # Run until the game ends
        # Check if the model should play, i.e. it's white's turn and the model plays as white
        # or it's black's turn and the model plays as black.
        if board.turn is model_color: # Generate a new move from the model and play it
            best_idx, _, _, _ = search_algo(board.fen(), model, **search_kwargs)
            move = chess.Move.from_uci(idx_to_move[best_idx])
        else: # Select a move from the Stockfish engine
            result = engine.play(board, chess.engine.Limit(nodes=100000))
            move = result.move
        board.push(move) # Play this move on the boad

    # board.result returns: "1-0", "0-1", "1/2-1/2" for win, loss, draw
    # we also return the final game FEN and also a text desc of the outcome
    return board.result(), board.fen(), board.outcome().termination.name


def evaluate_vs_stockfish(model, search_algo, search_kwargs, stockfish_path,
                          n_games=20, stockfish_elo=1320, idx_to_move=None,
                          verbose: bool = True):
    """
    Evaluates a model vs stockfish at a particular ELO strength for n_games and prints a summary of
    the outcome.

    :param model: An input model to play vs the engine.
    :param search_algo: A callable search algorithm used in conjunction with model to select moves to play
        vs the engine.
    :param search_kwargs: Additional kwargs to pass along with the current board FEN and a pointer to model
        when calling search_algo. This can contain e.g. n_iters which governs how the search function behaves
        or how deep it searches.
    :param stockfish_path: A str path to the stockfish engine executable.
    :param n_games: The number of games to have the model play vs the stockfish engine.
    :param stockfish_elo: Sets the ELO strength rating of the stockfish engine. The min value is 1320.
    :param idx_to_move: A dictionary mapping integer index values [0, 1967] to uci moves strings.
    :param verbose: If true, then verbose printing is done for each game played.
    :returns: Returns a tuple of length 4 (wins, losses, draws) that sums to n_games plus score where
        score = (model_wins + 0.5 * draws) / n_games.
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path) # Init the engine
    model_wins, model_losses, draws = 0, 0, 0 # Record the model's performance

    start_time = time.time()
    for i in range(n_games): # Play n_games sequentially between the model and the engine
        model_plays_white = (i % 2 == 0)  # Alternate starting colors each game
        result, fen, outcome = play_game(model, search_algo, search_kwargs, engine,
                                         model_plays_white=model_plays_white,
                                         stockfish_elo=stockfish_elo,
                                         idx_to_move=idx_to_move)

        # Interpret result from the model's perspective
        if result == "1/2-1/2":
            draws += 1
            res = "Draw"
        elif (result == "1-0" and model_plays_white) or \
             (result == "0-1" and not model_plays_white):
            model_wins += 1
            res = "Win"
        else:
            model_losses += 1
            res = "Loss"

        color = "White" if model_plays_white else "Black"
        if verbose:
            print(f"Game {i+1} (Model={color}): {res} {outcome}, {fen}")

    # Compute score as fraction of maximum possible points
    score = (model_wins + 0.5 * draws) / n_games

    if verbose:
    # Report final evaluation summary statistics
        print(f"\n--- Results vs Stockfish ELO {stockfish_elo} ({n_games} games) ---")
        print(f"Wins:   {model_wins}  ({100 * model_wins / n_games:.1f}%)")
        print(f"Draws:  {draws}  ({100 * draws / n_games:.1f}%)")
        print(f"Losses: {model_losses} ({100*model_losses / n_games:.1f}%)")
        print(f"Score: {score:.2f}/1.00")
        runtime = time.time() - start_time
        print(f"Runtime: {runtime:.2f}s, {runtime/n_games:.2f}s per game")

    engine.quit()
    return (model_wins, model_losses, draws), score


def play_game_dask(config_name: str, search_algo: Callable, search_kwargs: Dict, stockfish_path: str,
                   stockfish_elo: int = 1320, model_plays_white: bool = True, save_dir: str = None,
                   game_int: int = None) -> None:
    """
    This function loads the model associated with config_name and plays a match against stockfish at a
    given elo rating to assess the model's strength and saves a summary of the game to csv. The cached
    output includes:
        model_name: The name of the model class e.g. "cnn"
        earch_algo: The name of the search algo function used e.g. naive_search
        search_kwargs: Each key-value pair in the input search_kwargs is recorded
        stockfish_elo: The elo rating of the stockfish opponent the game was played against
        model_color: The color that model played i.e. either "white" or "black"
        result: The game result for the model, either "win", "draw", "loss"
        outcome: The game outcome, which will be CHECKMATE for win or loss or e.g. STALEMATE for certain
            draw outcomes. This field is informative for understanding the different types of draws.
        fen: The final FEN state encoding the game ended on for potential board analysis.
        runtime: The total time in seconds it took to run the game in its entirity.

    This function is meant to be run in parallel using dask to speed up evaluations. It therefore attempts
    to transfer a little data as possible between the scheduler and worker threads to minimize overhead e.g.
    the model is loaded in each task, not transfered through the task graph. Additionally, this function saves
    the results to csv after running to make sure that partial results are recorded even if the overall dask
    cluster crashes for some reason during computation or to allow for evals to be paused and resumed later.

    :param config_name: The name of a config file that will be used to load in the model from disk.
    :param search_algo: A callable search algorithm used in conjunction with model to select moves to play vs
        the engine.
    :param search_kwargs: Additional kwargs to pass along with the current board FEN and a pointer to model
        when calling search_algo. This can contain e.g. n_iters which governs how the search function behaves
        or how deep it searches.
    :param stockfish_path: A file path to the stockfish executable so that a new stockfish instance can be
        launched for this evaluation.
    :param stockfish_elo: Sets the ELO strength rating of the stockfish engine. The min value is 1320.
    :param model_plays_white: True if the model should play as white, otherwise False.
    :param save_dir: The output directory to save the results to when finished.
    :param game_int: A unique integer for this game which will be used to name the output file e.g.
        game_15.csv, this should be chosen to avoid collisions.
    :returns: None, caches the results to disk instead to avoid data loss incase a dask cluster crashes.
    """
    start_time = time.time() # Track the total time it takes to run this function
    assert save_dir is not None, "save_dir must be specified"
    assert game_int is not None, "game_int must be specified"
    # Create a dictionary mapping integer index values [0, 1967] to uci moves strings
    idx_to_move = {v:k for k,v in create_move_to_idx_map().items()}
    out = pd.Series() # Collect values to be saved to disk at the end
    model = load_model(config_name) # Load in the model associated with the config_name given
    model.eval() # Switch to eval mode for testing
    out["model_name"] = model.name
    out["search_algo"] = search_algo.__name__
    for k, v in search_kwargs.items(): # Record the search kwargs used to run the search function
        out[k] = v
    out["stockfish_elo"] = stockfish_elo

    board = chess.Board() # Instantiate a new game board at the starting positions
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path) # Init the engine
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    model_color = chess.WHITE if model_plays_white else chess.BLACK
    out["model_color"] = "white" if model_color is chess.WHITE else "black"

    while not board.is_game_over(): # Run until the game ends
        # Check if the model should play, i.e. it's white's turn and the model plays as white
        # or it's black's turn and the model plays as black.
        if board.turn is model_color: # Generate a new move from the model and play it
            best_idx, _, _, _ = search_algo(board.fen(), model, **search_kwargs)
            move = chess.Move.from_uci(idx_to_move[best_idx])
        else: # Select a move from the Stockfish engine
            result = engine.play(board, chess.engine.Limit(nodes=100000))
            move = result.move
        board.push(move) # Play this move on the boad

    # board.result returns: "1-0", "0-1", "1/2-1/2" for win, loss, draw where the first number is
    # for white, the second is for black always
    result = board.result()
    if result == "1/2-1/2":
        out["result"] = "draw"
    elif (result == "1-0" and model_plays_white) or (result == "0-1" and not model_plays_white):
        out["result"] = "win"
    else:
        out["result"] = "loss"
    out["outcome"] = board.outcome().termination.name
    out["fen"] = board.fen()
    out["runtime"] = time.time() - start_time
    out.to_frame().T.to_csv(os.path.join(save_dir, f"game_{game_int}.csv"))
    engine.quit()
