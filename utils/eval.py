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
from typing import Callable, Dict, List, Tuple
from utils.general import read_yaml, create_move_to_idx_map
from core.torch_models import load_model
from IPython.display import display
from utils.chess_env import ChessEnv, save_recording


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
    board = chess.Board()  # Instantiate a new game board at the starting positions
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    model_color = chess.WHITE if model_plays_white else chess.BLACK

    while not board.is_game_over():  # Run until the game ends
        # Check if the model should play, i.e. it's white's turn and the model plays as white
        # or it's black's turn and the model plays as black.
        if board.turn is model_color:  # Generate a new move from the model and play it
            best_idx, _, _, _ = search_algo(board.fen(), model, **search_kwargs)
            move = chess.Move.from_uci(idx_to_move[best_idx])
        else:  # Select a move from the Stockfish engine
            result = engine.play(board, chess.engine.Limit(nodes=100000))
            move = result.move
        board.push(move)  # Play this move on the board

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
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)  # Init the engine
    model_wins, model_losses, draws = 0, 0, 0  # Record the model's performance

    start_time = time.time()
    for i in range(n_games):  # Play n_games sequentially between the model and the engine
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
            print(f"Game {i + 1} (Model={color}): {res} {outcome}, {fen}")

    # Compute score as fraction of maximum possible points
    score = (model_wins + 0.5 * draws) / n_games

    if verbose:
        # Report final evaluation summary statistics
        print(f"\n--- Results vs Stockfish ELO {stockfish_elo} ({n_games} games) ---")
        print(f"Wins:   {model_wins}  ({100 * model_wins / n_games:.1f}%)")
        print(f"Draws:  {draws}  ({100 * draws / n_games:.1f}%)")
        print(f"Losses: {model_losses} ({100 * model_losses / n_games:.1f}%)")
        print(f"Score: {score:.2f}/1.00")
        runtime = time.time() - start_time
        print(f"Runtime: {runtime:.2f}s, {runtime / n_games:.2f}s per game")

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
    start_time = time.time()  # Track the total time it takes to run this function
    assert save_dir is not None, "save_dir must be specified"
    assert game_int is not None, "game_int must be specified"
    # Create a dictionary mapping integer index values [0, 1967] to uci moves strings
    idx_to_move = {v: k for k, v in create_move_to_idx_map().items()}
    out = pd.Series()  # Collect values to be saved to disk at the end
    model = load_model(config_name)  # Load in the model associated with the config_name given
    model.eval()  # Switch to eval mode for testing
    out["model_name"] = model.name
    out["search_algo"] = search_algo.__name__
    for k, v in search_kwargs.items():  # Record the search kwargs used to run the search function
        out[k] = v
    out["stockfish_elo"] = stockfish_elo

    board = chess.Board()  # Instantiate a new game board at the starting positions
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)  # Init the engine
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    model_color = chess.WHITE if model_plays_white else chess.BLACK
    out["model_color"] = "white" if model_color is chess.WHITE else "black"

    while not board.is_game_over():  # Run until the game ends
        # Check if the model should play, i.e. it's white's turn and the model plays as white
        # or it's black's turn and the model plays as black.
        if board.turn is model_color:  # Generate a new move from the model and play it
            best_idx, _, _, _ = search_algo(board.fen(), model, **search_kwargs)
            move = chess.Move.from_uci(idx_to_move[best_idx])
        else:  # Select a move from the Stockfish engine
            result = engine.play(board, chess.engine.Limit(nodes=100000))
            move = result.move
        board.push(move)  # Play this move on the boad

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


def head_to_head_match(white_cfg: Tuple, black_cfg: str, starting_fen: str = None,
                       record_path: str = None) -> List[chess.Move]:
    """
    Plays a head-to-head match between 2 chess agents. Return the chess.Board object that results which
    contains the move stack, board outcome and FEN.

    e.g. Example usage:
        white_cfg = ("cnn", sa.null_search, {"temp": 0})
        black_cfg = ("cnn", sa.naive_search, {"batch_size": 64})
        board = head_to_head_match(white_cfg, black_cfg, None,
                                   os.path.join(os.getcwd(), "recordings/game_1.mp4"))

    :param white_cfg: A length-3 tuple of (config_name, search_algo, search_kwargs) for white.
    :param black_cfg: A length-3 tuple of (config_name, search_algo, search_kwargs) for black.
    :param starting_fen: A FEN board encoding of the starting state of the game. If not provided, the game
        begins at the standard chess start.
    :param step_limit: Sets and upper bound for the max number of moves allowed.
    :param record_path: If provided, then a recording of the match will be saved to the file path specified,
        which must end in .mp4.
    :return: Returns the chess.Board object after the game has finished.
    """
    if record_path is not None:
        assert record_path.endswith(".mp4"), "record_path must end with the .mp4 file extension"

    idx_to_move = {v: k for k, v in create_move_to_idx_map().items()}
    # Load in the models from disk using the config names
    model_w = load_model(white_cfg[0])
    model_b = load_model(black_cfg[0])

    board = chess.Board() if starting_fen is None else chess.Board(starting_fen)
    while not board.is_game_over():  # Run until the game ends
        if board.turn:  # White to move next
            best_idx, _, _, _ = white_cfg[1](board.fen(), model_w, **white_cfg[2])
            move = chess.Move.from_uci(idx_to_move[best_idx])
        else:  # Black to move next
            best_idx, _, _, _ = black_cfg[1](board.fen(), model_b, **black_cfg[2])
            move = chess.Move.from_uci(idx_to_move[best_idx])
        board.push(move)  # Play this move on the boad

    if record_path is not None:  # Save a recording of the match up if a record path is specified
        save_recording(board.move_stack, output_path=record_path)

    # Report the game result
    result = board.result()
    result_name = board.outcome().termination.name
    result = board.result()
    if result == "1/2-1/2":
        print(f"Draw - {result_name}")
    elif result == "1-0":
        print(f"White wins - {result_name}")
    else:
        print(f"Black wins - {result_name}")

    return board


def interactive_match(model_cfg: Tuple, player_color: str = "white", starting_fen: str = None,
                      verbose: bool = False) -> None:
    """
    This function allows a user to play an interactive match vs one of the chess agents saved to disk in
    the results folder. Players will be prompted to input their next move and the chess agent will play
    the other side of the board until the game is over.

    Moves are expected to be entered in standard algebraic notation (SAN) e.g. e4, Qh7, g1g5 etc.
    If "quit" is input, the game session ends. If "undo" is entered, the last 2 moves are reversed i.e. the
    agent's last move and the player's prior move. Games run until an end game condition is met
    e.g. checkmate, stalemate, insufficient material, repetition etc.

    :param model_cfg: A length-3 tuple of (config_name, search_algo, search_kwargs) for the agent.
    :param player_color: The color of the user playing i.e. "white" or "black".
    :param initial_state: A FEN board encoding of the starting state of the game. If not provided, the game
        begins at the standard chess start.
    :param verbose: If set to True, then a verbose print out of the agent decision making is provided.
    :return: None.
    """
    move_to_idx = create_move_to_idx_map()
    idx_to_move = {v: k for k, v in move_to_idx.items()}
    player_color = player_color.lower()
    assert player_color.lower() in ["white", "black"], "Player color must be either white or black"
    player_color = chess.WHITE if player_color == "white" else chess.BLACK

    model = load_model(model_cfg[0])
    board = chess.Board() if starting_fen is None else chess.Board(starting_fen)
    check = board.king(board.turn) if board.is_check() else None
    display(chess.svg.board(board, orientation=player_color, check=check))

    while not board.is_game_over():  # Play until the game is finished
        print(board.fen())  # Show the FEN of the game on each turn before the next move is made
        if board.turn == player_color:
            move_valid = False
            while not move_valid:
                try:
                    player_move_san = input("Input move in SAN: ")
                    if player_move_san == "quit":  # Exit immediately if the user enters "quit"
                        return None
                    elif player_move_san == "undo" and len(board.move_stack) >= 2:
                        # Undo the last 2 moves to get back to your last move
                        board.pop()  # Undo opponent's last move
                        board.pop()  # Undo player's last move before that
                        if len(board.move_stack) > 0:
                            check = board.king(board.turn) if board.is_check() else None
                            display(chess.svg.board(board, orientation=player_color, lastmove=board.peek(),
                                                    check=check))
                        else:
                            display(chess.svg.board(board, orientation=player_color))
                    else:  # All other inputs are interpreted as moves
                        board.push_san(player_move_san)
                        move_valid = True
                except:
                    print("Move invalid, please try again")

        else:  # Otherwise it's the turn of the AI chess bot agent to play
            best_idx, state_value, action_values, info = model_cfg[1](board.fen(), model, **model_cfg[2])
            best_move_uci = idx_to_move[best_idx]

            if verbose:  # Verbose printing, show what the model was thinking on this play
                print(f"\nState Value: {state_value:.2f}")
                print(f"Best Action: {best_idx} {str(best_move_uci)}")
                print(f"Nodes Evaluated: {info[0]}, Max Depth: {info[1]}, Terminal Nodes: {info[2]}")
                move_vals = [(move, val) for move, val in zip(list(board.legal_moves), action_values)]
                # List each possible move and its approx value in descending order
                for move, value in sorted(move_vals, key=lambda x: -x[1]):
                    print(f"   {move} {value:.3f}")

            board.push(chess.Move.from_uci(best_move_uci))  # Make the move of the chess agent

        check = board.king(board.turn) if board.is_check() else None
        display(chess.svg.board(board, orientation=player_color, lastmove=board.peek(), check=check))

    print(board.fen())  # Report the FEN of the game on the last move as well
    outcome = board.outcome()
    msg = ("- black wins!" if board.turn else "- white wins!") if board.is_checkmate() else ""
    print(f"Outcome: {outcome.termination.name} {msg}")


if __name__ == "__main__":
    import core.search_algos as sa

    # Play vs the CNN agent
    # model_cfg = ("cnn", sa.null_search, {"batch_size": 64})
    model_cfg = ("cnn", sa.monte_carlo_tree_search, {"batch_size": 64, "n_iters": 3000})
    interactive_match(model_cfg, "white", None, True)
