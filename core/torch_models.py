"""
This module defines the deep learning models used to analyze the board and produce policy and value outputs.
    1. A multi-layer perceptron (MLP) neural network
    2. A CNN-based neural network
    3. A transformer-based model
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch, chess
import torch.nn as nn
import torch.nn.functional as F
from utils.chess_env import relative_material_diff
from typing import Tuple, List, Dict

torch.backends.mkldnn.enabled = True  # Usually enabled, but set to be sure


##################################################
### Pre-Training Material Heuristic Definition ###
##################################################
# TODO: Section marker

class MaterialHeuristic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = "cpu"  # Always run on the CPU
        self.pos_embeddings = nn.Parameter(torch.zeros(1))  # Needs a parameter for the optimizer init

    def forward(self, state_batch: List[str]) -> torch.Tensor:
        """
        Forward pass through the model which generates a value estimate of the current board position for
        each state observation in the input state_batch i.e. an estimate of the expected reward from the
        current state position.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, ) with the value estimates for each stating position.
        """
        if len(state_batch) > 0:
            return torch.Tensor([relative_material_diff(state) for state in state_batch])
        else:  # If an empty batch is passed, return an empty torch.Tensor
            return torch.zeros(0)


####################################
### MLP Value-Network Definition ###
####################################
# TODO: Section marker

class ResBlockMLP(nn.Module):
    """
    Residual block sub-unit of MLP network model class.
    """

    def __init__(self, size: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(size)
        self.fc1 = nn.Linear(size, size)
        self.norm2 = nn.LayerNorm(size)
        self.fc2 = nn.Linear(size, size)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.activation(self.fc1(self.norm1(x))))
        h = self.fc2(self.norm2(h))
        return x + h


class MLP(nn.Module):
    """
    Implementation of a multi-layer perceptron (MLP) / fully-connected neural network (FCNN) model for chess
    board value and policy estimation.
    """

    def __init__(self, config: Dict = None, *args, **kwargs):
        """
        Initializes a fully-connected neural network with residual connections to output value and policy
        estimates. This network architecture follows that of the TD-λ evaluator.

        The input to this network will be a batch of FEN string state representation of the current game
        state and output a torch.Tensor of the same length detailing the model's policy logits and value
        estimates for each input state.
        """
        super().__init__()
        config = {"model": {}} if config is None else config
        self.n_blocks = int(config["model"].get("n_blocks", 12))
        self.hidden_size = int(config["model"].get("hidden_size", 256))
        self.dropout = float(config["model"].get("dropout", 0.2))

        input_shape = 8 * 8 * 6 + 4  # 8 rows, 8 cols, 6 piece types, -1, 0, 1 values denoting a piece as
        # friendly or foe or if the cell is empty for a total size of 388 input features +4 castling rights

        # 1). Begin with a shared backbone for both the policy head and value head
        self.input_layer = nn.Sequential(
            nn.Linear(input_shape, self.hidden_size),
            nn.LeakyReLU(),
        )

        # Uniform-width MLP residual block tower — depth is where the learning happens
        self.res_tower = nn.Sequential(
            *[ResBlockMLP(self.hidden_size, self.dropout) for _ in range(self.n_blocks)]
        )  # Output: (batch_size, hidden_size)

        # 2). Add a policy head sub-component that will operate off the output features from self.res_tower
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1968)  # 1968 possible UCI moves
        )  # Output: (batch_size, 1968) - returns raw logits, no softmax applied here

        # 3). Add a value head sub-component that will operate off the input features from self.res_tower
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 1),
            # Use a final Tanh activation function at the end to produce value estimates [-1, +1]
            nn.Tanh()
        )

    def get_device(self) -> str:
        """
        Returns the device the model is currently on by returning the device of the parameters.
        """
        return next(self.parameters()).device.type

    @staticmethod
    def state_to_model_input(state_batch: List[str]) -> torch.Tensor:
        """
        Converts an input batch of board states (encoded using FEN as a string) into the expected state
        representations for this model (torch.Tensor).

        This MLP model operates on board representations that are [8 x 8 x 6] which record the spatial
        location of each piece and type along each dimension while the value denotes (+1) for friend,
        (-1) for foe, or (0) for an empty square. In the last dimension, index 0 = pawn, index 1 = knight,
        index 2 = bishop, index 3 = rook, index 4 = queen, index 5 = king.

        Locationally, the friendly pieces are always shown at the bottom of the board and foe pieces always
        are shown at the top of the board. Therefore, there is some board flipping depending on the color
        whose move it is next so that the model learns the same chess playing from the same perspective.

        The [8 x 8 x 6] board tensors are flattened and then a few additional info nodes are appended to
        indicate if each side can still king-side castle or queen-side castle.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, (8 * 8 * 7 + 4)) = (batch_size, 452)
        """
        sym_to_int = {s: i for i, s in enumerate("pnbrqk")}  # Mapping from symbol e.g. "b" to index e.g. 2
        output = torch.zeros((len(state_batch), 8, 8, 7))  # 8 rows, 8 cols, 6 piece types, encode -1, 0, or 1
        # plus 1 more for en passant moves allowed if any on the [8 x 8] grid. Encode as +1 is it can be done
        extra_info = torch.zeros((len(state_batch), 4))  # 4 castling rights
        # A). Add spatial information about the pieces on the board
        for i, state in enumerate(state_batch):
            board = chess.Board(state)  # Use the FEN string encoding to create the board
            for cell, piece in board.piece_map().items():  # Iter over the piece dictionary
                r, c = divmod(cell, 8)  # Get the row and column of this piece
                # By default, White's pieces will be in cells 0, 1, ... etc. and black's will be in cells
                # ... 63, 64 etc. We want to have friendly pieces always in row 7 so if its black's turn to
                # move, we will flip the columns only so that cell 64 (h8) maps to the bottom left corner. On
                # white's move, we will flip the rows only so that cell 0 (a1) maps to the bottom left corner
                if board.turn:  # White's turn to move
                    r = 7 - r  # Reverse the row order
                else:  # Black's turn to move
                    c = 7 - c  # Reverse the col order
                if board.turn:  # If it is white's move, then upper-case (white) are the player's pieces
                    val = -1 if piece.symbol().islower() else 1
                else:  # If it is black's move, then upper-case (white) pieces are the opponent's
                    val = 1 if piece.symbol().islower() else -1
                output[i, r, c, sym_to_int[piece.symbol().lower()]] = val

            # B). Add en passant rights as well as 7th plane if there are any
            if board.ep_square:  # Will be None if no en passant is possible, there is at most 1 en passant
                # on the board at any given time, it must immediately follow a pawn 2-cell jump
                ep_r, ep_c = divmod(board.ep_square, 8)
                # Apply the same perspective flip as for pieces
                if board.turn:  # White's turn
                    ep_r = 7 - ep_r
                else:  # Black's turn
                    ep_c = 7 - ep_c
                output[i, ep_r, ep_c, 6] = 1

            # C). Add castling rights on the king and queen side for friendly and foe
            if board.turn is chess.WHITE:  # If it is white's turn, then white is friendly and black is foe
                friendly_color, foe_color = chess.WHITE, chess.BLACK
            else:  # Otherwise if it is black's turn, then black is friendly and white is foe
                friendly_color, foe_color = chess.BLACK, chess.WHITE
            extra_info[i, 0] = 1 if board.has_kingside_castling_rights(friendly_color) else 0
            extra_info[i, 1] = 1 if board.has_queenside_castling_rights(friendly_color) else 0
            extra_info[i, 2] = 1 if board.has_kingside_castling_rights(foe_color) else 0
            extra_info[i, 3] = 1 if board.has_queenside_castling_rights(foe_color) else 0

            # D). 50 move rule counter - Encodes the number of half-moves since last capture or pawn move
            # when this counter reaches 100, then 50 whole moves have been made and the game ends in a draw
            # extra_info[i, 4] = min(board.halfmove_clock, 100) / 100.0  # Scale to be [0, 1]
            # This data is not found in our training dataset so we will omit it from the state tensor

        output = output.flatten(start_dim=1)  # (batch_size, 8, 8, 7) -> (batch_size, 448)
        # Add extra info about castling rights
        output = torch.concatenate([output, extra_info], dim=1)  # (B, 448) +  (B, 4) =  (B, 452)
        # Move the model_input to the required device so it can be run through the network before returning
        return output

    def forward(self, state_batch: List[str]) -> Tuple[torch.Tensor]:
        """
        Forward pass through the model which generates a policy logit vector over all moves and a value
        estimate of the current board position for each input state observation in the input state_batch
        i.e. an estimate of what the best move would be to play next among all possible moves and also
        the expected reward from the perspective of the player to go next.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A tuple of torch.Tensors:
            policy_logits of size (batch_size, 1968)
            value_estimates of size (batch_size, )
        """
        device = next(self.parameters()).device  # Get the appropriate device
        if len(state_batch) == 0:  # For an empty input state_batch of length 0, return outputs of 0s
            return torch.zeros(0, 1968).to(device), torch.zeros(0).to(device)
        else:
            # Convert the input board into the expected state representation and pass it through the network
            if isinstance(state_batch, list):
                x = self.state_to_model_input(state_batch).to(device)
            elif isinstance(state_batch, torch.Tensor):  # Skip the conversion if already done
                x = state_batch.to(device)
            else:
                raise TypeError("state_batch expected to be either a list or torch.Tensor")

            # Get the shared deep latent features (batch_size, hidden_size)
            features = self.res_tower(self.input_layer(x))
            # Pass these shared features to the policy and value sub-components to compute outputs
            policy_logits = self.policy_head(features)  # (batch_size, 1968)
            value_estimates = self.value_head(features).squeeze(1)  # (batch_size, )
            return policy_logits, value_estimates


####################################
### CNN Value-Network Definition ###
####################################
# TODO: Section marker

class ResBlockCNN(nn.Module):
    """
    Residual block sub-unit of CNN network model class.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.activation(x + h)


class CNN(nn.Module):
    """
    Implementation of a convolutional neural network (CNN) model chess board value and policy estimation.
    """

    def __init__(self, config: Dict = None, *args, **kwargs):
        """
        Initializes the required value and policy network model as a convolutional neural network (CNN). This
        network architecture follows a blend of various other resnet-based CNNs including the one from
        AlphaZero.

        The input to this network will be a batch of FEN string state representation of the current game
        state and the output will be 2 torch.Tensors:
            A). A (batch_size, 1) value estimate for each board for the play to go next [-1, +1]
            B). A (batch_size, 1968) policy vector of logits over all possible UCI moves
        """
        super().__init__()
        config = {"model": {}} if config is None else config
        self.n_blocks = int(config["model"].get("n_blocks", 12))
        self.channels = int(config["model"].get("channels", 128))

        # 1). Begin with a shared backbone for both the policy and valid heads
        self.input_layer = nn.Sequential(
            # Conv2d Block 1
            nn.Conv2d(in_channels=17, out_channels=self.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(),  # (batch_size, channels, 8, 8)
        )

        # Uniform-width residual tower — depth is where the learning happens
        self.res_tower = nn.Sequential(
            *[ResBlockCNN(self.channels) for _ in range(self.n_blocks)]
        )  # (batch_size, channels, 8, 8)

        # 2). Add a policy head sub-component that will operate off the output features from self.res_tower
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=2, kernel_size=1),  # 1x1 conv compress channels
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),  # (batch_size, 2 * 8 * 8) = (batch_size, 128)
            nn.Linear(2 * 8 * 8, 1968)  # 1968 possible UCI moves
        )  # Output: (batch_size, 1968) - returns raw logits, no softmax applied here

        # 3). Add a value head sub-component that will operate off the output features from self.res_tower
        self.value_head = nn.Sequential(
            nn.Conv2d(self.channels, 1, kernel_size=1),  # Compress to (batch_size, 1, 8, 8)
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Flatten(),  # (batch_size, 1, 8, 8) -> (batch_size, 64)
            nn.Linear(8 * 8, 256),  # (batch_size, 64) -> (batch_size, 256)
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),  # Dropout regularization to reduce potential overfitting
            nn.Linear(256, 1),  # (batch_size, 256) -> (batch_size, 1)
            # Use a final Tanh activation function at the end to produce value estimates [-1, +1]
            nn.Tanh()
        )  # Output: (batch_size, 1)

    def get_device(self) -> str:
        """
        Returns the device the model is currently on by returning the device of the parameters.
        """
        return next(self.parameters()).device.type

    @staticmethod
    def state_to_model_input(state_batch: List[str]) -> torch.Tensor:
        """
        Converts an input batch of board states (encoded using FEN as a string) into the expected state
        representations for this model (torch.Tensor).

        This CNN model operates on board representations that are [(12 + 4 + 1), 8, 8]
        or [17, 8, 8] in total. The meaning of each plate (channel) input is as follows:
            1. Location of friendly pawns encoded with 1s
            2. Location of friendly knights encoded with 1s
            3. Location of friendly bishops encoded with 1s
            4. Location of friendly rooks encoded with 1s
            5. Location of friendly queens encoded with 1s
            6. Location of friendly king encoded with a 1
            7. - 12. Location of foe pawns, knights, bishops, rooks, queens, king
            13. If player's side can king-side castle (all 1s if yes, otherwise 0s)
            14. If player's side can queen-side castle (all 1s if yes, otherwise 0s)
            15. & 16. Same for opponent's king-side and queen-side castling rights
            17. En-Passant square - Encodes a 1 in a cell where an en passant capture move can be made if any

        Locationally, the friendly pieces are always shown at the bottom of the board and foe pieces always
        are shown at the top of the board. Therefore, there is some board flipping depending on the color of
        whose move it is next so that the model learns the same chess playing from the same perspective.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, 17, 8, 8)
        """
        sym_to_int = {s: i for i, s in enumerate("pnbrqk")}  # Mapping from symbol e.g. "b" to index e.g. 2
        output = torch.zeros((len(state_batch), 17, 8, 8))  # 17 channels, 8 rows, 8 cols

        for i, state in enumerate(state_batch):  # Add each state in the batch to the output torch.Tensor
            board = chess.Board(state)  # Use the FEN string encoding to create the board
            for cell, piece in board.piece_map().items():  # Iter over the piece dictionary
                r, c = divmod(cell, 8)  # Get the row and column of this piece
                # By default, White's pieces will be in cells 0, 1, ... etc. and black's will be in cells
                # ... 63, 64 etc. We want to have friendly pieces always in row 7 so if its black's turn to
                # move, we will flip the columns only so that cell 64 (h8) maps to the bottom left corner. On
                # white's move, we will flip the rows only so that cell 0 (a1) maps to the bottom left corner
                if board.turn:  # White's turn to move
                    r = 7 - r  # Reverse the row order so that (0, 0) is in the bottom left corner instead of
                    # the top left corner of the grid representation
                else:  # Black's turn to move
                    c = 7 - c  # Reverse the col order, no need to reverse the row ordering since it already
                    # has the black pieces in the last row of the board, but the cols need reversing so that
                    # the king is in the correct position from black's perspective

                p = sym_to_int[piece.symbol().lower()]  # Convert from piece symbol (str) to integer [0, 5]
                if board.turn:  # If it is white's move, then upper-case (white) are the player's pieces
                    # which we will record in the indices [0, 5] of the third dimension
                    p = p + (6 if piece.symbol().islower() else 0)
                else:  # If it is black's move, then upper-case (white) pieces are the opponent's
                    p = p + (0 if piece.symbol().islower() else 6)
                output[i, p, r, c] = 1  # Record using one-hot-encoding

            # Add additional plates for encoding other important state information
            # A). Add castling rights on the king and queen side for friendly and foe
            if board.turn is chess.WHITE:  # If it is white's turn, then white is friendly and black is foe
                friendly_color, foe_color = chess.WHITE, chess.BLACK
            else:  # Otherwise if it is black's turn, then black is friendly and white is foe
                friendly_color, foe_color = chess.BLACK, chess.WHITE

            output[i, 12, :, :] = 1 if board.has_kingside_castling_rights(friendly_color) else 0
            output[i, 13, :, :] = 1 if board.has_queenside_castling_rights(friendly_color) else 0
            output[i, 14, :, :] = 1 if board.has_kingside_castling_rights(foe_color) else 0
            output[i, 15, :, :] = 1 if board.has_queenside_castling_rights(foe_color) else 0

            # B). Add en passant rights as well as a 17th plane if there are any
            if board.ep_square:  # Will be None if no en passant is possible, there is at most 1 en passant
                # on the board at any given time, it must immediately follow a pawn 2-cell jump
                ep_r, ep_c = divmod(board.ep_square, 8)
                # Apply the same perspective flip as for pieces
                if board.turn:  # White's turn
                    ep_r = 7 - ep_r
                else:  # Black's turn
                    ep_c = 7 - ep_c
                output[i, 16, ep_r, ep_c] = 1

            # C). 50 move rule counter - Encodes the number of half-moves since last capture or pawn move
            # when this counter reaches 100, then 50 whole moves have been made and the game ends in a draw
            # output[i, 17, :, :] = min(board.halfmove_clock, 100) / 100.0  # Scale to be [0, 1]

        return output

    def forward(self, state_batch: List[str]) -> Tuple[torch.Tensor]:
        """
        Forward pass through the model which generates a policy logit vector over all moves and a value
        estimate of the current board position for each input state observation in the input state_batch
        i.e. an estimate of what the best move would be to play next among all possible moves and also
        the expected reward from the perspective of the player to go next.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A tuple of torch.Tensors:
            policy_logits of size (batch_size, 1968)
            value_estimates of size (batch_size, )
        """
        device = next(self.parameters()).device  # Get the appropriate device
        if len(state_batch) == 0:  # For an empty input state_batch of length 0, return outputs of 0s
            return torch.zeros(0, 1968).to(device), torch.zeros(0).to(device)
        else:
            # Convert the input board into the expected state representation and pass it through the network
            if isinstance(state_batch, list):
                x = self.state_to_model_input(state_batch).to(device)
            elif isinstance(state_batch, torch.Tensor):  # Skip the conversion if already done
                x = state_batch.to(device)
            else:
                raise TypeError("state_batch expected to be either a list or torch.Tensor")

            # Get the shared deep latent features (batch_size, channels, 8, 8)
            features = self.res_tower(self.input_layer(x))
            # Pass these shared features to the policy and value sub-components to compute outputs
            policy_logits = self.policy_head(features)  # (batch_size, 1968)
            value_estimates = self.value_head(features).squeeze(1)  # (batch_size, )
            return policy_logits, value_estimates


############################################
### Transformer Value-Network Definition ###
############################################
# TODO: Section marker

class Transformer(nn.Module):
    """
    Implementation of a multi-headed self-attention transformer model for chess board policy and value
    estimation.
    """

    def __init__(self, config: Dict = None, *args, **kwargs):
        """
        Initializes a value network model as a transformer that follows the architecture of the original
        transformer paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762) with multiple
        layers of self-attention blocks followed by an average pooling and linear projection layer.

        The input to this network will be a batch of FEN string state representation of the current game
        state and output a torch.Tensor of the same length detailing the model's policy logits and value
        estimates for each input state.
        """
        super().__init__()
        config = {"model": {}} if config is None else config
        # Extract config parameters from the passed model config dictionary
        self.hidden_size = int(config["model"].get("hidden_size", 256))
        self.n_heads = int(config["model"].get("n_heads", 8))
        self.num_layers = int(config["model"].get("num_layers", 3))
        self.ff_dim = int(config["model"].get("ff_dim", 1024))
        self.dropout = float(config["model"].get("dropout", 0.2))

        # 1). Create a token-embedding layer for the pieces and castling rights. We have 1 token for blank
        # squares, 6 for friendly pieces, 6 for foe pieces, and 2 castling rights, either True or False
        # for a total of 1 + 6 + 6 + 2 = 15 unique token integer indices
        # Note that the blank squares with token value of 0 do not get a vector embedding representation,
        # only a positional encoding
        self.token_embeddings = nn.Embedding(num_embeddings=15, embedding_dim=self.hidden_size, padding_idx=0)

        # 2). Create a matrix of size [64, hidden_size] of learnable parameters which will serve as the
        # positional embeddings for each unique square on the board and will be added to the piece encodings
        self.pos_embeddings = nn.Parameter(torch.zeros(1, 64, self.hidden_size))

        # 3). Create a CLS token that will be added to the tokens before being passed to the encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # 4). Create the multi-headed self-attention transformer blocks, the core of the transformer model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads,
                                                        dim_feedforward=self.ff_dim, activation="gelu",
                                                        batch_first=True, norm_first=True, bias=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers,
                                             norm=nn.LayerNorm(self.hidden_size))

        # 5). Add a policy head sub-component that will operate off the output features from self.encoder
        # after they have been globally average pooled into a (batch_size, hidden_size) tensor
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 1968)  # 1968 possible UCI moves
        )  # Output: (batch_size, 1968) - returns raw logits, no softmax applied here

        # 6). Add a value head sub-component that will operate off the output features from self.encoder
        # after they have been globally average pooled into a (batch_size, hidden_size) tensor
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 1),
            # Use a final Tanh activation function at the end to produce value estimates [-1, +1]
            nn.Tanh()
        )

    def get_device(self) -> str:
        """
        Returns the device the model is currently on by returning the device of the parameters.
        """
        return next(self.parameters()).device.type

    @staticmethod
    def state_to_model_input(state_batch: List[str]) -> torch.Tensor:
        """
        Converts an input batch of board states (encoded using FEN as a string) into the expected state
        representations for this model (torch.Tensor).

        This transformer model operates on board representations that are (64 + 4, hidden_size) = (68, E) in
        size. The first 64 entries of the first dimension are the 64 positional embedding vectors of the board
        plus their respective piece token embeddings (if non-empty, empty squares use only a positional
        embedding vector). The last 4 entries along the first dimension are token embeddings for the 4
        castling rights variables i.e. white king-side, white queen-side, black king-side, black queen-side.
        For each castling rights token, there are 2 possible options, one representing True and the other
        representing False. Each token embedding is a length E (e.g. 128) vector and there will always be a
        fixed number (i.e. 68) passed in per batch element (board state).

        Locationally, the friendly pieces are always shown at the bottom of the board and foe pieces always
        are shown at the top of the board. Therefore, there is some board flipping depending on the color
        whose move it is next so that the model learns the same chess playing from the same perspective.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A torch.Tensor of size (batch_size, 68, E) containing vector embeddings.
        """
        sym_to_int = {s: (i + 1) for i, s in enumerate("pnbrqk")}  # Mapping from symbol to int starting at 1
        output = torch.zeros((len(state_batch), 8, 8), dtype=torch.int)  # 8 x 8 = 64 squares on a chess
        # board, use 0 to denote blank squares, [1, 6] for the friendly pieces and [7, 12] for foe
        castling = torch.zeros(len(state_batch), 4, dtype=torch.int)  # 4 possible castling rights

        for i, state in enumerate(state_batch):  # Add each state in the batch to the output torch.Tensor
            board = chess.Board(state)  # Use the FEN string encoding to create the board
            for cell, piece in board.piece_map().items():  # Iter over the piece dictionary
                r, c = divmod(cell, 8)  # Get the row and column of this piece
                # By default, White's pieces will be in cells 0, 1, ... etc. and black's will be in cells
                # ... 63, 64 etc. We want to have friendly pieces always in row 7 so if its black's turn to
                # move, we will flip the columns only so that cell 64 (h8) maps to the bottom left corner. On
                # white's move, we will flip the rows only so that cell 0 (a1) maps to the bottom left corner
                if board.turn:  # White's turn to move
                    r = 7 - r  # Reverse the row order
                else:  # Black's turn to move
                    c = 7 - c  # Reverse the col order
                p = sym_to_int[piece.symbol().lower()]  # Convert from piece symbol (str) to integer [1, 6]
                if board.turn:  # If it is white's move, then upper-case (white) are the player's pieces
                    # which we will record with piece tokens [1, 6], otherwise record them with [7, 12] ints
                    p = p + (6 if piece.symbol().islower() else 0)
                else:  # If it is black's move, then upper-case (white) pieces are the opponent's pieces,
                    # which we will record with piece tokens [1, 6], otherwise record them with [7, 12] ints
                    p = p + (6 if piece.symbol().isupper() else 0)
                output[i, r, c] = p  # Record the piece token int [1, 12], leave zero for blank cells

            # Encode the castling rights
            if board.turn is chess.WHITE:  # If it is white's turn, then white is friendly and black is foe
                friendly_color, foe_color = chess.WHITE, chess.BLACK
            else:  # Otherwise if it is black's turn, then black is friendly and white is foe
                friendly_color, foe_color = chess.BLACK, chess.WHITE

            # Encode the True and False of each castling right for each player as a separate input, use
            # 1 token for True and 1 for False for if castling rights are there for each player on each side
            castling[i, 0] = 13 if board.has_kingside_castling_rights(friendly_color) else 14
            castling[i, 1] = 13 if board.has_queenside_castling_rights(friendly_color) else 14
            castling[i, 2] = 13 if board.has_kingside_castling_rights(foe_color) else 14
            castling[i, 3] = 13 if board.has_queenside_castling_rights(foe_color) else 14

        # Output is now (batch_size, 8, 8) and has the integers [1, 6] for friendly pieces, [7, 12] for foe
        # pieces and 0s for the empty squares which will be treated as padding tokens by the embedding layer
        output = output.reshape(len(state_batch), -1)  # Reshape to (batch_size, 64) to flatten

        # Add the additional 4 tokens for castling rights before returning the state representation
        return torch.concat([output, castling], dim=1)  # (batch_size, 68) ints torch.Tensor

    def forward(self, state_batch: List[str]) -> torch.Tensor:
        """
        Forward pass through the model which generates a value estimate of the current board position for
        each state observation in the input state_batch i.e. an estimate of the expected reward from the
        current state position.

        :param state_batch: A batch of FEN states as a list of strings.
        :return: A tuple of torch.Tensors:
            policy_logits of size (batch_size, 1968)
            value_estimates of size (batch_size, )
        """
        device = next(self.parameters()).device  # Get the appropriate device
        if len(state_batch) == 0:  # For an empty input state_batch of length 0, return outputs of 0s
            return torch.zeros(0, 1968).to(device), torch.zeros(0).to(device)
        else:
            # Convert the input board into the expected state representation and pass it through the network
            if isinstance(state_batch, list):
                x = self.state_to_model_input(state_batch).to(device)
            elif isinstance(state_batch, torch.Tensor):  # Skip the conversion if already done
                x = state_batch.to(device)
            else:
                raise TypeError("state_batch expected to be either a list or torch.Tensor")

            # Pass the token integers through the embedding layer to convert them into embedding vectors
            x = self.token_embeddings(x)  # (batch_size, 68) -> (batch_size, 68, hidden_size)

            # Add positional encodings to the first 64 elements corresponding to tiles on the chess board
            x[:, :64, :] += self.pos_embeddings  # (batch_size, 68, hidden_size)

            # Prepend CLS token to the sequence
            cls = self.cls_token.expand(x.size(0), -1, -1)  # (batch_size, 1, hidden_size)
            x = torch.cat([cls, x], dim=1)  # (batch_size, 69, hidden_size)

            x = self.encoder(x)  # Pass x through the encoder blocks, (batch_size, 68, hidden_size)
            # Use CLS token output as the final set of hidden features to operate on in each head
            x = x[:, 0, :]  # (batch_size, hidden_size) — take the first token's output

            # Pass the global average pooling outputs from the encoder to each head for final predictions
            policy_logits = self.policy_head(x)
            value_estimates = self.value_head(x).squeeze(1)  # (batch_size, )
            return policy_logits, value_estimates
