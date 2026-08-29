const boardElement = document.getElementById("board");
const statusElement = document.getElementById("status");
const movesElement = document.getElementById("moves");
const newGameButton = document.getElementById("new-game");


const promotionModal =
    document.getElementById("promotion-modal");

const promotionButtons =
    document.querySelectorAll(
        ".promotion-button"
    );

const PIECES = {
    "P": "/static/pieces/white-pawn.svg.webp",
    "N": "/static/pieces/white-knight.svg.webp",
    "B": "/static/pieces/white-bishop.svg.webp",
    "R": "/static/pieces/white-rook.svg.webp",
    "Q": "/static/pieces/white-queen.svg.webp",
    "K": "/static/pieces/white-king.svg.webp",

    "p": "/static/pieces/black-pawn.svg.webp",
    "n": "/static/pieces/black-knight.svg.webp",
    "b": "/static/pieces/black-bishop.svg.webp",
    "r": "/static/pieces/black-rook.svg.webp",
    "q": "/static/pieces/black-queen.svg.webp",
    "k": "/static/pieces/black-king.svg.webp"
};


let position = [
    ["r","n","b","q","k","b","n","r"],
    ["p","p","p","p","p","p","p","p"],
    [null,null,null,null,null,null,null,null],
    [null,null,null,null,null,null,null,null],
    [null,null,null,null,null,null,null,null],
    [null,null,null,null,null,null,null,null],
    ["P","P","P","P","P","P","P","P"],
    ["R","N","B","Q","K","B","N","R"]
];


let selectedSquare = null;
let playerTurn = true;
let moveNumber = 1;
let lastMove = null;
let pendingPromotion = null;
let checkSquare = null;
let legalMoves = [];

promotionModal.classList.remove(
    "show"
);

// =====================================
// DRAW BOARD
// =====================================

function drawBoard() {
    boardElement.innerHTML = "";
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const square = document.createElement("div");
            square.classList.add("square");

            // Board colors
            if ((row + col) % 2 === 0) {
                square.classList.add("light");
            } else {
                square.classList.add("dark");
            }


            // Square name, e.g. e4
            const squareName = String.fromCharCode(97 + col) + (8 - row);
            square.dataset.square = squareName;

            if (legalMoves.includes(squareName)) {
                const targetPiece = position[row][col];

                if (targetPiece) {
                    square.classList.add("legal-capture");
                } else {
                    square.classList.add("legal");
                }
            }

            // Last-move highlighting
            if (
                lastMove &&
                (
                    squareName === lastMove.from ||
                    squareName === lastMove.to
                )
            ) {

                square.classList.add("last-move");
            }

            // King-in-check highlighting
            if (squareName === checkSquare) {square.classList.add("in-check");
            }


            // Piece

            const piece =
                position[row][col];


            if (piece) {

                const pieceElement =
                    document.createElement("img");

                pieceElement.classList.add("piece");

                pieceElement.src =
                    PIECES[piece];

                pieceElement.alt = "";

                pieceElement.draggable = false;

                square.appendChild(
                    pieceElement
                );
            }


            // Click handler

            square.addEventListener(
                "click",
                function () {
                    handleSquareClick(squareName);
                }
            );


            boardElement.appendChild(square);
        }
    }
}


// =====================================
// HANDLE SQUARE CLICK
// =====================================
async function handleSquareClick(square) {
    if (!playerTurn) return;

    if (!selectedSquare) {
        if (getPiece(square)) {
            selectedSquare = square;
            await showLegalMoves(square);
        }
        return;
    }

    if (selectedSquare === square) {
        selectedSquare = null;
        legalMoves = [];
        drawBoard();
        return;
    }

    const from = selectedSquare;
    const to = square;

    if (!legalMoves.includes(to)) {
        return;
    }

    selectedSquare = null;
    legalMoves = [];
    drawBoard();

    if (isPromotionMove(from, to)) {
        showPromotionDialog(from, to);
        return;
    }

    makeMove(from, to);
}

async function showLegalMoves(square) {
    try {
        const response = await fetch("/legal-moves", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ square })
        });

        const result = await response.json();

        if (!result.success) return;

        legalMoves = result.moves;
        drawBoard();
        highlightSquare(square);

    } catch (error) {
        console.error("Error getting legal moves:", error);
    }
}



// =====================================
// HIGHLIGHT SELECTED PIECE
// =====================================

function highlightSquare(squareName) {

    const square =
        document.querySelector(
            `[data-square="${squareName}"]`
        );

    if (square) {
        square.classList.add("selected");
    }
}

function showPromotionDialog(from, to) {

    pendingPromotion = {
        from: from,
        to: to
    };

    promotionModal.classList.add("show");
}

promotionButtons.forEach(function(button) {

    button.addEventListener("click", function() {

        if (!pendingPromotion) {
            console.error(
                "No pending promotion"
            );
            return;
        }

        const promotion = button.dataset.piece;
        const from = pendingPromotion.from;
        const to = pendingPromotion.to;

        // Build the complete UCI move.
        // Example: e7e8q
        const move = from + to + promotion;
        console.log("Promotion move:", move);

        // Close the dialog
        promotionModal.classList.remove("show");

        // Clear pending promotion
        pendingPromotion = null;

        // Send complete move
        makeMove(from, to, promotion);
    });
});




// =====================================
// GET PIECE
// =====================================

function getPiece(squareName) {

    const col =
        squareName.charCodeAt(0) - 97;

    const row =
        8 - parseInt(squareName[1]);

    return position[row][col];
}


function isPromotionMove(from, to) {

    const piece = getPiece(from);

    // White pawn reaching rank 8
    if (piece === "P" && to[1] === "8") {
        return true;
    }

    // Black pawn reaching rank 1
    if (piece === "p" && to[1] === "1") {
        return true;
    }

    return false;
}


// =====================================
// SEND MOVE TO FLASK
// =====================================

async function makeMove(from, to, promotion = null) {

    // Prevent another move while
    // the request is happening.

    playerTurn = false;
    statusElement.textContent = "Your move...";
    const move = from + to + (promotion || "");

    try {
        // =================================
        // STEP 1
        // Send player's move
        // =================================

        const response =
            await fetch("/move", {
                method: "POST",
                headers: {
                "Content-Type": "application/json"
                },

                body: JSON.stringify({move: move
                })
            });


        const result =
            await response.json();


        // =================================
        // ILLEGAL MOVE
        // =================================

        if (!result.success) {
            statusElement.textContent = "Illegal move";
            playerTurn = true;
            drawBoard();
            return;
        }


        // =================================
        // PLAYER'S MOVE IS NOW COMPLETE
        // =================================

        lastMove = {from: from, to: to};
        checkSquare = result.check_square || null;


        // =================================
        // UPDATE BOARD IMMEDIATELY
        // =================================

        loadFEN(result.fen);
        addMove(move, null);


        // =================================
        // DID THE PLAYER'S MOVE END
        // THE GAME?
        // =================================
        if (result.game_over) {
            statusElement.textContent = result.status;
            playerTurn = false;
            return;
        }


        // =================================
        // AI IS NOW THINKING
        // =================================

        statusElement.textContent = "AI Chess Engine is thinking...";


        // =================================
        // STEP 2
        // Ask Flask to make AI move
        // =================================

        await makeAIMove();


    } catch (error) {

        console.error(
            "Error communicating with Flask:",
            error
        );

        statusElement.textContent =
            "Server error";

        playerTurn = true;
    }
}

async function makeAIMove() {

    try {
        const response = await fetch("/ai-move", {method: "POST"});
        const result = await response.json();

        if (!result.success) {
            statusElement.textContent = "AI error";
            playerTurn = true;
            return;
        }


        // =================================
        // RECORD AI'S MOVE
        // =================================
        if (result.ai_move) {
            lastMove = {
                from: result.ai_move.substring(0, 2),
                to: result.ai_move.substring(2, 4)
            };
            addMove(null, result.ai_move);
        }

        // =================================
        // UPDATE CHECK STATUS
        // =================================

        checkSquare =
            result.check_square || null;


        // =================================
        // UPDATE BOARD
        // =================================

        loadFEN(result.fen);


        // =================================
        // GAME OVER?
        // =================================

        if (result.game_over) {

            statusElement.textContent =
                result.status;

            playerTurn = false;

            return;
        }


        // =================================
        // BACK TO PLAYER
        // =================================

        statusElement.textContent =
            "Your turn";

        playerTurn = true;


    } catch (error) {

        console.error(
            "Error communicating with AI:",
            error
        );

        statusElement.textContent =
            "AI error";

        playerTurn = true;
    }
}






// =====================================
// FEN → JAVASCRIPT BOARD
// =====================================

function loadFEN(fen) {

    const boardPart =
        fen.split(" ")[0];

    const rows =
        boardPart.split("/");

    const newPosition = [];


    for (const row of rows) {

        const boardRow = [];


        for (const character of row) {

            if (
                character >= "1" &&
                character <= "8"
            ) {

                const count =
                    parseInt(character);


                for (
                    let i = 0;
                    i < count;
                    i++
                ) {

                    boardRow.push(null);
                }

            } else {

                boardRow.push(character);
            }
        }


        newPosition.push(boardRow);
    }


    position =
        newPosition;

    drawBoard();
}


// =====================================
// MOVE LIST
// =====================================

function addMove(playerMove = null, aiMove = null) {
    const move = document.createElement("div");
    move.classList.add("move");

    if (playerMove) {
        move.textContent = `${moveNumber}. ${playerMove}`;
    } else if (aiMove) {
        move.textContent = `   ${aiMove}`;
        moveNumber++;
    }

    movesElement.appendChild(move);
    movesElement.scrollTop = movesElement.scrollHeight;
}

// =====================================
// NEW GAME
// =====================================

newGameButton.addEventListener(
    "click",
    async function () {

        try {

            const response =
                await fetch(
                    "/new-game",
                    {
                        method: "POST"
                    }
                );


            const result =
                await response.json();


        if (result.success) {
            moveNumber = 1;
            playerTurn = true;
            selectedSquare = null;
            lastMove = null;
            checkSquare = null;
            legalMoves = [];
            pendingPromotion = null;

            promotionModal.classList.remove("show");
            movesElement.innerHTML = "";
            loadFEN(result.fen);
            statusElement.textContent = "Your turn";
        }


        } catch (error) {

            console.error(
                "Error starting new game:",
                error
            );

            statusElement.textContent =
                "Server error";
        }
    }
);


// =====================================
// START GAME
// =====================================

drawBoard();
