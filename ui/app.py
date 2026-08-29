from flask import Flask, render_template, request, jsonify

from game import Game

app = Flask(__name__)

game = Game()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/move")
def move():
    data = request.json
    player_move = data["move"]
    success = game.make_player_move(player_move)

    if not success:
        return jsonify({
            "success": False,
            "error": "Illegal move"
        })

    game_status = game.get_game_status()
    check_square = game.get_check_square()

    return jsonify({
        "success": True,
        "fen": game.get_fen(),
        "game_over": game_status["game_over"],
        "status": game_status["status"],
        "check_square": check_square
    })


@app.post("/ai-move")
def ai_move():
    # Let the AI make its move
    ai_move = game.make_engine_move()
    # Check resulting game state
    game_status = game.get_game_status()
    check_square = game.get_check_square()

    return jsonify({
        "success": True,
        "fen": game.get_fen(),
        "ai_move": ai_move,
        "game_over": game_status["game_over"],
        "status": game_status["status"],
        "check_square": check_square
    })


@app.post("/new-game")
def new_game():
    game.reset()

    return jsonify({
        "success": True,
        "fen": game.get_fen()
    })


@app.post("/legal-moves")
def legal_moves():
    data = request.json
    square = data["square"]

    moves = game.get_legal_moves(square)

    return jsonify({
        "success": True,
        "moves": moves
    })


if __name__ == "__main__":
    app.run(debug=True)
