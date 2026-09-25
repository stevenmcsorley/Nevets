"""Text rendering of chess positions for the decision model.

The state lists every piece by colour and square; each legal move is an option written as piece,
from-square, '-' or 'x', to-square and optional promotion (e.g. "Ng1-f3", "Pe5xd6", "Pe7-e8=Q").
The from-square text also appears in the state, which lets a move attend to its piece.
Option keys are UCI strings so callers can apply the chosen move directly.
"""
import chess

ORDER = "KQRBNP"


def state_text(board: chess.Board) -> str:
    side = "White" if board.turn == chess.WHITE else "Black"
    parts = [f"{side} to move."]
    for colour, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        pieces = []
        for symbol in ORDER:
            for sq in sorted(board.pieces(chess.Piece.from_symbol(symbol).piece_type, colour)):
                pieces.append(symbol + chess.square_name(sq))
        parts.append(f"{name}: {' '.join(pieces)}.")
    castling = board.castling_xfen()
    parts.append(f"Castling: {castling if castling != '-' else 'none'}.")
    ep = board.ep_square if board.has_legal_en_passant() else None
    parts.append(f"En passant: {chess.square_name(ep) if ep is not None else 'none'}.")
    if board.is_check():
        parts.append("Check.")
    return " ".join(parts)


def move_text(board: chess.Board, move: chess.Move) -> str:
    piece = board.piece_at(move.from_square)
    text = piece.symbol().upper() + chess.square_name(move.from_square)
    text += ("x" if board.is_capture(move) else "-") + chess.square_name(move.to_square)
    if move.promotion:
        text += "=" + chess.piece_symbol(move.promotion).upper()
    if board.is_castling(move):
        text += " castles"
    return text


def question(board: chess.Board) -> dict:
    side = "White" if board.turn == chess.WHITE else "Black"
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    return {"type": "choice", "instructions": f"Which move is best for {side}?",
            "criteria": {m.uci(): move_text(board, m) for m in moves}}


def request(board: chess.Board, key: str = "move") -> dict:
    return {"state": state_text(board), "questions": {key: question(board)}}
