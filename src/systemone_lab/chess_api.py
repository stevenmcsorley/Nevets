"""Chess endpoints for the live board. Rules come from python-chess; the model only chooses moves."""
import os
import random
import threading
import time
from pathlib import Path

import chess
import chess.engine
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .chess_format import request
from .eval import predict_record
from .model import SystemOneModel
from .tokenizer import LabTokenizer
from .training import load_checkpoint, pick_device

CHESS_CKPT = os.environ.get("S1_CHESS_CHECKPOINT", "checkpoints/s1-35m-chess.pt")
ENGINE = Path(__file__).resolve().parents[2] / "tools" / "stockfish" / "stockfish" / "stockfish-windows-x86-64-universal.exe"
router = APIRouter()
_lock = threading.Lock()
_state = {"model": None, "tok": None, "engine": None, "device": pick_device()}


class ChessTurn(BaseModel):
    fen: str = chess.STARTING_FEN
    player: str = "model"          # model | random | stockfish | human
    move: str | None = None        # UCI, for player == "human"
    elo: int = 1320                # Stockfish strength when player == "stockfish"


def _model():
    if _state["model"] is None:
        if not Path(CHESS_CKPT).exists():
            raise HTTPException(503, f"Chess checkpoint not found: {CHESS_CKPT}")
        m, ck = load_checkpoint(CHESS_CKPT, SystemOneModel, _state["device"])
        _state["model"], _state["tok"] = m.to(_state["device"]).eval(), LabTokenizer(ck["tokenizer"])
    return _state["model"], _state["tok"]


def _engine():
    if _state["engine"] is None:
        if not ENGINE.exists(): raise HTTPException(503, "Stockfish not installed under tools/stockfish")
        _state["engine"] = chess.engine.SimpleEngine.popen_uci(str(ENGINE))
    return _state["engine"]


def _status(board):
    if board.is_checkmate(): return "checkmate", board.result()
    if board.is_stalemate(): return "stalemate", "1/2-1/2"
    if board.is_insufficient_material(): return "insufficient material", "1/2-1/2"
    if board.can_claim_draw(): return "draw by repetition or 50-move rule", "1/2-1/2"
    return ("check" if board.is_check() else "playing"), None


@router.post("/v1/chess")
def chess_turn(turn: ChessTurn):
    try:
        board = chess.Board(turn.fen)
    except ValueError as e:
        raise HTTPException(400, f"bad FEN: {e}")
    if _status(board)[1] is not None:
        raise HTTPException(400, "game is over")
    out = {"player": turn.player}
    t0 = time.perf_counter()
    with _lock:
        if turn.player == "model":
            model, tok = _model()
            with torch.no_grad():
                probs = predict_record(model, tok, request(board), _state["device"])["move"]
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            move = chess.Move.from_uci(ranked[0][0])
            out["candidates"] = [{"uci": u, "san": board.san(chess.Move.from_uci(u)), "p": p} for u, p in ranked[:6]]
            out["confidence"] = ranked[0][1]
        elif turn.player == "random":
            move = random.choice(list(board.legal_moves))
        elif turn.player == "stockfish":
            eng = _engine()
            eng.configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, min(3190, turn.elo))})
            move = eng.play(board, chess.engine.Limit(time=0.1)).move
        elif turn.player == "human":
            try:
                move = chess.Move.from_uci(turn.move or "")
            except ValueError:
                raise HTTPException(400, "bad move")
            if move not in board.legal_moves:
                # Auto-promote to a queen when the UI sends a bare pawn move to the last rank.
                queen = chess.Move(move.from_square, move.to_square, chess.QUEEN)
                if queen in board.legal_moves: move = queen
                else: raise HTTPException(400, "illegal move")
        else:
            raise HTTPException(400, "unknown player")
    out.update({"uci": move.uci(), "san": board.san(move), "ms": (time.perf_counter() - t0) * 1000})
    board.push(move)
    status, result = _status(board)
    out.update({"fen": board.fen(), "status": status, "result": result,
                "legal": [m.uci() for m in board.legal_moves]})
    return out


@router.get("/v1/chess/legal")
def chess_legal(fen: str = chess.STARTING_FEN):
    board = chess.Board(fen)
    return {"legal": [m.uci() for m in board.legal_moves], "status": _status(board)[0]}
