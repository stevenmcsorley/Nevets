"""Evaluate a chess move-choice checkpoint.

Position metrics on held-out Stockfish-labelled positions: agreement with Stockfish's best move,
top-3 agreement, mean centipawn loss of the chosen move, blunder rate (loss >= 200cp), ECE15.
Game metrics: full games against a uniform-random mover and against Stockfish at its weakest
strength setting (UCI_LimitStrength, UCI_Elo 1320), alternating colours.
"""
import argparse
import json
import random
from pathlib import Path

import chess
import chess.engine

from systemone_lab.chess_format import request
from systemone_lab.data.io import read_jsonl
from systemone_lab.eval import predict_record, ece
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint, pick_device

ENGINE = str(Path("tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe").resolve())


def model_move(model, tok, board, device):
    p = predict_record(model, tok, request(board), device)["move"]
    uci = max(p, key=p.get)
    return chess.Move.from_uci(uci), p[uci]


def play(model, tok, device, opponent, model_white, rng, engine=None, max_plies=300):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == model_white:
            move, _ = model_move(model, tok, board, device)
        elif opponent == "random":
            move = rng.choice(list(board.legal_moves))
        else:
            move = engine.play(board, chess.engine.Limit(time=0.05)).move
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "*": return "draw"
    won = (result == "1-0") == model_white
    return "draw" if result == "1/2-1/2" else ("win" if won else "loss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=3000); ap.add_argument("--games", type=int, default=20)
    a = ap.parse_args(); device = pick_device()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval(); tok = LabTokenizer(ck["tokenizer"])
    top1 = top3 = blunders = n = 0; loss = 0.0; conf = []; correct = []
    for rec in list(read_jsonl(a.data))[:a.positions]:
        p = predict_record(model, tok, rec, device)["move"]; scores = rec["meta"]["scores_cp"]
        ranked = sorted(p, key=p.get, reverse=True); best = max(scores, key=scores.get)
        cp = max(0, scores[best] - scores[ranked[0]])
        n += 1; top1 += ranked[0] == best; top3 += best in ranked[:3]; loss += min(cp, 1000); blunders += cp >= 200
        conf.append(p[ranked[0]]); correct.append(int(ranked[0] == best))
    out = {"ckpt": a.ckpt, "positions": n, "top1_agreement": top1 / n, "top3_agreement": top3 / n,
           "mean_cp_loss_capped1000": loss / n, "blunder_rate_200cp": blunders / n, "ece15_top1": ece(conf, correct, 15)}
    rng = random.Random(0); games = {}
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Threads": 1, "UCI_LimitStrength": True, "UCI_Elo": 1320})
    try:
        for opponent in ("random", "stockfish_1320"):
            tally = {"win": 0, "draw": 0, "loss": 0}
            for g in range(a.games):
                tally[play(model, tok, device, "random" if opponent == "random" else "stockfish", g % 2 == 0, rng, engine)] += 1
            games[opponent] = tally
    finally:
        engine.quit()
    out["games"] = games
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
