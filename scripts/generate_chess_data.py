"""Chess move-choice data: positions sampled from Lichess games, every legal move scored by Stockfish.

Each record's target is a distribution over legal moves, softmax(score_cp / tau) from a MultiPV search
covering all legal moves; the hard label is Stockfish's best move. Source games come from the Lichess
open database (CC0). Stockfish (GPL-3) runs as a separate process and is used only as a labeler.
"""
import argparse
import io
import json
import math
import multiprocessing as mp
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import zstandard

from systemone_lab.chess_format import request

ENGINE = str(Path("tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe").resolve())


def sample_positions(pgn_path, n_positions, per_game, seed, skip_fens=frozenset()):
    rng = random.Random(seed); seen = set(skip_fens); out = []
    with open(pgn_path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace")
        while len(out) < n_positions:
            game = chess.pgn.read_game(text)
            if game is None: break
            moves = list(game.mainline_moves())
            if len(moves) < 8: continue
            plies = rng.sample(range(4, len(moves)), min(per_game, len(moves) - 4))
            for ply in sorted(plies):
                board = game.board()
                for mv in moves[:ply]: board.push(mv)
                if board.is_game_over() or board.legal_moves.count() < 2: continue
                key = " ".join(board.fen().split()[:4])
                if key in seen: continue
                seen.add(key); out.append(board.fen())
    return out


def worker(args):
    fens, depth, tau, wid = args
    eng = chess.engine.SimpleEngine.popen_uci(ENGINE); eng.configure({"Threads": 1, "Hash": 32})
    rows = []
    try:
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            infos = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=board.legal_moves.count())
            scores = {info["pv"][0].uci(): info["score"].pov(board.turn).score(mate_score=10000) for info in infos}
            if len(scores) != board.legal_moves.count(): continue
            best = max(scores.values())
            weights = {m: math.exp(max(-30.0, (s - best) / tau)) for m, s in scores.items()}
            z = sum(weights.values())
            rec = request(board)
            rec.update({"id": f"chess-{wid}-{i}", "labels": {"move": max(scores, key=scores.get)},
                        "distributions": {"move": {m: w / z for m, w in weights.items()}},
                        "meta": {"fen": fen, "scores_cp": scores, "depth": depth}})
            rows.append(rec)
    finally:
        eng.quit()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, required=True); ap.add_argument("--per-game", type=int, default=2)
    ap.add_argument("--depth", type=int, default=10); ap.add_argument("--tau", type=float, default=80.0)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude", action="append", default=[], help="jsonl whose positions must not repeat")
    a = ap.parse_args()
    skip = set()
    for path in a.exclude:
        skip |= {" ".join(json.loads(l)["meta"]["fen"].split()[:4]) for l in open(path, encoding="utf-8")}
    fens = sample_positions(a.pgn, a.positions, a.per_game, a.seed, frozenset(skip))
    chunks = [fens[i::a.workers] for i in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        parts = pool.map(worker, [(c, a.depth, a.tau, w) for w, c in enumerate(chunks)])
    rows = [r for part in parts for r in part]; random.Random(a.seed).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"positions": len(rows), "depth": a.depth, "tau": a.tau, "source": a.pgn}))


if __name__ == "__main__":
    main()
