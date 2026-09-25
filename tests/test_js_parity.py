"""The browser pipeline (site/js/s1.js) must tokenize and pack exactly like Python."""
import itertools
import json
import shutil
import subprocess

import chess
import pytest

from systemone_lab.chess_format import request as chess_request
from systemone_lab.formatting import branch_attention_mask, pack_request
from systemone_lab.tokenizer import LabTokenizer

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
TEXTS = ['{"facts":[{"subject":"a"}]} | x (y, z)', "A is left of B. C is below B.", "obj_12 sits north of obj_3.", "café naïve — “quoted” 😀 x",
         "  leading and   multiple   spaces ", "Pe7-e8=Q Ke1-g1 castles", "don't we'll it's", "tab\there\nnewline", ""]


def cases():
    out = [{"text": t} for t in TEXTS]
    for path, step in (("reports/chain/eval_hops.jsonl", 40), ("reports/binding_stress/stress.jsonl", 40)):
        for line in itertools.islice(open(path, encoding="utf-8"), 0, 4000, step):
            r = json.loads(line); out.append({"state": r["state"], "questions": r["questions"], "bidirectional": True})
            out.append({"state": r["state"], "questions": r["questions"], "bidirectional": True, "isolate": True})
    from systemone_lab.worlds import DOMAINS, FORMATS
    wrng = __import__("random").Random(5)
    for dom in DOMAINS:  # every data-factory domain in every rendering (JSON, tables, kv, prose)
        for fmt in FORMATS:
            r, _ = DOMAINS[dom](wrng, fmt, "p")
            out.append({"state": r["state"], "questions": r["questions"], "bidirectional": True, "isolate": True})
    board = chess.Board()
    for mv in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O", "h3"]:
        board.push_san(mv); r = chess_request(board)
        out.append({"state": r["state"], "questions": r["questions"], "bidirectional": True})
    return out


def test_js_pack_matches_python(tmp_path):
    tok = LabTokenizer("data/tokenizer.json"); cs = cases()
    (tmp_path / "cases.json").write_text(json.dumps(cs), encoding="utf-8")
    res = subprocess.run(["node", "tests/js_parity/pack.mjs", "data/tokenizer.json", str(tmp_path / "cases.json")],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(res.stdout); assert len(js) == len(cs)
    for c, j in zip(cs, js):
        if "text" in c:
            ids, offs = tok.encode_with_offsets(c["text"])
            assert j["ids"] == ids, c["text"]
            # Byte-level offsets may include a leading space in the JS port; both must cover the same token text.
            for (a, b), (x, y) in zip(j["offsets"], offs):
                assert c["text"][a:b].strip() == c["text"][x:y].strip()
            continue
        iso = bool(c.get("isolate"))
        p = pack_request(tok, c["state"], c["questions"], isolate_options=iso)
        assert j["ids"] == p.input_ids.tolist() and j["pos"] == p.position_ids.tolist() and j["branch"] == p.branch_ids.tolist()
        assert j["opt"] == p.option_ids.tolist()
        m = branch_attention_mask(p.branch_ids, c["bidirectional"], p.option_ids if iso else None)
        assert j["mask_sum"] == int(m.sum()) and j["mask_rows"] == m.sum(-1).tolist()
        for jl, pl in zip(j["layouts"], p.layouts):
            assert jl["keys"] == pl.option_keys and jl["optionEnds"] == pl.option_end_positions and jl["decide"] == pl.decide_position
            want = [list(x) for x in pl.query_entity_state_positions] if pl.query_entity_state_positions else None
            assert jl["statePositions"] == want


def test_js_chess_requests_match_python(tmp_path):
    import random
    rng = random.Random(0); fens = []
    for _ in range(40):  # random games reach castling, promotions, en passant, checks
        b = chess.Board()
        for _ in range(rng.randint(0, 160)):
            if b.is_game_over(): break
            b.push(rng.choice(list(b.legal_moves))); fens.append(b.fen())
    fens += ["rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",   # legal en passant
             "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",      # ep square in FEN, still capturable
             "8/P6k/8/8/8/8/6Kp/8 w - - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"]
    (tmp_path / "fens.json").write_text(json.dumps(fens))
    res = subprocess.run(["node", "tests/js_parity/chess.mjs", str(tmp_path / "fens.json")],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(res.stdout)
    for fen, j in zip(fens, js):
        py = chess_request(chess.Board(fen))
        assert j["state"] == py["state"], fen
        assert j["questions"] == py["questions"], fen
