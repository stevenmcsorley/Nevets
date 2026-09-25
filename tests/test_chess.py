import chess
import pytest

pytest.importorskip("chess")
from systemone_lab.chess_format import state_text, move_text, question, request
from systemone_lab.formatting import pack_request
from systemone_lab.tokenizer import LabTokenizer


def test_state_lists_every_piece_and_side():
    b = chess.Board()
    s = state_text(b)
    assert s.startswith("White to move.") and "Ke1" in s and "Ke8" in s
    assert s.count(" P") == 16 and "Castling: KQkq." in s and "En passant: none." in s
    b.push_san("e4"); b.push_san("f5"); b.push_san("Qh5")
    assert state_text(b).endswith("Check.")


def test_options_are_all_legal_moves_keyed_by_uci():
    b = chess.Board("r3k2r/pPppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    q = question(b)
    assert set(q["criteria"]) == {m.uci() for m in b.legal_moves}
    assert q["criteria"]["e1g1"] == "Ke1-g1 castles"
    assert q["criteria"]["b7a8q"] == "Pb7xa8=Q"
    assert move_text(b, chess.Move.from_uci("e5f7")) == "Ne5xf7"


def test_chess_request_packs_with_every_option():
    tok = LabTokenizer("data/tokenizer.json")
    b = chess.Board()
    r = request(b)
    p = pack_request(tok, r["state"], r["questions"])
    assert len(p.layouts[0].option_keys) == 20 and p.layouts[0].query_entity_token_ids is None


def test_chess_api_rules(monkeypatch):
    from fastapi.testclient import TestClient
    from systemone_lab import api
    c = TestClient(api.app)
    r = c.post("/v1/chess", json={"player": "human", "move": "e2e4"}).json()
    assert r["san"] == "e4" and "e7e5" in r["legal"]
    assert c.post("/v1/chess", json={"player": "human", "move": "e2e5"}).status_code == 400
    promo = "8/P7/8/8/8/8/8/k6K w - - 0 1"
    assert c.post("/v1/chess", json={"fen": promo, "player": "human", "move": "a7a8"}).json()["san"] == "a8=Q+"
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert c.post("/v1/chess", json={"fen": mate, "player": "random"}).status_code == 400
