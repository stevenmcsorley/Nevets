"""Inference latency: single question, many questions over one shared state, and batched throughput."""
import argparse
import json
import time

import torch

from systemone_lab.chess_format import request as chess_request
from systemone_lab.formatting import pack_request, branch_attention_mask
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint, pick_device

LABELS = ["upper-left", "above", "upper-right", "left", "overlap", "right", "lower-left", "below", "lower-right"]


def timed(fn, reps):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out"); a = ap.parse_args()
    device = pick_device(); model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval()
    tok = LabTokenizer(ck["tokenizer"]); dtype = torch.bfloat16
    state = "A is left of B. C is above B. D is right of C. E sits south of D. F is northwest of A."
    names = "ABCDEF"

    def questions(n):
        qs = {}
        for i in range(n):
            x, y = names[i % 6], names[(i + 1 + i // 6) % 6]
            qs[f"q{i}"] = {"type": "choice", "instructions": f"What is the spatial relation of {x} to {y}?", "criteria": {l: l for l in LABELS}}
        return qs

    def run(packed):
        mask = model.attention_mask(packed)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            h = model.hidden(packed.input_ids[None], packed.position_ids[None], mask)
            for l in packed.layouts:
                model.decision_logits(h, l.decide_position, l.option_end_positions,
                                      query_entity_token_ids=l.query_entity_token_ids,
                                      query_entity_state_positions=l.query_entity_state_positions)

    out = {"ckpt": a.ckpt, "parameters": model.num_parameters(), "device": torch.cuda.get_device_name(0),
           "loop_iters": model.cfg.decision_head.get("loop_iters", 0)}
    for n in (1, 8, 32):
        p = pack_request(tok, state, questions(n), device=device, isolate_options=model.isolated_options)
        out[f"spatial_{n}_questions_ms"] = timed(lambda: run(p), 30); out[f"spatial_{n}_questions_tokens"] = p.input_ids.numel()
    import chess
    board = chess.Board("r3k2r/pPppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    r = chess_request(board); p = pack_request(tok, r["state"], r["questions"], device=device, isolate_options=model.isolated_options)
    out["chess_56_moves_ms"] = timed(lambda: run(p), 30); out["chess_tokens"] = p.input_ids.numel()
    print(json.dumps(out, indent=2))
    if a.out: open(a.out, "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
