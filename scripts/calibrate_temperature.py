"""Fit the cosine scorer's temperature on a development set and save a calibrated copy.

Temperature scaling does not change any prediction (argmax is unchanged); it only rescales confidence.
Fit only on development data, never on a locked evaluation set.
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from systemone_lab.model import SystemOneModel
from systemone_lab.training import load_checkpoint, pick_device, save_checkpoint, _target
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.formatting import pack_request, branch_attention_mask
from systemone_lab.data.io import read_jsonl
from systemone_lab.eval import ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dev", action="append", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--report", required=True)
    a = ap.parse_args(); device = pick_device()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, device); model.to(device).eval(); tok = LabTokenizer(ck["tokenizer"])
    if model.cfg.decision_head.get("scorer") != "cosine":
        raise ValueError("temperature calibration requires the cosine scorer")
    old = float(model.cfg.decision_head.get("temperature", 10.0))
    cosines, targets = [], []
    with torch.no_grad():
        for path in a.dev:
            for rec in read_jsonl(path):
                p = pack_request(tok, rec["state"], rec["questions"], device=device, isolate_options=model.isolated_options)
                h = model.hidden(p.input_ids[None], p.position_ids[None], model.attention_mask(p))
                for layout in p.layouts:
                    logits = model.decision_logits(h, layout.decide_position, layout.option_end_positions,
                                                   query_entity_token_ids=layout.query_entity_token_ids,
                                                   query_entity_state_positions=layout.query_entity_state_positions)
                    cosines.append(logits.float() / old); targets.append(_target(layout, rec))
    c = torch.stack(cosines); y = torch.tensor(targets, device=c.device)

    def stats(t):
        prob = F.softmax(t * c, -1); conf, pred = prob.max(-1)
        return {"nll": F.cross_entropy(t * c, y).item(),
                "ece15": ece(conf.tolist(), (pred == y).int().tolist(), 15), "accuracy": (pred == y).float().mean().item()}

    grid = [round(old * math.exp(k / 40), 4) for k in range(-80, 41)]
    best = min(grid, key=lambda t: stats(t)["nll"])
    report = {"ckpt": a.ckpt, "dev": a.dev, "n": len(targets), "old_temperature": old, "new_temperature": best,
              "before": stats(old), "after": stats(best)}
    model.cfg.decision_head = {**model.cfg.decision_head, "temperature": best}
    save_checkpoint(a.out, model, model.cfg, ck["tokenizer"], ck.get("step", 0),
                    {**ck.get("metrics", {}), "calibration": report})
    with open(a.report, "w") as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
