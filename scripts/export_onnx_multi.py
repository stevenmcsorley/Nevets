"""Export a decision checkpoint to a MULTI-QUESTION ONNX graph: one backbone pass scores every question of a request.

For server-side full-watch scoring (Channel Watch), where running the backbone once per question (export_onnx.py) is
3x the work. Graph inputs (one request):
  input_ids int64[1,T], position_ids int64[1,T], mask bool[1,T,T],
  decide int64[Q] (decide positions), options int64[Q,K] (option-end positions, padded with any valid position),
  bind float32[Q,T] (per-question signed binding weights; zeros = no binding)
Output: logits float32[Q,K]; the caller ignores padded option slots and applies softmax per question.
Parity: probabilities from the ONNX graph (fp32 and int8) are compared with the PyTorch per-question DecisionGraph on real
packed requests (--report).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from systemone_lab.formatting import pack_request
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint

import sys
sys.path.insert(0, str(Path(__file__).parent))
from export_onnx import DecisionGraph, graph_inputs  # noqa: E402


class MultiDecisionGraph(nn.Module):
    def __init__(self, model):
        super().__init__(); self.single = DecisionGraph(model); self.m = model

    def forward(self, input_ids, position_ids, mask, decide, options, bind):
        h = self.m.hidden(input_ids, position_ids, mask)[0].float()                  # [T, d]
        q = F.normalize(self.m.ptr_q(h[decide]), dim=-1)                               # [Q, p]
        if self.single.has_bind:
            q = q + self.m.ptr_bind(bind @ h)                                          # [Q, p]
        k = F.normalize(self.m.ptr_k(h[options]), dim=-1)                              # [Q, K, p]
        return self.single.temperature * torch.einsum("qkp,qp->qk", k, F.normalize(q, dim=-1))


def multi_inputs(model, packed):
    singles = [graph_inputs(model, packed) if i == 0 else None for i in range(1)]
    T = packed.input_ids.numel(); layouts = packed.layouts; K = max(len(l.option_end_positions) for l in layouts)
    decide = torch.tensor([l.decide_position for l in layouts])
    options = torch.tensor([list(l.option_end_positions) + [l.option_end_positions[0]] * (K - len(l.option_end_positions)) for l in layouts])
    bind = torch.zeros(len(layouts), T)
    for i, l in enumerate(layouts):
        first, second = l.query_entity_state_positions or ((), ())
        if first and second:
            for group, sign in ((first, 1.0), (second, -1.0)):
                for p in group: bind[i, p] += sign / len(group)
    base = singles[0]
    return {"input_ids": base["input_ids"], "position_ids": base["position_ids"], "mask": base["mask"],
            "decide": decide, "options": options, "bind": bind}, [len(l.option_end_positions) for l in layouts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--parity", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int, default=100); ap.add_argument("--iters", type=int)
    a = ap.parse_args()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, "cpu"); model.eval(); tok = LabTokenizer(ck["tokenizer"])
    if a.iters: model.iters = a.iters
    g = MultiDecisionGraph(model).eval()
    recs = [json.loads(l) for l in open(a.parity, encoding="utf-8")][:a.limit]
    packs = [pack_request(tok, r["state"], r["questions"], isolate_options=model.isolated_options) for r in recs]
    ex, _ = multi_inputs(model, packs[0])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); fp32 = out.with_suffix(".fp32.onnx")
    torch.onnx.export(g, tuple(ex.values()), str(fp32), input_names=list(ex), output_names=["logits"],
                      dynamic_axes={"input_ids": {1: "T"}, "position_ids": {1: "T"}, "mask": {1: "T", 2: "T"}, "decide": {0: "Q"},
                                    "options": {0: "Q", 1: "K"}, "bind": {0: "Q", 1: "T"}, "logits": {0: "Q", 1: "K"}},
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
    import onnxruntime as ort
    sessions = {"fp32": ort.InferenceSession(str(fp32)), "int8": ort.InferenceSession(str(out))}
    stats = {k: {"argmax_agree": 0, "max_abs_prob_diff": 0.0, "mean_abs_prob_diff": 0.0, "questions": 0} for k in sessions}
    for p in packs:
        inp, nk = multi_inputs(model, p)
        refs = []
        with torch.no_grad():  # reference: the per-question PyTorch graph used by the demos
            for qi, l in enumerate(p.layouts):
                single = graph_inputs(model, p) if qi == 0 else None
                s_inp = {"input_ids": inp["input_ids"], "position_ids": inp["position_ids"], "mask": inp["mask"],
                         "decide": torch.tensor([l.decide_position]), "options": torch.tensor(l.option_end_positions), "bind": inp["bind"][qi]}
                refs.append(torch.softmax(g.single(*s_inp.values()), -1).numpy())
        feed = {k: v.numpy() for k, v in inp.items()}
        for name, s in sessions.items():
            names = {i.name for i in s.get_inputs()}
            logits = s.run(None, {k: v for k, v in feed.items() if k in names})[0]
            for qi, ref in enumerate(refs):
                lg = logits[qi, :nk[qi]]; prob = np.exp(lg - lg.max()); prob /= prob.sum(); d = np.abs(prob - ref); st = stats[name]
                st["argmax_agree"] += int(prob.argmax() == ref.argmax()); st["questions"] += 1
                st["max_abs_prob_diff"] = max(st["max_abs_prob_diff"], float(d.max())); st["mean_abs_prob_diff"] += float(d.mean())
    for st in stats.values():
        st["argmax_agree"] /= st["questions"]; st["mean_abs_prob_diff"] /= st["questions"]
    report = {"ckpt": a.ckpt, "iters": getattr(model, "iters", None), "requests": len(packs), "fp32_bytes": fp32.stat().st_size,
              "int8_bytes": out.stat().st_size, **stats}
    Path(a.report).write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
