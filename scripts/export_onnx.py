"""Export a decision checkpoint (backbone + decision head) to ONNX for in-browser inference.

Graph inputs (one request, batch 1):
  input_ids int64[1,T], position_ids int64[1,T], mask bool[1,T,T] (True = attend),
  decide int64[1], options int64[K] (option-end positions), bind float32[T]
  (signed mean weights over the two query names' state occurrences; all zeros = no binding).
Output: logits float32[K]. Softmax is done by the caller.

With `bind` all zeros the query is normalize(q), which gives the same cosine logits as the
unbound PyTorch path, so one graph covers entity_binding and decide checkpoints.
Parity against PyTorch is measured on real packed requests and written to --report.
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


class DecisionGraph(nn.Module):
    def __init__(self, model):
        super().__init__(); self.m = model
        dh = model.cfg.decision_head
        if dh.get("scorer") != "cosine" or dh.get("query_mode", "decide") not in ("decide", "entity_binding"):
            raise ValueError("export supports cosine scorers with decide/entity_binding queries")
        if (model.cfg.arch or {}).get("type") == "looped" or model.loop_gate is not None:
            raise ValueError("export does not support looped checkpoints yet")
        self.temperature = float(dh.get("temperature", 10.0))
        self.has_bind = model.ptr_bind is not None

    def forward(self, input_ids, position_ids, mask, decide, options, bind):
        h = self.m.hidden(input_ids, position_ids, mask)[0].float()
        q = F.normalize(self.m.ptr_q(h[decide])[0], dim=-1)
        if self.has_bind:
            q = q + self.m.ptr_bind(bind @ h)
        k = F.normalize(self.m.ptr_k(h[options]), dim=-1)
        return self.temperature * (k @ F.normalize(q, dim=-1))


def graph_inputs(model, packed):
    layout = packed.layouts[0]; T = packed.input_ids.numel()
    bind = torch.zeros(T)
    first, second = layout.query_entity_state_positions or ((), ())
    if first and second:
        for group, sign in ((first, 1.0), (second, -1.0)):
            for p in group: bind[p] += sign / len(group)
    return {"input_ids": packed.input_ids[None], "position_ids": packed.position_ids[None],
            "mask": model.attention_mask(packed)[None], "decide": torch.tensor([layout.decide_position]),
            "options": torch.tensor(layout.option_end_positions), "bind": bind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--parity", required=True, help="jsonl of requests used for parity checks")
    ap.add_argument("--report", required=True); ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, "cpu"); model.eval(); tok = LabTokenizer(ck["tokenizer"])
    g = DecisionGraph(model).eval()
    recs = [json.loads(l) for l in open(a.parity, encoding="utf-8")][:a.limit]
    packs = [pack_request(tok, r["state"], r["questions"], isolate_options=model.isolated_options) for r in recs]
    example = graph_inputs(model, packs[0])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fp32 = out.with_suffix(".fp32.onnx")
    torch.onnx.export(g, tuple(example.values()), str(fp32), input_names=list(example), output_names=["logits"],
                      dynamic_axes={"input_ids": {1: "T"}, "position_ids": {1: "T"}, "mask": {1: "T", 2: "T"},
                                    "options": {0: "K"}, "bind": {0: "T"}, "logits": {0: "K"}},
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
    import onnxruntime as ort
    sessions = {"fp32": ort.InferenceSession(str(fp32)), "int8": ort.InferenceSession(str(out))}
    stats = {k: {"argmax_agree": 0, "max_abs_prob_diff": 0.0, "mean_abs_prob_diff": 0.0} for k in sessions}
    for p in packs:
        inp = graph_inputs(model, p)
        with torch.no_grad():
            ref = torch.softmax(g(*inp.values()), -1).numpy()
        feed = {k: v.numpy() for k, v in inp.items()}
        for name, s in sessions.items():
            names = {i.name for i in s.get_inputs()}  # unused inputs (e.g. bind) are pruned from the graph
            logits = s.run(None, {k: v for k, v in feed.items() if k in names})[0]; prob = np.exp(logits - logits.max()); prob /= prob.sum()
            d = np.abs(prob - ref); st = stats[name]
            st["argmax_agree"] += int(prob.argmax() == ref.argmax())
            st["max_abs_prob_diff"] = max(st["max_abs_prob_diff"], float(d.max())); st["mean_abs_prob_diff"] += float(d.mean())
    for st in stats.values():
        st["argmax_agree"] /= len(packs); st["mean_abs_prob_diff"] /= len(packs)
    report = {"ckpt": a.ckpt, "requests": len(packs), "fp32_bytes": fp32.stat().st_size, "int8_bytes": out.stat().st_size, **stats}
    Path(a.report).write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
