"""Search int8 quantization settings for the Channel Watch VESSEL graph that keep parity with fp32.

Package parity rule: argmax agreement >= 99% and max probability difference <= 0.02 against the fp32 ONNX graph, on the
held-out real vessels (vessel_c/eval_real). Variants: per-tensor vs per-channel weights, MatMul-only vs MatMul+Gather, and
excluding the most quantization-sensitive MatMuls (ranked by single-node error on a small calibration subset).
Usage: python scripts/quant_search.py --dir reports/vessel/onnx/vessel-v1c_s7 --ckpt checkpoints/vessel/vessel-v1c_s7.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
sys.path.insert(0, "scripts")
from export_onnx_multi import multi_inputs  # noqa: E402
from systemone_lab.formatting import pack_request
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint


def feeds(ckpt, path, limit):
    model, ck = load_checkpoint(ckpt, SystemOneModel, "cpu"); model.eval(); tok = LabTokenizer(ck["tokenizer"])
    out = []
    for line in list(open(path, encoding="utf-8"))[:limit]:
        r = json.loads(line); p = pack_request(tok, r["state"], r["questions"], isolate_options=model.isolated_options)
        inp, nk = multi_inputs(model, p); out.append(({k: v.numpy() for k, v in inp.items()}, nk))
    return out


def probs(sess, data):
    names = {i.name for i in sess.get_inputs()}; res = []
    for feed, nk in data:
        lg = sess.run(None, {k: v for k, v in feed.items() if k in names})[0]
        for qi, k in enumerate(nk):
            a = lg[qi, :k]; e = np.exp(a - a.max()); res.append(e / e.sum())
    return res


def compare(ref, got):
    ag = np.mean([r.argmax() == g.argmax() for r, g in zip(ref, got)]); md = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    return {"argmax_agree": float(ag), "max_abs_prob_diff": md, "passes": bool(ag >= 0.99 and md <= 0.02)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed/vessel_c/eval_real.jsonl"); ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--calib", type=int, default=24); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); d = Path(a.dir); fp = d / "vessel.folded.fp32.onnx"
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    sess = lambda p: ort.InferenceSession(str(p), so)
    data = feeds(a.ckpt, a.data, a.limit); ref = probs(sess(fp), data)
    g = onnx.load(str(fp)); inits = {i.name for i in g.graph.initializer}
    mm = [n.name for n in g.graph.node if n.op_type == "MatMul" and any(x in inits for x in n.input)]
    report = {"weight_matmuls": len(mm), "variants": {}}
    def variant(name, **kw):
        out = d / f"q_{name}.onnx"; quantize_dynamic(str(fp), str(out), weight_type=QuantType.QInt8, **kw)
        r = compare(ref, probs(sess(out), data)); r["bytes"] = out.stat().st_size; report["variants"][name] = r
        print(name, json.dumps(r), flush=True); return r
    variant("pertensor_mm", op_types_to_quantize=["MatMul"])
    variant("perchannel_mm", op_types_to_quantize=["MatMul"], per_channel=True)
    variant("perchannel_mm_gather", op_types_to_quantize=["MatMul", "Gather"], per_channel=True)
    # sensitivity: quantize one MatMul at a time on a calibration subset, rank by max probability error
    cal = data[:a.calib]; cref = ref[:sum(len(nk) for _, nk in cal)]; sens = []
    for i, node in enumerate(mm):
        out = d / "q_single.onnx"
        quantize_dynamic(str(fp), str(out), weight_type=QuantType.QInt8, per_channel=True, nodes_to_quantize=[node])
        got = probs(sess(out), cal); sens.append((max(float(np.abs(r - x).max()) for r, x in zip(cref, got)), node))
    sens.sort(reverse=True); report["most_sensitive"] = sens[:20]
    for k in (2, 4, 8, 16, 32):
        keep = [n for _, n in sens[k:]]
        r = variant(f"perchannel_excl{k}", op_types_to_quantize=["MatMul"], per_channel=True, nodes_to_quantize=keep)
        r["excluded"] = [n for _, n in sens[:k]]
        if r["passes"]: break
    (d / "quant_search.json").write_text(json.dumps(report, indent=1)); print("written", d / "quant_search.json")


if __name__ == "__main__":
    main()
