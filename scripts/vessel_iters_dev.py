"""How many loop passes does the deployed VESSEL model need? Speed vs answer change, chosen on a DEV set (never eval_real).

Dev set: Channel Watch Claude labels made after the model's live training data was frozen (label ids not in
vessel_real_v1/train.jsonl), from vessels outside the locked eval_real split (sha1(mmsi)[0] % 5 != 0), minus the recorded
mislabels (ml/label_exclusions.json). For K passes it reports agreement with Claude labels, agreement with the deployed K,
and CPU time per vessel (2 threads, as on the Pi). eval_real is scored once, only for a K chosen here (--final).
Usage: python scripts/vessel_iters_dev.py --ckpt checkpoints/vessel/vessel-v1c_s7.pt --labels E:/channel-watch/data/labels/labels-all.jsonl
"""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from systemone_lab.gates import predict_batch
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint

in_eval = lambda m: hashlib.sha1(str(int(m)).encode()).digest()[0] % 5 == 0


def dev_set(labels, train, excl):
    seen = {json.loads(l)["meta"].get("label_id") for l in open(train, encoding="utf-8") if l.strip()}
    ex = json.loads(Path(excl).read_text())["by_mmsi_before_label_id"]
    out = []
    for l in open(labels, encoding="utf-8"):
        r = json.loads(l); m = r["meta"]; lid = m.get("label_id")
        if lid in seen or in_eval(m["mmsi"]): continue
        if str(m["mmsi"]) in ex and lid < ex[str(m["mmsi"])]["before_label_id"]: continue
        out.append(r)
    return out


def run(model, tok, recs, K):
    model.iters = K; t0 = time.perf_counter(); preds = []
    for i in range(0, len(recs), 8): preds += predict_batch(model, tok, recs[i:i + 8], "cpu")
    return preds, (time.perf_counter() - t0) / len(recs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--train", default="data/processed/vessel_real_v1/train.jsonl"); ap.add_argument("--excl", default=r"E:\channel-watch\ml\label_exclusions.json")
    ap.add_argument("--ks", default="4,3,2,1"); ap.add_argument("--final", type=int, help="score eval_real once at this K and the deployed K")
    ap.add_argument("--out", default="reports/vessel/iters_dev.json")
    a = ap.parse_args(); torch.set_num_threads(2)
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, "cpu"); model.eval(); tok = LabTokenizer(ck["tokenizer"]); base_k = model.iters
    recs = dev_set(a.labels, a.train, a.excl) if not a.final else [json.loads(l) for l in open("data/processed/vessel_c/eval_real.jsonl", encoding="utf-8")]
    ks = [base_k, a.final] if a.final else [int(k) for k in a.ks.split(",")]
    res, ref = {"deployed_iters": base_k, "records": len(recs), "set": "eval_real (final, once)" if a.final else "post-freeze dev (no eval_real vessels)", "by_k": {}}, None
    with torch.no_grad():
        for K in ks:
            preds, spv = run(model, tok, recs, K)
            ans = [{q: max(d, key=d.get) for q, d in p.items()} for p in preds]
            if K == base_k: ref = ans
            agree = {q: sum(x[q] == str(r["labels"][q]).lower() for x, r in zip(ans, recs)) / len(recs) for q in ans[0]}
            same = {q: sum(x[q] == y[q] for x, y in zip(ans, ref)) / len(recs) for q in ans[0]} if ref else None
            res["by_k"][K] = {"agreement_with_claude_labels": agree, "same_answer_as_deployed": same, "cpu_s_per_vessel_2threads": round(spv, 4)}
            print(K, json.dumps(res["by_k"][K]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1)); print("written", a.out)


if __name__ == "__main__":
    main()
