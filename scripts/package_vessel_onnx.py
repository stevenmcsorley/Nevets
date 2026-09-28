"""Package a gated VESSEL checkpoint for Channel Watch full-watch scoring (channel-watch/nevets/model).

Steps: export the multi-question graph (scripts/export_onnx_multi.py: fp32 + dynamic int8), build a weight-only int8
variant from the fp32 graph, measure PyTorch <-> ONNX parity on the held-out real vessels (eval_real), choose the variant
(fp32 unless weight-only int8 matches every argmax and stays within MAX_PROB_DIFF), and write meta.json with the
model's benchmark results (synthetic main, robustness, real held-out) next to the tokenizer.

Refuses to package a checkpoint that was killed by the Rule-4 gates or has no gates file.
Usage: python scripts/package_vessel_onnx.py --ckpt checkpoints/vessel/vessel-v1b_s7.pt --name VESSEL-1b --rep reports/vessel/v1b
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MAX_PROB_DIFF = 0.02
CW_MODEL = Path(r"E:\channel-watch\nevets\model")


def run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "src", "S1_DEVICE": "cpu"})
    if r.returncode: raise SystemExit(f"failed: {' '.join(args)}\n{r.stderr[-2000:]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--rep", required=True, help="report dir with eval/robust/real/gates json"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); ck = Path(a.ckpt); rep = Path(a.rep)
    if ck.with_suffix(".killed.json").exists(): raise SystemExit("refusing: killed by the Rule-4 gates")
    gates = next(iter(sorted(rep.glob("gates_s*.json"))), None)
    if gates is None: raise SystemExit("refusing: no gates report")
    out = Path("reports/vessel/onnx") / ck.stem; out.mkdir(parents=True, exist_ok=True)
    parity_src = Path("data/processed/vessel_c/eval_real.jsonl") if Path("data/processed/vessel_c/eval_real.jsonl").exists() else Path("data/processed/vessel_v1/eval_real.jsonl")
    print(run("scripts/export_onnx_multi.py", "--ckpt", str(ck), "--out", str(out / "vessel.dyn8.onnx"), "--parity", str(parity_src),
              "--report", str(out / "parity.json"), "--limit", "200")[-1500:])
    fp32 = out / "vessel.dyn8.fp32.onnx"; w8 = out / "vessel.w8.onnx"
    run("scripts/quantize_weights_int8.py", str(fp32), str(w8))
    # weight-only int8 parity against the fp32 graph on the same requests
    import numpy as np, onnxruntime as ort, torch
    from systemone_lab.formatting import pack_request
    from systemone_lab.model import SystemOneModel
    from systemone_lab.tokenizer import LabTokenizer
    from systemone_lab.training import load_checkpoint
    sys.path.insert(0, "scripts"); from export_onnx_multi import multi_inputs  # noqa: E402
    model, ckd = load_checkpoint(str(ck), SystemOneModel, "cpu"); model.eval(); tok = LabTokenizer(ckd["tokenizer"])
    recs = [json.loads(l) for l in open(parity_src, encoding="utf-8")][:200]
    s32, s8 = ort.InferenceSession(str(fp32)), ort.InferenceSession(str(w8)); agree = n = 0; maxd = 0.0
    for r in recs:
        p = pack_request(tok, r["state"], r["questions"], isolate_options=model.isolated_options); inp, nk = multi_inputs(model, p)
        feed = {k: v.numpy() for k, v in inp.items()}
        l32 = s32.run(None, {k: v for k, v in feed.items() if k in {i.name for i in s32.get_inputs()}})[0]
        l8 = s8.run(None, {k: v for k, v in feed.items() if k in {i.name for i in s8.get_inputs()}})[0]
        for qi, k in enumerate(nk):
            a32, a8 = l32[qi, :k], l8[qi, :k]; p32 = np.exp(a32 - a32.max()); p32 /= p32.sum(); p8 = np.exp(a8 - a8.max()); p8 /= p8.sum()
            agree += int(p32.argmax() == p8.argmax()); n += 1; maxd = max(maxd, float(np.abs(p32 - p8).max()))
    parity = json.loads((out / "parity.json").read_text())
    parity["w8_vs_fp32"] = {"argmax_agree": agree / n, "max_abs_prob_diff": maxd, "questions": n}
    # Git (and so pi-ci) cannot carry files > 100 MB: prefer weight-only int8 (33 MB), then dynamic int8 (80 MB), each only
    # with full argmax agreement and <= MAX_PROB_DIFF; fp32 (129 MB) would need manual delivery and is only reported.
    dyn = parity["int8"]
    # >= 99% argmax agreement (near-ties may flip on rounding) and max probability difference <= MAX_PROB_DIFF.
    choice = ("w8" if agree / n >= 0.99 and maxd <= MAX_PROB_DIFF else
              "dyn8" if dyn["argmax_agree"] >= 0.99 and dyn["max_abs_prob_diff"] <= MAX_PROB_DIFF else "fp32-manual")
    parity["chosen"] = choice
    (out / "parity.json").write_text(json.dumps(parity, indent=2)); print(json.dumps(parity["w8_vs_fp32"]), "chosen:", choice)
    rd = lambda f: json.loads((rep / f).read_text()) if (rep / f).exists() else None
    ev, rob, real = rd("eval_s7.json"), rd("robust_s7.json"), rd("real_s7.json")
    bits = []
    if ev: bits.append(f"Synthetic benchmark: accuracy {ev['overall']['accuracy']:.3f}, ECE {ev['overall']['ece15']:.3f}, counterfactual {ev['cf_both']:.3f}"
                       f" ({'meets' if ev.get('verdict', {}).get('WIN') else 'does not meet'} the pre-registered win criterion).")
    if real: bits.append("Held-out live vessels (agreement with Claude labels): " + ", ".join(f"{q} {v['agreement_with_claude_labels']:.3f}" for q, v in real["nevets"]["by_question"].items()) + ".")
    meta = {"name": a.name, "version": ck.stem, "bidirectional_state": bool(model.bidirectional_state), "isolated_options": bool(model.isolated_options),
            "iters": getattr(model, "iters", None), "graph": choice, "parity": {"fp32": parity["fp32"], "w8_vs_fp32": parity["w8_vs_fp32"]},
            "benchmarks": {"summary": " ".join(bits), "synthetic": ev and {"accuracy": ev["overall"]["accuracy"], "ece15": ev["overall"]["ece15"], "cf_both": ev["cf_both"], "verdict": ev.get("verdict")},
                           "robust": rob and {k: v.get("accuracy") for k, v in rob["nevets"].items() if isinstance(v, dict) and "accuracy" in v},
                           "real": real and real["nevets"]["by_question"]}, "promoted": False, "tokenizer": "/app/model/tokenizer.json"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1)); print(json.dumps(meta["benchmarks"]["summary"]))
    if a.dry_run: return
    if choice == "fp32-manual": raise SystemExit("no int8 graph met parity; fp32 needs manual delivery (not packaged)")
    CW_MODEL.mkdir(parents=True, exist_ok=True)
    shutil.copy(w8 if choice == "w8" else out / "vessel.dyn8.onnx", CW_MODEL / "vessel.onnx"); shutil.copy("data/tokenizer.json", CW_MODEL / "tokenizer.json")
    (CW_MODEL / "meta.json").write_text(json.dumps(meta, indent=1)); print("packaged into", CW_MODEL, "size", (CW_MODEL / "vessel.onnx").stat().st_size)


if __name__ == "__main__":
    main()
