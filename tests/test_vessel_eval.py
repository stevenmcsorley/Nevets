import json
import os
import subprocess
import sys

from systemone_lab.config import ModelConfig
from systemone_lab.model import SystemOneModel
from systemone_lab.training import save_checkpoint


def _rec(i, act, dark, pair=None, variant=None):
    r = {"id": f"v{i}", "state": f"Track {i}: speed 3.1 kn, 14 turns per hour, gap {i} min.",
         "questions": {"activity": {"type": "choice", "instructions": "What is the vessel doing?",
                                    "criteria": {k: k for k in ("transit", "fishing", "loitering", "anchored", "unusual")}},
                       "went_dark_suspicious": {"type": "noul", "instructions": "Is the AIS gap suspicious?"}},
         "labels": {"activity": act, "went_dark_suspicious": dark}, "meta": {"domain": "vessel"}}
    if pair: r["meta"].update(pair_id=pair, variant=variant, cf_question="went_dark_suspicious")
    return r


def test_eval_vessel_contract_and_verdict(tmp_path):
    d = tmp_path / "vessel_v1"; d.mkdir()
    (d / "eval_synthetic.jsonl").write_text("".join(json.dumps(_rec(i, "fishing", i % 2 == 0)) + "\n" for i in range(6)))
    cf = [_rec(10, "transit", False, "p1", "base"), _rec(11, "transit", True, "p1", "counterfactual")]
    (d / "counterfactual.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cf))
    (d / "baselines.json").write_text(json.dumps({"rules": {"accuracy": 0.99, "ece15": 0.01, "cf_both": 1.0},
                                                   "lgbm": {"accuracy": 0.98, "ece15": 0.02, "cf_both": 1.0}}))
    cfg = ModelConfig(vocab_size=16000, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, pointer_dim=16,
                      decision_head={"scorer": "cosine", "temperature": 10.0})
    save_checkpoint(tmp_path / "m.pt", SystemOneModel(cfg), cfg, "data/tokenizer.json", 0, {})
    out = tmp_path / "e.json"
    subprocess.run([sys.executable, "scripts/eval_vessel.py", "--ckpt", str(tmp_path / "m.pt"), "--dir", str(d), "--out", str(out)],
                   check=True, env={**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": ""}, capture_output=True)
    res = json.loads(out.read_text())
    assert set(res["by_question"]) == {"activity", "went_dark_suspicious"} and res["overall"]["n"] == 12
    assert res["cf_pairs"] == 1 and res["verdict"]["WIN"] is False  # a random model cannot beat near-perfect baselines
