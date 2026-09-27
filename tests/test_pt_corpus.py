import json
import re
import sys

sys.path.insert(0, "scripts/pt")


def test_decontamination_catches_eval_text_and_passes_web_text():
    import build_corpus as bc
    grams, short, n = bc.eval_ngrams()
    assert n > 1000 and grams
    long_state = next(json.loads(l)["state"] for l in open("reports/chain/eval_hops.jsonl", encoding="utf-8")
                      if len(bc.words(json.loads(l)["state"])) >= 20)
    doc = "Here is a puzzle from a forum. " + long_state + " Can anyone solve it?"
    w = bc.words(doc)
    assert any(hash(" ".join(w[i:i + bc.N])) in grams for i in range(len(w) - bc.N + 1))
    web = ("Photosynthesis converts light energy into chemical energy stored in glucose. Plants, algae and some "
           "bacteria use chlorophyll to capture sunlight, releasing oxygen as a by-product of splitting water.")
    w = bc.words(web)
    assert not any(hash(" ".join(w[i:i + bc.N])) in grams for i in range(len(w) - bc.N + 1))
    assert not any(s in " ".join(w) for s in short)


def test_pretraining_structured_slice_excludes_p3_heldout_formats():
    import build_corpus as bc
    assert set(bc.P3_HELDOUT_FORMATS) == {"table", "symbolic"}
    texts = bc.structured_samples(300, 7, for_pretraining=True)
    assert not any("subject | relation | object" in t for t in texts)


def test_reverse_decontam_flags_question_and_passage_copies(tmp_path):
    import subprocess, sys
    ev = tmp_path / "ev.jsonl"; corp = tmp_path / "corp"; corp.mkdir()
    passage = ("The mitochondrion is a double membrane bound organelle found in most eukaryotic organisms and it "
               "generates most of the supply of adenosine triphosphate used as a source of chemical energy")
    recs = [{"id": "copied", "state": passage, "questions": {"q": {"type": "noul", "instructions": "Is the mitochondrion found in most eukaryotic organisms?"}}},
            {"id": "clean", "state": "Zorblat quenched the vivid lamp near seventeen purple otters by the river bend today",
             "questions": {"q": {"type": "noul", "instructions": "Did zorblat quench anything near the purple otters?"}}}]
    ev.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (corp / "a.jsonl").write_text(json.dumps({"text": "Biology notes. " + passage + ". Question: is the mitochondrion found in most eukaryotic organisms?"}) + "\n", encoding="utf-8")
    out = tmp_path / "rep.json"
    subprocess.run([sys.executable, "scripts/pt/reverse_decontam.py", "--eval", str(ev), "--corpus", str(corp / "*.jsonl"), "--out", str(out)], check=True, capture_output=True)
    rep = json.loads(out.read_text())
    assert rep["contaminated_items"] == 1 and "copied" in rep["items"] and "clean" not in rep["items"]
    assert rep["items"]["copied"]["question"] >= 1 and rep["items"]["copied"]["ngram13"] >= 1


def test_stream_pipeline_dedups_decontaminates_and_writes_shards(tmp_path):
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq, subprocess, sys
    from pathlib import Path
    long_state = next(json.loads(l)["state"] for l in open("reports/chain/eval_hops.jsonl", encoding="utf-8")
                      if len(re.findall(r"[a-z0-9_]+", json.loads(l)["state"].lower())) >= 20)
    texts = ["Photosynthesis converts light into chemical energy in plants."] * 3 +             ["Forum puzzle: " + long_state, "Rivers carve valleys over millions of years through erosion."]
    src = tmp_path / "in.parquet"; pq.write_table(pa.table({"text": texts}), src)
    pt = tmp_path / "pt"
    res = subprocess.run([sys.executable, "scripts/pt/stream_corpus.py", "--local-parquet", str(src), "--pt-dir", str(pt),
                          "--shard-tokens", "1000000"], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert res.returncode == 0, res.stderr[-800:]
    stats = json.loads(res.stdout.strip().splitlines()[-1])
    assert stats["docs_in"] == 5 and stats["dup_removed"] == 2 and stats["contam_removed"] == 1 and stats["docs_out"] == 2
    shards = list((pt / "shards").glob("*.bin")); tokens = sum(np.fromfile(f, dtype=np.uint16).size for f in shards)
    assert tokens == stats["tokens_train"] + stats["tokens_val"] > 0
    man = json.loads((pt / "stream_manifest.json").read_text()); assert man["inputs_done"] == [str(src)]
    again = subprocess.run([sys.executable, "scripts/pt/stream_corpus.py", "--local-parquet", str(src), "--pt-dir", str(pt)],
                           capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert json.loads(again.stdout.strip().splitlines()[-1])["docs_in"] == 5  # resumable: processed inputs are skipped


def test_train_pt_smoke_same_windows_checkpoints_and_resume(tmp_path):
    import subprocess, sys, json, os
    import numpy as np
    sys.path.insert(0, "scripts/pt")
    import train_pt
    rng = np.random.default_rng(0)
    for i in range(2): rng.integers(14, 300, 5000, dtype=np.uint16).tofile(tmp_path / f"train_{i:04d}.bin")
    rng.integers(14, 300, 3000, dtype=np.uint16).tofile(tmp_path / "val_0000.bin")
    maps, idx = train_pt.windows(sorted(str(p) for p in tmp_path.glob("train_*.bin")), 63, 6000)
    assert len(maps) == 2 and all(o + 64 <= len(maps[s]) for s, o in idx)  # windows never cross a shard
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text("name: tiny\nvocab_size: 32000\nd_model: 32\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nd_ff: 64\npointer_dim: 16\n"
                   "max_seq_len: 128\nrope_theta: 10000.0\ndropout: 0.0\ndecision_head:\n  scorer: cosine\n  temperature: 10.0\n")
    cmd = [sys.executable, "scripts/pt/train_pt.py", "--config", str(cfg), "--shards", str(tmp_path / "train_*.bin"),
           "--val", str(tmp_path / "val_0000.bin"), "--tokens", "4096", "--seq", "63", "--global-batch", "4", "--micro", "2",
           "--lr", "1e-3", "--ckpt-every", "2048", "--val-tokens", "512", "--no-compile", "--hold-file", "", "--out", str(tmp_path / "run")]
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run(cmd + ["--max-steps", "10"], check=True, env=env, capture_output=True)
    assert (tmp_path / "run/tok_0M.pt").exists() and (tmp_path / "run/resume.pt").exists()
    r = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    assert '"resumed_at_step": 8' in r.stdout  # resumed from the 2048-token checkpoint (8 steps x 256 tokens)
    from systemone_lab.model import SystemOneModel
    from systemone_lab.training import load_checkpoint
    m, ck = load_checkpoint(str(tmp_path / "run/tok_0M.pt"), SystemOneModel)  # 4096 tokens -> named tok_0M too
    vals = [json.loads(l) for l in open(tmp_path / "run/train_log.jsonl") if '"val_loss"' in l]
    assert [v["step"] for v in vals][-1] == 16 and all(np.isfinite(v["val_loss"]) for v in vals)
