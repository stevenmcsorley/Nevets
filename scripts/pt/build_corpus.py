"""PT-1: clean, decontaminate and tokenise the FineWeb-Edu slice into uint16 shards with a new tokenizer.

Stages (run in order; each writes to data/pt/ and never touches the legacy data/tokenizer.json):
  clean      parquet -> exact-dedup (normalised-text SHA-1) -> decontaminate -> data/pt/clean/*.jsonl
  tokenizer  train a 32k byte-level BPE (full byte alphabet, single-digit splitting) on a corpus sample
             plus rendered structured formats -> data/pt/tokenizer_32k.json (+ sha256)
  shards     tokenise clean text (doc + <eos>) into 100M-token uint16 shards; 0.5% of docs -> val shard
Decontamination: a document is dropped if it shares any word 13-gram with any registered eval/locked
state or question (systemone_lab.eval_registry) or chess eval position. A cheap marker prefilter picks
candidates; only candidates get the full n-gram check.
"""
import argparse
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "scripts")
PT = Path("data/pt"); SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>", "<state>", "</state>", "<q>", "</q>", "<opt>",
                                 "</opt>", "<decide>", "<noul>", "<choice>", "<score>"]
MARKERS = ("obj_", "spatial relation of", "happens with probability", "occupies the same position",
           "is upper-left of", "is lower-right of", "sits north of", "diagnostic reads positive", "which box was")
N = 13
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
words = lambda s: re.findall(r"[a-z0-9_]+", s.lower())


def eval_ngrams():
    from systemone_lab.eval_registry import EVAL_JSONL, EVAL_PAIRS, eval_states
    texts = set(eval_states())
    for rel in EVAL_JSONL:
        p = Path(rel)
        if p.exists():
            for line in open(p, encoding="utf-8"):
                for q in json.loads(line).get("questions", {}).values(): texts.add(q.get("instructions", ""))
    for p in ("data/processed/chess/eval_2013-02.jsonl",):
        if Path(p).exists():
            for line in open(p, encoding="utf-8"): texts.add(json.loads(line)["state"])
    grams = set(); short = set()
    for t in texts:
        w = words(t)
        if len(w) >= N: grams.update(hash(" ".join(w[i:i + N])) for i in range(len(w) - N + 1))
        elif len(w) >= 6: short.add(" ".join(w))  # short eval strings: exact word-sequence containment
    return grams, short, len(texts)


def stage_clean(a):
    grams, short, n_eval = eval_ngrams(); seen = set()
    stats = {"docs_in": 0, "dup_removed": 0, "contam_candidates": 0, "contam_removed": 0, "docs_out": 0, "eval_texts": n_eval}
    out_dir = PT / "clean"; out_dir.mkdir(parents=True, exist_ok=True); examples = []
    for path in sorted(glob.glob("data/pt/raw/**/*.parquet", recursive=True)):
        pf = pq.ParquetFile(path); out = open(out_dir / (Path(path).stem + ".jsonl"), "w", encoding="utf-8")
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                stats["docs_in"] += 1
                h = hashlib.sha1(norm(text).encode()).digest()
                if h in seen: stats["dup_removed"] += 1; continue
                seen.add(h); low = text.lower()
                if any(m in low for m in MARKERS):
                    stats["contam_candidates"] += 1; w = words(text); joined = " ".join(w)
                    hit = any(hash(" ".join(w[i:i + N])) in grams for i in range(len(w) - N + 1)) or any(s in joined for s in short)
                    if hit:
                        stats["contam_removed"] += 1
                        if len(examples) < 5: examples.append(text[:300])
                        continue
                out.write(json.dumps({"text": text}) + "\n"); stats["docs_out"] += 1
        out.close(); print(json.dumps({"file": path, **stats}), flush=True)
    stats["removed_examples"] = examples
    (PT / "clean_report.json").write_text(json.dumps(stats, indent=2))


def structured_samples(n, seed):
    """Rendered structured text so JSON/table/kv/CSV delimiters get sensible merges (fresh seeds only)."""
    from systemone_lab.worlds import DOMAINS, FORMATS
    rng = random.Random(seed); out = []
    for i in range(n):
        dom = rng.choice(list(DOMAINS)); fmt = rng.choice(FORMATS)
        rec, _ = DOMAINS[dom](rng, fmt, f"tok-{i}"); out.append(rec["state"])
        if i % 4 == 0:  # CSV rows and nested JSON records of generic business-like data
            rows = [{"id": rng.randint(1, 99999), "status": rng.choice(["open", "closed", "pending"]),
                     "amount": round(rng.uniform(0, 5000), 2), "region": rng.choice(["EU", "US", "APAC"])} for _ in range(rng.randint(2, 6))]
            out.append("id,status,amount,region\n" + "\n".join(f"{r['id']},{r['status']},{r['amount']},{r['region']}" for r in rows))
            out.append(json.dumps({"records": rows, "total": len(rows)}, indent=rng.choice([None, 2])))
    return out


def stage_tokenizer(a):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    rng = random.Random(a.seed); texts = []
    files = sorted(glob.glob(str(PT / "clean" / "*.jsonl"))); budget = a.tok_chars
    for f in files:
        for line in open(f, encoding="utf-8"):
            if rng.random() < a.tok_sample_rate: texts.append(json.loads(line)["text"])
            if sum(map(len, texts[-1:])) and len(texts) % 5000 == 0 and sum(map(len, texts)) > budget: break
    struct = structured_samples(max(1, len(texts) // 9), a.seed + 1)  # ~10% structured
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=SPECIAL, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts + struct, trainer=trainer)
    path = PT / f"tokenizer_{a.vocab // 1000}k.json"; tok.save(str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rep = {"path": str(path), "sha256": sha, "vocab": tok.get_vocab_size(), "corpus_docs": len(texts), "corpus_chars": sum(map(len, texts)),
           "structured_samples": len(struct), "digits": "individual", "byte_alphabet": "full (no <unk> for any byte)"}
    probe = ['{"facts":[{"subject":"a"}]}', "id,status,amount\n17,open,3.50", "With rain, heat happens with probability 3/20."]
    rep["probe"] = {p: len(tok.encode(p).ids) for p in probe}; rep["unk_in_probe"] = any(3 in tok.encode(p).ids for p in probe)
    (PT / "tokenizer_report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


def stage_shards(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(PT / f"tokenizer_{a.vocab // 1000}k.json")); eos = tok.token_to_id("<eos>")
    assert tok.get_vocab_size() < 65536
    sd = PT / "shards"; sd.mkdir(parents=True, exist_ok=True); rng = random.Random(a.seed)
    buf, val, shard_i, manifest = [], [], 0, {"tokenizer": str(PT / f"tokenizer_{a.vocab // 1000}k.json"), "shards": []}
    def flush(tokens, name):
        arr = np.asarray(tokens, dtype=np.uint16); p = sd / name; arr.tofile(p)
        manifest["shards"].append({"file": name, "tokens": int(arr.size), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        print(json.dumps(manifest["shards"][-1]), flush=True)
    batch = []
    def drain():
        nonlocal shard_i, buf
        for enc in tok.encode_batch(batch):
            target = val if rng.random() < 0.005 else buf
            target.extend(enc.ids); target.append(eos)
        batch.clear()
        while len(buf) >= a.shard_tokens:
            flush(buf[:a.shard_tokens], f"train_{shard_i:04d}.bin"); buf = buf[a.shard_tokens:]; shard_i += 1
    for f in sorted(glob.glob(str(PT / "clean" / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            batch.append(json.loads(line)["text"])
            if len(batch) >= 2000: drain()
    drain()
    if buf: flush(buf, f"train_{shard_i:04d}.bin")
    flush(val, "val_0000.bin")
    manifest["train_tokens"] = sum(s["tokens"] for s in manifest["shards"] if s["file"].startswith("train"))
    manifest["val_tokens"] = sum(s["tokens"] for s in manifest["shards"] if s["file"].startswith("val"))
    (PT / "shards_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["clean", "tokenizer", "shards"])
    ap.add_argument("--vocab", type=int, default=32000); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tok-sample-rate", type=float, default=0.12); ap.add_argument("--tok-chars", type=int, default=400_000_000)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    a = ap.parse_args(); {"clean": stage_clean, "tokenizer": stage_tokenizer, "shards": stage_shards}[a.stage](a)
