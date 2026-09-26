"""PT streaming corpus pipeline: raw parquet -> (dedup, decontaminate, tokenise) -> uint16 shards, one pass.

Disk stays flat: cleaned text is never persisted, and with --delete-raw each downloaded parquet is removed
once its tokens are committed. Resumable: `data/pt/stream_manifest.json` records processed inputs, shard
hashes and counts; shard numbering continues after existing shards. Dedup is corpus-wide through a
persistent SHA-1 set (`data/pt/seen_sha1.bin`), seeded from already-cleaned documents. Decontamination is
identical to build_corpus.py (13-gram + short-string checks against every registered eval/locked text).
"""
import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "scripts/pt")
import build_corpus as bc  # noqa: E402

PT = Path("data/pt"); MAN = PT / "stream_manifest.json"; SEEN = PT / "seen_sha1.bin"


def set_pt_dir(d):
    global PT, MAN, SEEN
    PT = Path(d); MAN = PT / "stream_manifest.json"; SEEN = PT / "seen_sha1.bin"
TOKENIZER_SHA256 = "229a91f7d1befea96cd93af911c79cd62765101610470f4866a7d2477c9e1126"


def load_seen():
    seen = set()
    if SEEN.exists():
        b = SEEN.read_bytes(); seen = {b[i:i + 20] for i in range(0, len(b), 20)}
    else:  # seed from documents already cleaned by PT-1
        for f in sorted(glob.glob(str(PT / "clean" / "*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                seen.add(hashlib.sha1(bc.norm(json.loads(line)["text"]).encode()).digest())
        SEEN.write_bytes(b"".join(seen))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"); ap.add_argument("--subset", default="sample/10BT")
    ap.add_argument("--files", default="2:14", help="slice of sorted parquet files to process, e.g. 2:14")
    ap.add_argument("--local-parquet", nargs="*", help="process these local files instead of downloading (tests)")
    ap.add_argument("--delete-raw", action="store_true"); ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--tokenizer", default="tokenizers/pt_32k.json"); ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=None); ap.add_argument("--pt-dir", default="data/pt")
    a = ap.parse_args(); set_pt_dir(a.pt_dir); a.out = a.out or str(PT / "shards"); PT.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer
    assert hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest() == TOKENIZER_SHA256 or a.local_parquet, "tokenizer hash mismatch"
    tok = Tokenizer.from_file(a.tokenizer); eos = tok.token_to_id("<eos>")
    grams, short, n_eval = bc.eval_ngrams(); seen = load_seen(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads(MAN.read_text()) if MAN.exists() else {"tokenizer": a.tokenizer, "tokenizer_sha256": TOKENIZER_SHA256,
                                                            "revision": a.revision, "inputs_done": [], "shards": [], "stats": {}}
    existing = sorted(glob.glob(str(out / "train_*.bin"))); next_train = len(existing)
    next_val = len(glob.glob(str(out / "val_*.bin"))); rng = random.Random(a.seed + len(man["inputs_done"]))
    st = man["stats"]
    for k in ("docs_in", "dup_removed", "contam_candidates", "contam_removed", "docs_out", "tokens_train", "tokens_val"): st.setdefault(k, 0)

    if a.local_parquet: inputs = [(p, p) for p in a.local_parquet]
    else:
        from huggingface_hub import HfApi, hf_hub_download
        info = {f.path: f for f in HfApi().list_repo_tree("HuggingFaceFW/fineweb-edu", repo_type="dataset", revision=a.revision, path_in_repo=a.subset)}
        lo, hi = map(int, a.files.split(":")); names = sorted(p for p in info if p.endswith(".parquet"))[lo:hi]
        inputs = [(n, None) for n in names]
    buf, val = [], []
    def flush(tokens, kind):
        nonlocal next_train, next_val
        name = f"train_{next_train:04d}.bin" if kind == "train" else f"val_{next_val:04d}.bin"
        arr = np.asarray(tokens, dtype=np.uint16); p = out / name; arr.tofile(p)
        man["shards"].append({"file": name, "tokens": int(arr.size), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        if kind == "train": next_train += 1
        else: next_val += 1
    for name, local in inputs:
        if name in man["inputs_done"]: continue
        if local is None:
            local = hf_hub_download("HuggingFaceFW/fineweb-edu", name, repo_type="dataset", revision=a.revision, local_dir=str(PT / "raw"))
            h = hashlib.sha256()
            with open(local, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
            assert info[name].lfs is None or h.hexdigest() == info[name].lfs.sha256, f"checksum mismatch {name}"
        fh = open(local, "rb"); pf = pq.ParquetFile(fh); new_seen = []; batch = []
        def drain():
            for enc in tok.encode_batch(batch):
                (val if rng.random() < 0.005 else buf).extend(enc.ids + [eos])
            batch.clear()
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                st["docs_in"] += 1; h = hashlib.sha1(bc.norm(text).encode()).digest()
                if h in seen: st["dup_removed"] += 1; continue
                seen.add(h); new_seen.append(h); low = text.lower()
                if any(m in low for m in bc.MARKERS):
                    st["contam_candidates"] += 1; w = bc.words(text); joined = " ".join(w)
                    if any(hash(" ".join(w[i:i + bc.N])) in grams for i in range(len(w) - bc.N + 1)) or any(s in joined for s in short):
                        st["contam_removed"] += 1; continue
                st["docs_out"] += 1; batch.append(text)
                if len(batch) >= 2000: drain()
            while len(buf) >= a.shard_tokens:
                flush(buf[:a.shard_tokens], "train"); st["tokens_train"] += a.shard_tokens; buf = buf[a.shard_tokens:]
        drain(); pf = None; fh.close()  # release the file handle (Windows cannot delete an open file)
        # Crash-safe: commit every token of this input (partial shards allowed) BEFORE marking it done.
        if buf: flush(buf, "train"); st["tokens_train"] += len(buf); buf = []
        if val: flush(val, "val"); st["tokens_val"] += len(val); val = []
        with open(SEEN, "ab") as f: f.write(b"".join(new_seen))
        man["inputs_done"].append(name); MAN.write_text(json.dumps(man, indent=2))
        if a.delete_raw and not a.local_parquet: Path(local).unlink()
        print(json.dumps({"input": name, **st}), flush=True)
    if buf: flush(buf, "train"); st["tokens_train"] += len(buf)
    if val: flush(val, "val"); st["tokens_val"] += len(val)
    man["eval_texts"] = n_eval; MAN.write_text(json.dumps(man, indent=2)); print(json.dumps(st))


if __name__ == "__main__":
    main()
