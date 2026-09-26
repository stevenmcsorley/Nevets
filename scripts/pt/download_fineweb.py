"""PT-1: download FineWeb-Edu parquet shards at a pinned revision and verify their SHA-256 (LFS oid).

FineWeb-Edu (HuggingFaceFW/fineweb-edu) is licensed ODC-By v1.0 (attribution required).
"""
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb-edu"
ap = argparse.ArgumentParser()
ap.add_argument("--revision", default="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
ap.add_argument("--subset", default="sample/10BT"); ap.add_argument("--files", type=int, default=2)
ap.add_argument("--out", default="data/pt/raw")
a = ap.parse_args()
info = {f.path: f for f in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=a.revision, path_in_repo=a.subset)}
paths = sorted(p for p in info if p.endswith(".parquet"))[:a.files]
manifest = {"repo": REPO, "revision": a.revision, "license": "ODC-By v1.0", "files": []}
for p in paths:
    local = hf_hub_download(REPO, p, repo_type="dataset", revision=a.revision, local_dir=a.out)
    h = hashlib.sha256()
    with open(local, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    want = info[p].lfs.sha256 if info[p].lfs else None
    assert want is None or h.hexdigest() == want, f"checksum mismatch for {p}"
    manifest["files"].append({"path": p, "sha256": h.hexdigest(), "bytes": Path(local).stat().st_size, "verified_against_lfs": want is not None})
    print(json.dumps(manifest["files"][-1]), flush=True)
Path(a.out, "download_manifest.json").write_text(json.dumps(manifest, indent=2))
