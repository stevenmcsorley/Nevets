# PT-1 corpus manifest

| Field | Value |
|---|---|
| Source | `HuggingFaceFW/fineweb-edu` sample/10BT/000_00000.parquet, sample/10BT/001_00000.parquet |
| Revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Licence | ODC-By v1.0 (attribution: FineWeb-Edu, Hugging Face) |
| Download SHA-256 (verified against LFS oid) | 000_00000.parquet: `b1ba7b2ce4cb5ea6…`; 001_00000.parquet: `3fcf2dc69cd52503…` |
| Documents in | 1,455,000 |
| Exact duplicates removed | 7,799 |
| Decontamination: marker candidates / confirmed 13-gram or short-string hits | 34 / 0 (against 59,469 registered eval + locked texts and chess eval positions) |
| Documents out | 1,447,201 |
| Tokenizer | `tokenizers/pt_32k.json` (copy of `data/pt/tokenizer_32k.json`), SHA-256 `229a91f7d1befea96cd93af911c79cd62765101610470f4866a7d2477c9e1126`, vocab 32000, byte-level BPE, individual digits, full byte alphabet |
| Tokenizer training text | 90,000 corpus docs (428,278,013 chars) + 15,000 rendered structured samples (all worlds formats incl. P3 held-out ones — allowed for tokenizer only) |
| Train tokens | 1,547,579,810 in 16 uint16 shards |
| Validation tokens | 7,847,636 (0.5% of documents, held out) |
| Pretraining structured slice | none — shards are FineWeb-Edu text only (P3 held-out formats: table, symbolic) |

Shard hashes: `reports/pt/shards_manifest.json`. Reproduce: `scripts/pt/download_fineweb.py --files 2`, then `scripts/pt/build_corpus.py clean|tokenizer|shards`.
