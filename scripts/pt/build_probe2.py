"""Freeze a copy of TM-1 training with later-registered eval collisions removed."""
import hashlib
import json
from pathlib import Path
from systemone_lab.eval_registry import eval_states

src = Path('data/processed/worlds_v1/train.jsonl')
dst = Path('data/processed/probe2/train.jsonl')
rows = [json.loads(line) for line in src.read_text(encoding='utf-8').splitlines()]
banned = eval_states()
bad_pairs = {r.get('meta', {}).get('pair_id') for r in rows if r['state'] in banned} - {None}
kept = [r for r in rows if r['state'] not in banned and r.get('meta', {}).get('pair_id') not in bad_pairs]
payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept).encode('utf-8')
dst.parent.mkdir(parents=True, exist_ok=True)
if dst.exists() and dst.read_bytes() != payload:
    raise SystemExit('Frozen PROBE-2 training changed; review required')
dst.write_bytes(payload)
manifest = dict(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                records=len(kept), removed=len(rows)-len(kept), sha256=hashlib.sha256(payload).hexdigest())
(dst.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest))
