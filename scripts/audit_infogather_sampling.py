"""CPU-only audit of diagnostic-vs-repair sampling; no evaluation predictions used."""
import json
from collections import Counter
from pathlib import Path
from systemone_lab.training import domain_balanced_weights


def main():
    counts = Counter(); mapping = {}; records = []
    source = Path('data/processed/general_v3/train.jsonl')
    for line in source.open(encoding='utf-8'):
        r = json.loads(line)
        if r['meta']['domain'] != 'infogather': continue
        label = r['labels']['answer']; kind = 'test' if label == 'run the diagnostic' else 'act'
        counts[kind] += 1; mapping[r['id']] = kind; records.append(r)
    legacy = domain_balanced_weights(records, 'infogather=1')
    fixed = domain_balanced_weights(records, 'infogather=1', True)
    realised = Counter()
    for line in Path('checkpoints/general/general-v4_s7.diagnostics.jsonl').open():
        r = json.loads(line)
        for rid in r.get('batch_ids', []):
            if rid in mapping: realised[mapping[rid]] += 1
    result = dict(training_record_counts=dict(counts), logged_G4_counts=dict(realised),
        unweighted_test_fraction=counts['test']/sum(counts.values()),
        legacy_weighted_test_fraction=sum(w for r,w in zip(records,legacy) if mapping[r['id']]=='test'),
        semantic_weighted_test_fraction=sum(w for r,w in zip(records,fixed) if mapping[r['id']]=='test'),
        caveat='Shared by earlier label-balanced recipes; not proven to cause G4-minus-G1 regression. Sampling change requires controlled training.')
    out=Path('reports/general/diagnosis_20260929'); out.mkdir(parents=True, exist_ok=True)
    (out/'sampling.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
