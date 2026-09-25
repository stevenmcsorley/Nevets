"""One-hop spatial worlds rendered twice with different object names.

Train/held-out latents are disjoint. Both renderings carry the same label.
"""
import argparse
import json
import random
from pathlib import Path

from systemone_lab.data.spatial_worlds import LABELS, TEMPLATES

ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ'


def _record(pair_id, variant, label, template, names):
    a, b = names
    return {
        'id': f'{pair_id}-{variant}',
        'state': template.format(a=a, b=b),
        'questions': {'spatial': {
            'type': 'choice',
            'instructions': f'What is the spatial relation of {a} to {b}?',
            'criteria': {key: key for key in LABELS},
        }},
        'labels': {'spatial': label},
        'meta': {'pair_id': pair_id, 'variant': variant, 'hops': 1},
    }


def generate(out_dir, train_pairs=800, heldout_pairs=200, counterfactual_pairs=100, seed=19091):
    if min(train_pairs, heldout_pairs, counterfactual_pairs) < 1:
        raise ValueError('pair counts must be positive')
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed); seen = set()
    splits = {}
    for split, n in [('train', train_pairs), ('heldout', heldout_pairs)]:
        rows = []; pairs = []
        for i in range(n):
            label = LABELS[i % len(LABELS)]
            for _ in range(10000):
                template = rng.choice(TEMPLATES[label])
                numbers = rng.sample(range(24), 2)
                letters = rng.sample(ALPHABET, 2)
                numbered = _record(f'{split}-{i}', 'numbered', label, template,
                                   (f'obj_{numbers[0]}', f'obj_{numbers[1]}'))
                renamed = _record(f'{split}-{i}', 'renamed', label, template, letters)
                keys = [(r['state'], r['questions']['spatial']['instructions']) for r in (numbered, renamed)]
                if all(k not in seen for k in keys): break
            else:
                raise RuntimeError('unable to create unique pair')
            seen.update(keys)
            rows.extend((numbered, renamed))
            pairs.append({'id': f'{split}-{i}', 'label': label,
                          'numbered': numbered, 'renamed': renamed})
        (root / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        splits[split] = pairs
    (root / 'name_pairs.json').write_text(json.dumps(splits['heldout'], indent=2), encoding='utf-8')

    cf_rows = []
    for i in range(counterfactual_pairs):
        before_label = LABELS[i % len(LABELS)]
        after_label = rng.choice([x for x in LABELS if x != before_label])
        for _ in range(10000):
            names = rng.sample(ALPHABET, 2)
            before = _record(f'cf-{i}', 'base', before_label, rng.choice(TEMPLATES[before_label]), names)
            after = _record(f'cf-{i}', 'counterfactual', after_label, rng.choice(TEMPLATES[after_label]), names)
            keys = [(r['state'], r['questions']['spatial']['instructions']) for r in (before, after)]
            if len(set(keys)) == 2 and all(k not in seen for k in keys): break
        else:
            raise RuntimeError('unable to create unique counterfactual pair')
        seen.update(keys); cf_rows.extend((before, after))
    (root / 'counterfactual.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in cf_rows), encoding='utf-8')
    (root / 'manifest.json').write_text(json.dumps({
        'seed': seed, 'train_pairs': train_pairs, 'train_records': 2 * train_pairs,
        'heldout_pairs': heldout_pairs, 'heldout_records': 2 * heldout_pairs,
        'counterfactual_pairs': counterfactual_pairs,
        'labels': LABELS, 'numbered_and_letter_names_for_each_world': True,
    }, indent=2), encoding='utf-8')
    return root


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='reports/paired_onehop')
    ap.add_argument('--train-pairs', type=int, default=800)
    ap.add_argument('--heldout-pairs', type=int, default=200)
    ap.add_argument('--counterfactual-pairs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=19091)
    args = ap.parse_args()
    print(generate(args.out_dir, args.train_pairs, args.heldout_pairs,
                   args.counterfactual_pairs, args.seed))
