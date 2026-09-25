"""Create paired name and wording variants for the same one-hop spatial world."""
import argparse
import json
import random
from pathlib import Path

from systemone_lab.data.spatial_worlds import LABELS, TEMPLATES
from generate_paired_onehop import ALPHABET, _record


def key(record):
    return record['state'], record['questions']['spatial']['instructions']


def protected_keys():
    root = Path('reports/paired_onehop'); keys = set(); sources = []
    for name in ('heldout.jsonl', 'counterfactual.jsonl'):
        path = root / name
        if path.exists():
            sources.append(str(path))
            keys.update(key(json.loads(line)) for line in path.read_text().splitlines())
    path = root / 'vertical_probe_pairs.json'
    if path.exists():
        sources.append(str(path))
        for pair in json.loads(path.read_text()):
            keys.add(key(pair['numbered'])); keys.add(key(pair['renamed']))
    return keys, sources


def generate(out_dir, train_worlds=800, heldout_worlds=200, seed=20091):
    if min(train_worlds, heldout_worlds) < 1:
        raise ValueError('world counts must be positive')
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed); seen, excluded = protected_keys(); rows_by_split = {}
    for split, n in [('train', train_worlds), ('heldout', heldout_worlds)]:
        rows = []; pairs = []
        for i in range(n):
            label = LABELS[i % len(LABELS)]
            for _ in range(10000):
                numbers = rng.sample(range(24), 2); letters = rng.sample(ALPHABET, 2)
                names = {'numbered': (f'obj_{numbers[0]}', f'obj_{numbers[1]}'),
                         'renamed': tuple(letters)}
                variants = []
                for template_index, template in enumerate(TEMPLATES[label]):
                    pair_id = f'{split}-{i}-template-{template_index}'
                    numbered = _record(pair_id, 'numbered', label, template, names['numbered'])
                    renamed = _record(pair_id, 'renamed', label, template, names['renamed'])
                    variants.append((numbered, renamed))
                keys = [key(r) for pair in variants for r in pair]
                if len(set(keys)) == len(keys) and all(k not in seen for k in keys): break
            else:
                raise RuntimeError(f'unable to create unique {split} world {i}')
            seen.update(keys)
            for numbered, renamed in variants:
                rows.extend((numbered, renamed))
                pairs.append({'id': numbered['meta']['pair_id'], 'label': label,
                              'numbered': numbered, 'renamed': renamed})
        (root / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        rows_by_split[split] = rows
        if split == 'heldout':
            (root / 'name_pairs.json').write_text(json.dumps(pairs, indent=2), encoding='utf-8')
    manifest = {'seed': seed, 'train_worlds': train_worlds,
                'train_records': len(rows_by_split['train']),
                'heldout_worlds': heldout_worlds,
                'heldout_records': len(rows_by_split['heldout']),
                'heldout_name_pairs': len(rows_by_split['heldout']) // 2,
                'all_wording_templates_per_world': True,
                'both_name_renderings_per_template': True,
                'protected_evaluation_sources': excluded}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return root


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='reports/paraphrase_onehop')
    ap.add_argument('--train-worlds', type=int, default=800)
    ap.add_argument('--heldout-worlds', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20091)
    args = ap.parse_args()
    print(generate(args.out_dir, args.train_worlds, args.heldout_worlds, args.seed))
