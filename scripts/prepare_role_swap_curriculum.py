"""Pair both queried-object orders for each one-hop fact without eval overlap."""
import argparse
import json
import re
from pathlib import Path

from systemone_lab.data.spatial_worlds import INVERSE_REL

QUESTION = re.compile(r'What is the spatial relation of (.+) to (.+)\?')


def key(record):
    return record['state'], record['questions']['spatial']['instructions']


def swap_record(record):
    r = json.loads(json.dumps(record))
    if list(r['questions']) != ['spatial']:
        raise ValueError('expected exactly one spatial question')
    match = QUESTION.fullmatch(r['questions']['spatial']['instructions'])
    if match is None or match[1] == match[2]:
        raise ValueError('invalid query object names')
    label = r['labels']['spatial']
    if label not in INVERSE_REL:
        raise ValueError(f'unknown relation: {label}')
    r['id'] += '-role-swapped'
    r['questions']['spatial']['instructions'] = (
        f'What is the spatial relation of {match[2]} to {match[1]}?')
    r['labels']['spatial'] = INVERSE_REL[label]
    r['meta']['query_role'] = 'reversed'
    return r


def original_and_swapped(record):
    original = json.loads(json.dumps(record))
    original['meta']['query_role'] = 'forward'
    return original, swap_record(record)


def source_groups(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(records) % 2:
        raise ValueError('expected numbered/renamed pairs')
    for numbered, renamed in zip(records[::2], records[1::2]):
        if numbered['meta']['pair_id'] != renamed['meta']['pair_id']:
            raise ValueError('pair IDs differ')
        if [numbered['meta']['variant'], renamed['meta']['variant']] != ['numbered', 'renamed']:
            raise ValueError('pair order must be numbered, renamed')
        group = [r for base in (numbered, renamed) for r in original_and_swapped(base)]
        if len({key(r) for r in group}) != 4:
            raise ValueError('duplicate prompt inside pair group')
        yield group, {'id': numbered['meta']['pair_id'], 'label': numbered['labels']['spatial'],
                      'numbered': numbered, 'renamed': renamed}


def protected_keys(source):
    paths = [source / 'heldout.jsonl',
             Path('reports/paired_onehop/vertical_probe_pairs.json'),
             Path('reports/paired_onehop/counterfactual.jsonl')]
    protected = set(); used = []
    for path in paths:
        if not path.exists():
            if path == source / 'heldout.jsonl': raise FileNotFoundError(path)
            continue
        used.append(str(path))
        if path.name.endswith('_pairs.json'):
            records = [r for pair in json.loads(path.read_text())
                       for r in (pair['numbered'], pair['renamed'])]
        else:
            records = [json.loads(line) for line in path.read_text().splitlines()]
        for r in records:
            for variant in original_and_swapped(r): protected.add(key(variant))
    return protected, used


def augment(source_dir, out_dir):
    source = Path(source_dir); root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    protected, sources = protected_keys(source)
    manifest = {'source_dir': str(source), 'inverse_labels': INVERSE_REL,
                'protected_evaluation_sources': sources, 'query_roles_per_rendering': 2}
    seen_train = set(); output = {}
    for split in ('train', 'heldout'):
        rows = []; pairs = []; seen_split = set(); skipped = 0
        for group, pair in source_groups(source / f'{split}.jsonl'):
            keys = {key(r) for r in group}
            # Keep all four renderings of a world together or drop the group.
            if (keys & (protected | seen_train | seen_split) if split == 'train'
                    else keys & seen_split):
                skipped += 1; continue
            seen_split.update(keys); rows.extend(group); pairs.append(pair)
        if split == 'train': seen_train = seen_split
        elif seen_train & seen_split: raise AssertionError('train/held-out overlap')
        (root / f'{split}.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        output[split] = rows
        manifest[f'{split}_records'] = len(rows)
        manifest[f'{split}_four_rendering_groups'] = len(pairs)
        manifest[f'{split}_groups_skipped_for_duplicates_or_leakage'] = skipped
        if split == 'heldout':
            (root / 'name_pairs.json').write_text(json.dumps(pairs, indent=2), encoding='utf-8')
    manifest['train_and_heldout_disjoint'] = True
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return root


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', default='reports/paraphrase_onehop')
    ap.add_argument('--out-dir', default='reports/role_swap_onehop')
    args = ap.parse_args()
    print(augment(args.source_dir, args.out_dir))
