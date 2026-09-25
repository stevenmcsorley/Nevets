import json
from collections import Counter
from pathlib import Path

from scripts.generate_paired_onehop import generate


def test_paired_onehop_splits_and_name_invariance(tmp_path):
    root = generate(tmp_path, train_pairs=27, heldout_pairs=18, counterfactual_pairs=9, seed=14)
    train = [json.loads(x) for x in (root / 'train.jsonl').read_text().splitlines()]
    heldout = [json.loads(x) for x in (root / 'heldout.jsonl').read_text().splitlines()]
    pairs = json.loads((root / 'name_pairs.json').read_text())
    assert len(train) == 54 and len(heldout) == 36 and len(pairs) == 18
    assert len({(r['state'], r['questions']['spatial']['instructions']) for r in train + heldout}) == 90
    assert set(Counter(r['labels']['spatial'] for r in train).values()) == {6}
    for pair in pairs:
        numbered, renamed = pair['numbered'], pair['renamed']
        assert numbered['labels'] == renamed['labels']
        assert numbered['questions']['spatial']['criteria'] == renamed['questions']['spatial']['criteria']
        assert 'obj_' in numbered['state'] and 'obj_' not in renamed['state']
        assert numbered['meta']['pair_id'] == renamed['meta']['pair_id']


def test_paired_onehop_counterfactual_changes_fact(tmp_path):
    root = generate(tmp_path, train_pairs=9, heldout_pairs=9, counterfactual_pairs=9, seed=15)
    rows = [json.loads(x) for x in (root / 'counterfactual.jsonl').read_text().splitlines()]
    for before, after in zip(rows[::2], rows[1::2]):
        assert before['questions'] == after['questions']
        assert before['state'] != after['state']
        assert before['labels'] != after['labels']
        assert before['meta']['pair_id'] == after['meta']['pair_id']
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from systemone_lab.data.spatial_worlds import TEMPLATES


def test_paired_paraphrase_generator_covers_each_wording(tmp_path):
    script=Path('scripts/generate_paired_paraphrase_onehop.py').resolve()
    result=subprocess.run([sys.executable,str(script),'--out-dir',str(tmp_path),
                           '--train-worlds','9','--heldout-worlds','9','--seed','27'],
                          capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    train=[json.loads(x) for x in (tmp_path/'train.jsonl').read_text().splitlines()]
    heldout=[json.loads(x) for x in (tmp_path/'heldout.jsonl').read_text().splitlines()]
    assert len(train)==len(heldout)==2*sum(len(v) for v in TEMPLATES.values())
    all_keys=[(r['state'],r['questions']['spatial']['instructions']) for r in train+heldout]
    assert len(all_keys)==len(set(all_keys))
    groups=defaultdict(list)
    for r in train:
        groups[r['meta']['pair_id'].rsplit('-template-',1)[0]].append(r)
    for records in groups.values():
        label=records[0]['labels']['spatial']
        assert len(records)==2*len(TEMPLATES[label])
        assert Counter(r['meta']['variant'] for r in records)=={'numbered':len(TEMPLATES[label]),'renamed':len(TEMPLATES[label])}
        assert len({r['questions']['spatial']['instructions'] for r in records})==2
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['all_wording_templates_per_world']
from scripts.prepare_role_swap_curriculum import swap_record, augment
from systemone_lab.data.spatial_worlds import INVERSE_REL, LABELS


def test_role_swap_inverts_all_nine_labels():
    for label in LABELS:
        record={'id':'x','state':'A is related to B.',
                'questions':{'spatial':{'type':'choice','instructions':'What is the spatial relation of A to B?',
                                        'criteria':{k:k for k in LABELS}}},
                'labels':{'spatial':label},'meta':{}}
        swapped=swap_record(record)
        assert swapped['state']==record['state']
        assert swapped['questions']['spatial']['instructions']=='What is the spatial relation of B to A?'
        assert swapped['labels']['spatial']==INVERSE_REL[label]
        assert swap_record(swapped)['labels']==record['labels']


def test_role_swap_curriculum_keeps_groups_and_splits_disjoint(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    from scripts.generate_paired_onehop import _record
    from systemone_lab.data.spatial_worlds import TEMPLATES
    for split,names in [('train',('obj_1','obj_0')),('heldout',('obj_3','obj_2'))]:
        rows=[_record(f'{split}-0',variant,'above',TEMPLATES['above'][0],pair)
              for variant,pair in [('numbered',names),('renamed',('A','B') if split=='train' else ('C','D'))]]
        (source/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (source/'name_pairs.json').write_text('[]')
    root=augment(source,tmp_path/'output')
    train=[json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()]
    heldout=[json.loads(x) for x in (root/'heldout.jsonl').read_text().splitlines()]
    assert len(train)==len(heldout)==4
    keys=lambda rows:{(r['state'],r['questions']['spatial']['instructions']) for r in rows}
    assert not keys(train)&keys(heldout)
    for rows in (train,heldout):
        assert Counter(r['meta']['query_role'] for r in rows)=={'forward':2,'reversed':2}
        assert Counter(r['labels']['spatial'] for r in rows)=={'above':2,'below':2}


def test_binding_stress_pairs_are_role_swaps(tmp_path):
    import json, re, subprocess, sys
    from collections import defaultdict
    from systemone_lab.data.spatial_worlds import INVERSE_REL
    out = tmp_path / 'stress.jsonl'
    subprocess.run([sys.executable, 'scripts/generate_binding_stress.py', '--out', str(out),
                    '--worlds-per-suite', '5', '--seed', '3'], check=True, capture_output=True)
    rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 4 * 5 * 2 * 2
    pairs = defaultdict(dict)
    for r in rows: pairs[r['meta']['pair_id']][r['meta']['query_role']] = r
    for p in pairs.values():
        f, b = p['forward'], p['reversed']
        assert f['state'] == b['state'] and b['labels']['spatial'] == INVERSE_REL[f['labels']['spatial']]
        x, y = re.fullmatch(r'What is the spatial relation of (.+) to (.+)\?', f['questions']['spatial']['instructions']).groups()
        assert b['questions']['spatial']['instructions'] == f'What is the spatial relation of {y} to {x}?'
        if f['meta']['suite'] == 'onehop_mention':
            assert len(re.findall(rf'(?<![\w-]){re.escape(x)}(?![\w-])', f['state'])) + \
                   len(re.findall(rf'(?<![\w-]){re.escape(y)}(?![\w-])', f['state'])) > 2
