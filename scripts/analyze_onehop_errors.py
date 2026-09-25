"""Break down matched-name errors by the exact one-hop wording template."""
import argparse
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from systemone_lab.data.spatial_worlds import TEMPLATES

ap=argparse.ArgumentParser()
ap.add_argument('--pairs',default='reports/paired_onehop/name_pairs.json')
ap.add_argument('--results',default='reports/paired_onehop/name_results.json')
ap.add_argument('--out',default='reports/paired_onehop/template_errors.json')
a=ap.parse_args()
pairs=json.loads(Path(a.pairs).read_text())
results=json.loads(Path(a.results).read_text())['results']
if len(pairs)!=len(results): raise ValueError('pair/result count mismatch')
counts=defaultdict(Counter)
for pair,z in zip(pairs,results):
    if pair['id']!=z['id'] or pair['label']!=z['label']:
        raise ValueError('pair/result identity mismatch')
    label=pair['label']
    for variant in ('numbered','renamed'):
        rec=pair[variant]
        match=re.fullmatch(r'What is the spatial relation of (.+) to (.+)\?',rec['questions']['spatial']['instructions'])
        if match is None: raise ValueError('unrecognized question')
        template=next((t for t in TEMPLATES[label] if t.format(a=match[1],b=match[2])==rec['state']),None)
        if template is None: raise ValueError('state does not match label template')
        row=counts[(label,template,variant)]
        row['n']+=1; row['correct']+=int(z[f'{variant}_correct']); row[f'predicted_{z[variant]}']+=1
output=[{'label':label,'template':template,'names':variant,'n':c['n'],
         'accuracy':c['correct']/c['n'],
         'predictions':{k.removeprefix('predicted_'):v for k,v in c.items() if k.startswith('predicted_')}}
        for (label,template,variant),c in sorted(counts.items())]
Path(a.out).write_text(json.dumps(output,indent=2))
for row in output:
    if row['accuracy']<.8: print(row)
