"""Controlled scorer/LR comparison, only after both 32-example gates pass."""
import json
from pathlib import Path
from sanity_decision_overfit import tiny_records, run

def main():
    for gate in ('micro32_frozen','micro32_unfrozen'):
        r=json.loads(Path(f'reports/{gate}.json').read_text())
        if r['accuracy']<.99 or r['numerical_failures']: raise SystemExit(f'{gate} failed')
    results=[]
    for scorer in ('scaled_dot','cosine'):
        for lr in (3e-6,1e-5,3e-5,1e-4):
            results.append(run(tiny_records(),True,scorer,lr,1500,name=f'ablation_{scorer}_{lr}'))
            Path('reports/pointer_scorer_ablation.json').write_text(json.dumps(results,indent=2))
    lines=['| Scorer | LR | Accuracy | Final loss | Max q norm | Max k norm | Max logit | Max gradient norm | Numerical failures | Steps to 99% |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append('| '+' | '.join(str(r[k]) for k in ('scorer','lr','accuracy','final_loss','max_q_norm','max_k_norm','max_logit','max_gradient_norm','numerical_failures','steps_to_99'))+' |')
    Path('reports/pointer_scorer_ablation.md').write_text('\n'.join(lines))
if __name__=='__main__': main()
