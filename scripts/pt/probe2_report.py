"""Fail-closed positive control, then preregistered noise band and slope comparison."""
import argparse
import json
import math
import statistics
from pathlib import Path

METRICS = ('in_format', 'table', 'cf_both', 'ece_in')
SEEDS = (11, 12, 13)
GRID = (250, 500, 750, 1000)


def metrics(rep, name):
    w = json.loads((rep / f'probe_{name}_worlds.json').read_text())
    result = dict(in_format=w['eval_in_format']['overall']['accuracy'],
                  table=w['eval_heldout_table']['overall']['accuracy'],
                  cf_both=statistics.mean(v['both_correct'] for v in w['counterfactual'].values()),
                  ece_in=w['eval_in_format']['overall']['ece15'])
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError(f'Invalid metrics: {name}')
    return result


def control_result(legacy, random):
    if len(legacy) != 3 or len(random) != 3:
        raise ValueError('Positive control requires three seeds per arm')
    delta = statistics.mean(legacy) - statistics.mean(random)
    band = 2 * math.sqrt(statistics.variance(legacy) / 3 + statistics.variance(random) / 3)
    return dict(legacy=legacy, random=random, delta=delta, noise_band=band,
                passed=delta > band and delta >= .03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['control', 'band', 'compare'])
    ap.add_argument('--rep', type=Path, default=Path('reports/pt/probe2'))
    ap.add_argument('--ledger', type=Path, default=Path('reports/RESEARCH_LEDGER.md'))
    a = ap.parse_args(); rep = a.rep
    if a.mode == 'control':
        result = control_result(
            [metrics(rep, f'legacy_s1_35m_s{s}')['in_format'] for s in SEEDS],
            [metrics(rep, f'pt_35m_tok_0M_s{s}')['in_format'] for s in SEEDS])
        (rep / 'control.json').write_text(json.dumps(result, indent=2))
        with a.ledger.open('a', encoding='utf-8') as f:
            f.write('\n\n**PROBE-2 positive control:** ' + json.dumps(result) + '\n')
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['passed'] else 2)
    if not json.loads((rep / 'control.json').read_text())['passed']:
        raise ValueError('Positive control did not pass')
    if a.mode == 'band':
        cells = [[metrics(rep, f'pt_{size}_tok_{m}M_s{s}') for s in SEEDS]
                 for size in ('35m', '150m') for m in (250, 1000)]
        result = {k: 2 * math.sqrt(statistics.mean(statistics.variance(r[k] for r in cell)
                    for cell in cells) * 2 / 5) for k in METRICS}
        (rep / 'band.json').write_text(json.dumps(result, indent=2))
        with a.ledger.open('a', encoding='utf-8') as f:
            f.write('\n\n**PROBE-2 noise band (before slope comparison):** ' + json.dumps(result) + '\n')
    else:
        band = json.loads((rep / 'band.json').read_text())
        points = {size: [metrics(rep, f'pt_{size}_tok_{m}M_s11') for m in GRID]
                  for size in ('35m', '150m')}
        slopes = {size: {k: sum((i - 2.5) * r[k] for i, r in enumerate(rows, 1)) / 5
                         for k in METRICS} for size, rows in points.items()}
        decision = {}
        for k in METRICS:
            d = slopes['150m'][k] - slopes['35m'][k]
            decision[k] = ('within noise' if abs(d) <= band[k] else
                           '150M' if (d < 0 if k == 'ece_in' else d > 0) else '35M')
        result = dict(points=points, slope_per_250M=slopes, band=band, decision=decision,
                      pt3='Not started. Review positive slopes and calibration before owner approval.')
        (rep / 'report.json').write_text(json.dumps(result, indent=2))
        with a.ledger.open('a', encoding='utf-8') as f:
            f.write('\n\n**PROBE-2 slope comparison:** ' + json.dumps(result) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
