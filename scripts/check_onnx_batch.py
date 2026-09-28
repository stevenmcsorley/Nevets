"""Check variable-length vessel batches against single-request PyTorch; never deploys."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from systemone_lab.formatting import pack_request
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.training import load_checkpoint
from export_onnx_multi import MultiDecisionGraph, multi_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--graphs', nargs='+', required=True)
    ap.add_argument('--records', default='data/processed/vessel_v1/eval_real.jsonl')
    ap.add_argument('--limit', type=int, default=12)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); torch.set_num_threads(2)
    model, ck = load_checkpoint(a.ckpt, SystemOneModel, 'cpu'); model.eval()
    tok = LabTokenizer(ck['tokenizer']); graph = MultiDecisionGraph(model).eval()
    rows = [json.loads(l) for l in Path(a.records).read_text(encoding='utf-8').splitlines()][:a.limit]
    if not rows: raise ValueError('No parity records')
    inputs, counts, refs = [], [], []
    with torch.no_grad():
        for r in rows:
            if any(q.get('query_entities') for q in r['questions'].values()):
                raise ValueError('Vessel-only graph: entity binding unsupported')
            inp, nk = multi_inputs(model, pack_request(tok, r['state'], r['questions'], isolate_options=model.isolated_options))
            inputs.append(inp); counts.append(nk); refs.append(graph(*inp.values()).numpy())
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
    report = {}
    softmax = lambda x: np.exp(x-x.max()) / np.exp(x-x.max()).sum()
    for path in a.graphs:
        session = ort.InferenceSession(path, so)
        for batch in (1, 2, 4):
            agree = n = 0; diff = 0.
            for start in range(0, len(inputs), batch):
                chunk = inputs[start:start+batch]; T = max(i['input_ids'].shape[1] for i in chunk)
                K = max(i['options'].shape[1] for i in chunk)
                feed = dict(input_ids=np.zeros((len(chunk),T),np.int64), position_ids=np.zeros((len(chunk),T),np.int64),
                            mask=np.zeros((len(chunk),T,T),bool)); ds=[]; opts=[]
                for b, inp in enumerate(chunk):
                    t = inp['input_ids'].shape[1]
                    for k in ('input_ids','position_ids'): feed[k][b,:t] = inp[k][0].numpy()
                    feed['mask'][b,:t,:t] = inp['mask'][0].numpy()
                    for j in range(t,T): feed['mask'][b,j,j] = True
                    ds.extend((inp['decide']+b*T).tolist())
                    for row in inp['options'].tolist(): opts.append([v+b*T for v in row+[row[0]]*(K-len(row))])
                feed.update(decide=np.array(ds,np.int64), options=np.array(opts,np.int64))
                pred = session.run(None,feed)[0]; qi = 0
                for b in range(len(chunk)):
                    for q,k in enumerate(counts[start+b]):
                        ref = softmax(refs[start+b][q,:k]); prob = softmax(pred[qi,:k]); qi += 1
                        agree += int(ref.argmax()==prob.argmax()); n += 1
                        diff = max(diff,float(np.max(np.abs(ref-prob))))
            report[f'{Path(path).name}/B{batch}'] = dict(questions=n,argmax_agree=agree/n,max_abs_prob_diff=diff,
                                                       passes_existing_threshold=agree/n>=.99 and diff<=.02)
    Path(a.out).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
