"""Experimental vessel batching graph; flattened readout positions isolate each vessel.

One [B,T] backbone pass; decide/options index into flattened [B*T,D] hidden states.
Binding is omitted: vessel questions have no query_entities. Not a general spatial exporter.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from systemone_lab.model import SystemOneModel
from systemone_lab.training import load_checkpoint
from systemone_lab.tokenizer import LabTokenizer
from systemone_lab.formatting import pack_request
from export_onnx_multi import multi_inputs


class VesselBatchGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.temperature = float(model.cfg.decision_head.get('temperature', 10))

    def forward(self, input_ids, position_ids, mask, decide, options):
        hidden = self.m.hidden(input_ids, position_ids, mask)
        h = hidden.reshape(-1, hidden.shape[-1]).float()
        q = F.normalize(self.m.ptr_q(h[decide]), dim=-1)
        k = F.normalize(self.m.ptr_k(h[options]), dim=-1)
        return self.temperature * torch.einsum('nkp,np->nk', k, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    m, ck = load_checkpoint(a.ckpt, SystemOneModel, 'cpu'); m.eval()
    tok = LabTokenizer(ck['tokenizer'])
    records = [json.loads(l) for l in Path('data/processed/vessel_v1/eval_real.jsonl').read_text().splitlines()][:2]
    r = records[0]
    assert not any(q.get('query_entities') for q in r['questions'].values())
    p = pack_request(tok, r['state'], r['questions'], isolate_options=m.isolated_options)
    inp, _ = multi_inputs(m, p); inp.pop('bind')
    graph = VesselBatchGraph(m).eval()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fp = out.with_suffix('.fp32.onnx')
    torch.onnx.export(graph, tuple(inp.values()), str(fp), input_names=list(inp), output_names=['logits'],
                      dynamic_axes={'input_ids': {0:'B',1:'T'},'position_ids': {0:'B',1:'T'},
                                    'mask': {0:'B',1:'T',2:'T'},'decide': {0:'N'},
                                    'options': {0:'N',1:'K'},'logits': {0:'N',1:'K'}},opset_version=17,dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp),str(out),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gather'])
    import onnx
    # Recurrent weights are aliased by Identity nodes; ORT's dynamic quantizer otherwise
    # sees only the first pass as an initializer and leaves later passes in fp32.
    folded = onnx.load(str(fp))
    aliases = {n.output[0]: n.input[0] for n in folded.graph.node if n.op_type == 'Identity'}
    for node in folded.graph.node:
        for i, name in enumerate(node.input):
            while name in aliases:
                name = aliases[name]
            node.input[i] = name
    nodes = [n for n in folded.graph.node if n.op_type != 'Identity']
    del folded.graph.node[:]; folded.graph.node.extend(nodes)
    onnx.checker.check_model(folded)
    folded_path = out.with_suffix('.folded.fp32.onnx')
    onnx.save(folded, str(folded_path))
    full = out.with_suffix('.fullint8.onnx')
    quantize_dynamic(str(folded_path),str(full),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gather'])
    counts = {str(f):dict(Counter(n.op_type for n in onnx.load(str(f)).graph.node)) for f in [fp,out,full]}
    out.with_suffix('.ops.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
