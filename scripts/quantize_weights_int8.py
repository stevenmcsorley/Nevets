"""Weight-only int8 for ONNX: large MatMul/Gather weights stored as int8 with per-channel scales.

Each weight W is replaced by DequantizeLinear(W_q, scale, zero_point=0, axis), so compute stays in
fp32 and only storage is quantized. Uses standard opset-13 ops that every onnxruntime backend
(including onnxruntime-web WASM) supports. Symmetric per-channel: scale = max|w| / 127 along the
output channel (MatMul weights [in, out]: axis=1; Gather tables [rows, dim]: axis=0).
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--min-size", type=int, default=65536); a = ap.parse_args()
    model = onnx.load(a.src); g = model.graph
    inits = {i.name: i for i in g.initializer}
    use = {}
    for n in g.node:
        if n.op_type == "MatMul" and len(n.input) > 1 and n.input[1] in inits: use.setdefault(n.input[1], ("matmul", 1))
        if n.op_type == "Gather" and n.input[0] in inits: use.setdefault(n.input[0], ("gather", 0))
    new_inits, new_nodes, done = [], [], 0
    for name, (kind, axis) in use.items():
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        if w.ndim != 2 or w.size < a.min_size: continue
        scale = np.abs(w).max(axis=1 - axis, keepdims=False) / 127.0
        scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        q = np.clip(np.round(w / (scale[None, :] if axis == 1 else scale[:, None])), -127, 127).astype(np.int8)
        g.initializer.remove(inits[name])
        new_inits += [numpy_helper.from_array(q, name + "_q"), numpy_helper.from_array(scale, name + "_scale"),
                      numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + "_zp")]
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale", name + "_zp"], [name],
                                          axis=axis, name=name + "_dequant"))
        done += 1
    g.initializer.extend(new_inits)
    for n in reversed(new_nodes): g.node.insert(0, n)
    onnx.checker.check_model(model); onnx.save(model, a.dst)
    print(f"quantized {done} weights -> {a.dst}")


if __name__ == "__main__":
    main()
