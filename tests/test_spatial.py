import random
import tempfile
from pathlib import Path

import torch

from systemone_lab.config import ModelConfig
from systemone_lab.eval import predict_record
from systemone_lab.model import SystemOneModel
from systemone_lab.tokenizer import LabTokenizer, train_tokenizer
from systemone_lab.data.spatial_worlds import make_example, relation


def test_generator_labels_match_coords():
    r=random.Random(1)
    for _ in range(100):
        ex=make_example(r,hops=r.randint(1,6),distractors=2,transform=r.choice(["identity","rot90","mirror_x"]),translate=(4,-3),renamed=True)
        a,b=ex.meta["query"]
        assert ex.label==relation(ex.meta["coords"][a],ex.meta["coords"][b])


def test_predict_record_works_in_eval_mode():
    with tempfile.TemporaryDirectory() as tmp:
        tok_path = Path(tmp) / "tokenizer.json"
        train_tokenizer([
            "A is left of B. B is above C. What is the spatial relation of A to C?",
            "upper-left above left right below lower-left lower-right overlap",
        ], tok_path, vocab_size=256)
        tokenizer = LabTokenizer(tok_path)
        model = SystemOneModel(ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            pointer_dim=16,
        ))
        model.eval()
        rec = {
            "state": "A is left of B. B is above C.",
            "questions": {
                "spatial": {
                    "type": "choice",
                    "instructions": "What is the spatial relation of A to C?",
                    "criteria": {
                        "upper-left": "upper-left",
                        "above": "above",
                        "left": "left",
                        "right": "right",
                        "below": "below",
                        "upper-right": "upper-right",
                        "lower-left": "lower-left",
                        "lower-right": "lower-right",
                        "overlap": "overlap",
                    },
                }
            },
            "labels": {"spatial": "upper-left"},
        }
        out = predict_record(model, tokenizer, rec, torch.device("cpu"))
        assert set(out) == {"spatial"}
        assert all(isinstance(v, dict) for v in out.values())
