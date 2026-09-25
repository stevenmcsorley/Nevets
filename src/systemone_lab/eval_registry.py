"""Single source of truth for every evaluation file a training mix must never overlap.

Add new evaluation or locked sets here. scripts/check_contamination.py refuses a training file whose
states occur in any of them. Paired files (.json) store {"numbered": rec, "renamed": rec} items.
"""
import json
from pathlib import Path

EVAL_JSONL = [
    "reports/chain/eval_hops.jsonl", "reports/chain/transforms.jsonl", "reports/binding_stress/stress.jsonl",
    "reports/role_swap_onehop/heldout.jsonl", "reports/paired_onehop/counterfactual.jsonl",
    "reports/paired_onehop/heldout.jsonl", "reports/paraphrase_onehop/heldout.jsonl",
    "reports/locked_final/chains.jsonl", "reports/locked_final/stress.jsonl", "reports/locked_final/transforms.jsonl",
    "data/processed/spatial_locked.jsonl", "data/processed/spatial_counterfactual_locked.jsonl",
    "data/processed/worlds_v1/eval_in_format.jsonl", "data/processed/worlds_v1/eval_heldout_table.jsonl",
    "data/processed/worlds_v1/counterfactual.jsonl",
    "data/processed/worlds_v2/eval_in_format.jsonl", "data/processed/worlds_v2/eval_heldout_table.jsonl",
    "data/processed/worlds_v2/counterfactual.jsonl",
]
EVAL_PAIRS = ["reports/paired_onehop/vertical_probe_pairs.json", "reports/paired_onehop/name_pairs.json",
              "reports/role_swap_onehop/name_pairs.json", "reports/paraphrase_onehop/name_pairs.json"]


def eval_states(root="."):
    """Map every evaluation state string to the files it occurs in (missing files are skipped)."""
    out = {}
    for rel in EVAL_JSONL:
        p = Path(root) / rel
        if p.exists():
            for line in open(p, encoding="utf-8"):
                out.setdefault(json.loads(line)["state"], set()).add(rel)
    for rel in EVAL_PAIRS:
        p = Path(root) / rel
        if p.exists():
            for item in json.loads(p.read_text(encoding="utf-8")):
                for k in ("numbered", "renamed"):
                    out.setdefault(item[k]["state"], set()).add(rel)
    return out
