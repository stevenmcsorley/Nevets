from dataclasses import dataclass
from typing import Any
import json
import torch

@dataclass
class QuestionLayout:
    key: str
    qtype: str
    option_keys: list[str]
    option_end_positions: list[int]
    decide_position: int

@dataclass
class PackedRequest:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    branch_ids: torch.Tensor
    layouts: list[QuestionLayout]


def _text(x: Any) -> str:
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pack_request(tokenizer, state: Any, questions: dict[str, dict], device=None) -> PackedRequest:
    ids: list[int] = [tokenizer.id("<bos>"), tokenizer.id("<state>")]
    branch_ids: list[int] = [0, 0]
    positions: list[int] = [0, 1]
    state_tokens = tokenizer.encode(_text(state))
    ids.extend(state_tokens)
    branch_ids.extend([0] * len(state_tokens))
    positions.extend(range(2, 2 + len(state_tokens)))
    ids.append(tokenizer.id("</state>"))
    branch_ids.append(0)
    positions.append(2 + len(state_tokens))
    state_len = len(ids)

    layouts: list[QuestionLayout] = []
    for b, (key, q) in enumerate(questions.items(), start=1):
        qtype = q["type"].lower()
        local: list[int] = [tokenizer.id("<q>"), tokenizer.id(f"<{qtype}>")]
        local += tokenizer.encode(_text(q.get("instructions", "")))
        option_keys: list[str] = []
        options: list[str] = []
        if qtype == "noul":
            option_keys = ["false", "true"]
            options = ["No", "Yes"]
            crit = q.get("criteria")
            if crit:
                local += tokenizer.encode(" Criteria: " + _text(crit))
        elif qtype == "choice":
            crit = q.get("criteria", q.get("options", {}))
            if isinstance(crit, dict):
                option_keys = list(crit.keys())
                options = [_text(v) for v in crit.values()]
            elif isinstance(crit, list):
                option_keys = [str(i) for i in range(len(crit))]
                options = [_text(v) for v in crit]
            else:
                raise ValueError("choice criteria must be dict or list")
        elif qtype == "score":
            levels = q.get("criteria", q.get("levels", []))
            if isinstance(levels, dict):
                option_keys = list(levels.keys())
                options = [_text(v) for v in levels.values()]
            else:
                option_keys = [str(i) for i in range(len(levels))]
                options = [_text(v) for v in levels]
        else:
            raise ValueError(f"Unknown question type: {qtype}")

        option_end_positions: list[int] = []
        for opt in options:
            local.append(tokenizer.id("<opt>"))
            local.extend(tokenizer.encode(opt))
            local.append(tokenizer.id("</opt>"))
            option_end_positions.append(len(ids) + len(local) - 1)
        local.append(tokenizer.id("<decide>"))
        decide_position = len(ids) + len(local) - 1
        local.append(tokenizer.id("</q>"))

        ids.extend(local)
        branch_ids.extend([b] * len(local))
        # Reset position numbering for every question branch immediately after state.
        positions.extend(range(state_len, state_len + len(local)))
        layouts.append(QuestionLayout(key, qtype, option_keys, option_end_positions, decide_position))

    return PackedRequest(
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(branch_ids, dtype=torch.long, device=device),
        layouts,
    )


def branch_attention_mask(branch_ids: torch.Tensor) -> torch.Tensor:
    """Boolean [T,T] mask. True means attention is permitted."""
    T = branch_ids.numel()
    idx = torch.arange(T, device=branch_ids.device)
    qi = idx[:, None]
    kj = idx[None, :]
    qbranch = branch_ids[:, None]
    kbranch = branch_ids[None, :]
    causal = kj <= qi
    state_key = kbranch == 0
    same_branch = (qbranch == kbranch) & (qbranch != 0)
    state_query = qbranch == 0
    allow = torch.where(state_query, causal & state_key, state_key | (same_branch & causal))
    return allow
