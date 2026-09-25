from dataclasses import dataclass
from typing import Any
import json
import re
import torch

@dataclass
class QuestionLayout:
    key: str
    qtype: str
    option_keys: list[str]
    option_end_positions: list[int]
    decide_position: int
    query_entity_token_ids: tuple[list[int], list[int]] | None = None
    query_entity_state_positions: tuple[list[int], list[int]] | None = None

@dataclass
class PackedRequest:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    branch_ids: torch.Tensor
    layouts: list[QuestionLayout]
    option_ids: torch.Tensor | None = None  # 0 = not an option token; k = k-th option of its branch


def _text(x: Any) -> str:
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def entity_state_positions(state_text: str, offsets: list[tuple[int, int]], name: str,
                           first_position: int) -> list[int]:
    """Packed positions of the last state token of each whole-word occurrence of name."""
    positions: list[int] = []
    for match in re.finditer(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", state_text):
        overlapping = [i for i, (lo, hi) in enumerate(offsets) if lo < match.end() and hi > match.start()]
        if overlapping:
            positions.append(first_position + overlapping[-1])
    return positions


def pack_request(tokenizer, state: Any, questions: dict[str, dict], device=None,
                 isolate_options: bool = False) -> PackedRequest:
    """Pack a shared state and isolated question branches into one sequence.

    With isolate_options, every option starts at the same position id right after the question
    text and (with the matching mask) sees only the state, the question text and itself, and the
    decide token sees no option. Probabilities are then invariant to the order options are listed.
    """
    ids: list[int] = [tokenizer.id("<bos>"), tokenizer.id("<state>")]
    option_ids: list[int] = [0, 0]
    branch_ids: list[int] = [0, 0]
    positions: list[int] = [0, 1]
    state_text = _text(state)
    state_tokens, state_offsets = tokenizer.encode_with_offsets(state_text)
    ids.extend(state_tokens)
    branch_ids.extend([0] * len(state_tokens))
    positions.extend(range(2, 2 + len(state_tokens)))
    ids.append(tokenizer.id("</state>"))
    branch_ids.append(0)
    positions.append(2 + len(state_tokens))
    option_ids.extend([0] * (len(ids) - len(option_ids)))
    state_len = len(ids)

    layouts: list[QuestionLayout] = []
    for b, (key, q) in enumerate(questions.items(), start=1):
        qtype = q["type"].lower()
        local: list[int] = [tokenizer.id("<q>"), tokenizer.id(f"<{qtype}>")]
        instruction = _text(q.get("instructions", ""))
        local += tokenizer.encode(instruction)
        query_entities = q.get("query_entities")
        if query_entities is None:
            match = re.fullmatch(r"What is the spatial relation of (.+) to (.+)\?", instruction)
            if match:
                query_entities = match.groups()
        query_entity_token_ids = None
        query_entity_state_positions = None
        if query_entities is not None:
            if len(query_entities) != 2 or not all(isinstance(x, str) and x for x in query_entities):
                raise ValueError("query_entities must contain two nonempty names")
            query_entity_token_ids = tuple(tokenizer.encode(x) for x in query_entities)
            if not all(query_entity_token_ids):
                raise ValueError("query entity has no tokens")
            query_entity_state_positions = tuple(
                entity_state_positions(state_text, state_offsets, x, 2) for x in query_entities)
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

        if not option_keys or len(set(option_keys)) != len(option_keys):
            raise ValueError("candidates must be nonempty and uniquely named")
        option_end_positions: list[int] = []
        head = len(local); local_opt = [0] * head; local_pos = list(range(head))
        for k, opt in enumerate(options, start=1):
            toks = [tokenizer.id("<opt>")] + tokenizer.encode(opt) + [tokenizer.id("</opt>")]
            start = head if isolate_options else len(local)
            local.extend(toks); local_opt.extend([k] * len(toks)); local_pos.extend(range(start, start + len(toks)))
            option_end_positions.append(len(ids) + len(local) - 1)
        tail = head if isolate_options else len(local)
        local.append(tokenizer.id("<decide>")); local_opt.append(0); local_pos.append(tail)
        decide_position = len(ids) + len(local) - 1
        local.append(tokenizer.id("</q>")); local_opt.append(0); local_pos.append(tail + 1)

        ids.extend(local)
        branch_ids.extend([b] * len(local))
        option_ids.extend(local_opt)
        # Reset position numbering for every question branch immediately after state.
        positions.extend(state_len + p for p in local_pos)
        layouts.append(QuestionLayout(key, qtype, option_keys, option_end_positions,
                                      decide_position, query_entity_token_ids,
                                      query_entity_state_positions))

    return PackedRequest(
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(branch_ids, dtype=torch.long, device=device),
        layouts,
        torch.tensor(option_ids, dtype=torch.long, device=device),
    )


def branch_attention_mask(branch_ids: torch.Tensor, bidirectional_state: bool = False,
                          option_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean [T,T] mask. True means attention is permitted.

    With bidirectional_state, state tokens attend to the whole state (the state is never generated,
    so it need not be causal). Question branches are unchanged either way.
    """
    T = branch_ids.size(-1)
    idx = torch.arange(T, device=branch_ids.device)
    qi = idx[:, None]
    kj = idx[None, :]
    qbranch = branch_ids[..., :, None]  # also accepts a [B,T] batch
    kbranch = branch_ids[..., None, :]
    causal = kj <= qi
    state_key = kbranch == 0
    same_branch = (qbranch == kbranch) & (qbranch != 0)
    if option_ids is not None:  # isolated options: never attend to a sibling option
        qopt, kopt = option_ids[..., :, None], option_ids[..., None, :]
        same_branch = same_branch & ((kopt == 0) | (kopt == qopt))
    state_query = qbranch == 0
    state_rows = state_key if bidirectional_state else causal & state_key
    allow = torch.where(state_query, state_rows, state_key | (same_branch & causal))
    return allow
