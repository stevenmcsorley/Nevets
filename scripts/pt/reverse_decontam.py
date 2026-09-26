"""Reverse decontamination: check an evaluation set built AFTER the corpus against the corpus.

Required for every eval set created after the PT corpus build (PT-5, and anything derived from
BoolQ/Wikipedia-like sources that the web corpus may contain). For each eval item, anchors are:
  - the normalised question text (question-level check; catches paraphrase-free copies), and
  - the normalised state/passage word 13-grams (n-gram check).
All anchors go into one Aho-Corasick automaton, so the corpus is scanned once, in linear time.
Every hit is reported per item; items with any hit must be dropped from the eval set (or the
matching documents removed from the corpus before training). Output: a JSON report.
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import ahocorasick

N = 13
norm = lambda s: " ".join(re.findall(r"[a-z0-9_]+", s.lower()))


def anchors(rec, min_q_words=6):
    out = []
    for q in rec.get("questions", {}).values():
        t = norm(q.get("instructions", ""))
        if len(t.split()) >= min_q_words: out.append(("question", t))
    w = norm(rec.get("state", "") if isinstance(rec.get("state"), str) else json.dumps(rec.get("state"))).split()
    out += [("ngram13", " ".join(w[i:i + N])) for i in range(0, max(0, len(w) - N + 1))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="jsonl evaluation file built after the corpus")
    ap.add_argument("--corpus", default="data/pt/clean/*.jsonl"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = ahocorasick.Automaton(); owners = defaultdict(set); recs = [json.loads(l) for l in open(a.eval, encoding="utf-8")]
    for i, r in enumerate(recs):
        for kind, t in anchors(r):
            key = " " + t + " "  # word-boundary padding
            owners[key].add((i, kind)); A.add_word(key, key)
    A.make_automaton()
    hits = defaultdict(lambda: {"question": 0, "ngram13": 0, "docs": 0}); docs = 0
    for f in sorted(glob.glob(a.corpus)):
        for line in open(f, encoding="utf-8"):
            docs += 1; text = " " + norm(json.loads(line)["text"]) + " "; seen = set()
            for _, key in A.iter(text):
                for i, kind in owners[key]:
                    hits[i][kind] += 1
                    if i not in seen: hits[i]["docs"] += 1; seen.add(i)
    report = {"eval": a.eval, "items": len(recs), "corpus_docs": docs, "contaminated_items": len(hits),
              "question_level_hits": sum(1 for h in hits.values() if h["question"]),
              "ngram13_hits": sum(1 for h in hits.values() if h["ngram13"]),
              "items": {recs[i].get("id", str(i)): h for i, h in list(hits.items())[:500]}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "items"}))


if __name__ == "__main__":
    main()
