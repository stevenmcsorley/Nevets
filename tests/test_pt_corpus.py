import json
import re
import sys

sys.path.insert(0, "scripts/pt")


def test_decontamination_catches_eval_text_and_passes_web_text():
    import build_corpus as bc
    grams, short, n = bc.eval_ngrams()
    assert n > 1000 and grams
    long_state = next(json.loads(l)["state"] for l in open("reports/chain/eval_hops.jsonl", encoding="utf-8")
                      if len(bc.words(json.loads(l)["state"])) >= 20)
    doc = "Here is a puzzle from a forum. " + long_state + " Can anyone solve it?"
    w = bc.words(doc)
    assert any(hash(" ".join(w[i:i + bc.N])) in grams for i in range(len(w) - bc.N + 1))
    web = ("Photosynthesis converts light energy into chemical energy stored in glucose. Plants, algae and some "
           "bacteria use chlorophyll to capture sunlight, releasing oxygen as a by-product of splitting water.")
    w = bc.words(web)
    assert not any(hash(" ".join(w[i:i + bc.N])) in grams for i in range(len(w) - bc.N + 1))
    assert not any(s in " ".join(w) for s in short)
