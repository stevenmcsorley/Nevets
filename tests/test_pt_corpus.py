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


def test_pretraining_structured_slice_excludes_p3_heldout_formats():
    import build_corpus as bc
    assert set(bc.P3_HELDOUT_FORMATS) == {"table", "symbolic"}
    texts = bc.structured_samples(300, 7, for_pretraining=True)
    assert not any("subject | relation | object" in t for t in texts)


def test_reverse_decontam_flags_question_and_passage_copies(tmp_path):
    import subprocess, sys
    ev = tmp_path / "ev.jsonl"; corp = tmp_path / "corp"; corp.mkdir()
    passage = ("The mitochondrion is a double membrane bound organelle found in most eukaryotic organisms and it "
               "generates most of the supply of adenosine triphosphate used as a source of chemical energy")
    recs = [{"id": "copied", "state": passage, "questions": {"q": {"type": "noul", "instructions": "Is the mitochondrion found in most eukaryotic organisms?"}}},
            {"id": "clean", "state": "Zorblat quenched the vivid lamp near seventeen purple otters by the river bend today",
             "questions": {"q": {"type": "noul", "instructions": "Did zorblat quench anything near the purple otters?"}}}]
    ev.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (corp / "a.jsonl").write_text(json.dumps({"text": "Biology notes. " + passage + ". Question: is the mitochondrion found in most eukaryotic organisms?"}) + "\n", encoding="utf-8")
    out = tmp_path / "rep.json"
    subprocess.run([sys.executable, "scripts/pt/reverse_decontam.py", "--eval", str(ev), "--corpus", str(corp / "*.jsonl"), "--out", str(out)], check=True, capture_output=True)
    rep = json.loads(out.read_text())
    assert rep["contaminated_items"] == 1 and "copied" in rep["items"] and "clean" not in rep["items"]
    assert rep["items"]["copied"]["question"] >= 1 and rep["items"]["copied"]["ngram13"] >= 1
