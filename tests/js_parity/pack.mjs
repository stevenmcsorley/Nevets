// Pack requests with the browser pipeline (site/js/s1.js) so pytest can compare against Python.
import { readFileSync } from "node:fs";
import { Tokenizer, pack, mask } from "../../site/js/s1.js";
const [tokPath, casesPath] = process.argv.slice(2);
const tok = new Tokenizer(JSON.parse(readFileSync(tokPath, "utf8")));
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
const out = cases.map(c => {
  if (c.text !== undefined) { const e = tok.encode(c.text); return { ids: e.ids, offsets: e.offsets }; }
  const p = pack(tok, c.state, c.questions);
  const m = mask(p.branch, c.bidirectional);
  return { ids: p.ids, pos: p.pos, branch: p.branch, mask_sum: m.reduce((s, v) => s + v, 0),
           mask_rows: Array.from({ length: p.ids.length }, (_, i) => m.subarray(i * p.ids.length, (i + 1) * p.ids.length).reduce((s, v) => s + v, 0)),
           layouts: p.layouts.map(l => ({ keys: l.keys, optionEnds: l.optionEnds, decide: l.decide, statePositions: l.statePositions })) };
});
process.stdout.write(JSON.stringify(out));
