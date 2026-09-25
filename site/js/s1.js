// Browser port of the SystemOne request pipeline: byte-level BPE tokenizer (with character offsets),
// request packing (shared state + isolated question branches), attention mask and ONNX scoring.
// Mirrors src/systemone_lab/{tokenizer,formatting}.py; tests/js_parity checks it token-for-token.

const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice(); let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const map = {}; bs.forEach((b, i) => { map[b] = String.fromCodePoint(cs[i]); }); return map;
}

export class Tokenizer {
  constructor(json) {
    this.vocab = json.model.vocab;
    this.ranks = new Map();
    json.model.merges.forEach((m, i) => this.ranks.set(Array.isArray(m) ? m.join(" ") : m, i));
    this.special = {}; for (const t of json.added_tokens) this.special[t.content] = t.id;
    this.byteMap = bytesToUnicode(); this.cache = new Map(); this.enc = new TextEncoder();
  }
  id(token) { const v = this.special[token] ?? this.vocab[token]; if (v === undefined) throw new Error(`unknown token ${token}`); return v; }
  bpe(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1, rank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < rank) { rank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1], merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i++; } else merged.push(parts[i]);
      }
      parts = merged;
    }
    this.cache.set(word, parts); return parts;
  }
  // Returns ids and [lo, hi) offsets into `text` (UTF-16 indices, like Python str indices for BMP text).
  encode(text) {
    const ids = [], offsets = [];
    for (const m of text.matchAll(PRETOKENIZE)) {
      const chunk = m[0], start = m.index;
      const charOfByte = []; let ci = 0;
      for (const ch of chunk) { const nb = this.enc.encode(ch).length; for (let k = 0; k < nb; k++) charOfByte.push(ci); ci += ch.length; }
      const mapped = Array.from(this.enc.encode(chunk), b => this.byteMap[b]).join("");
      let b = 0;
      for (const piece of this.bpe(mapped)) {
        const nbytes = Array.from(piece).length, id = this.vocab[piece];
        if (id === undefined) throw new Error(`piece not in vocab: ${piece}`);
        const lo = start + charOfByte[b], last = b + nbytes - 1;
        const hi = start + charOfByte[last] + (chunk.codePointAt(charOfByte[last]) > 0xffff ? 2 : 1);
        ids.push(id); offsets.push([lo, hi]); b += nbytes;
      }
    }
    return { ids, offsets };
  }
}

function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

function entityStatePositions(stateText, offsets, name, first) {
  const out = [], re = new RegExp(`(?<![\\w-])${escapeRe(name)}(?![\\w-])`, "g");
  for (const m of stateText.matchAll(re)) {
    const lo = m.index, hi = m.index + m[0].length; let last = -1;
    offsets.forEach(([a, b], i) => { if (a < hi && b > lo) last = i; });
    if (last >= 0) out.push(first + last);
  }
  return out;
}

// Pack one request (choice questions). Same layout as formatting.pack_request (causal options).
export function pack(tok, state, questions) {
  const ids = [tok.id("<bos>"), tok.id("<state>")], branch = [0, 0], pos = [0, 1];
  const st = tok.encode(state);
  st.ids.forEach((t, i) => { ids.push(t); branch.push(0); pos.push(2 + i); });
  ids.push(tok.id("</state>")); branch.push(0); pos.push(2 + st.ids.length);
  const stateLen = ids.length, layouts = [];
  Object.entries(questions).forEach(([key, q], qi) => {
    const b = qi + 1, local = [tok.id("<q>"), tok.id(`<${q.type.toLowerCase()}>`)];
    local.push(...tok.encode(q.instructions).ids);
    let entities = q.query_entities;
    if (!entities) { const m = q.instructions.match(/^What is the spatial relation of (.+) to (.+)\?$/); if (m) entities = [m[1], m[2]]; }
    const statePositions = entities ? entities.map(n => entityStatePositions(state, st.offsets, n, 2)) : null;
    const keys = Object.keys(q.criteria), optionEnds = [];
    for (const k of keys) {
      local.push(tok.id("<opt>"), ...tok.encode(String(q.criteria[k])).ids, tok.id("</opt>"));
      optionEnds.push(ids.length + local.length - 1);
    }
    local.push(tok.id("<decide>")); const decide = ids.length + local.length - 1; local.push(tok.id("</q>"));
    local.forEach((t, i) => { ids.push(t); branch.push(b); pos.push(stateLen + i); });
    layouts.push({ key, keys, optionEnds, decide, statePositions });
  });
  return { ids, pos, branch, layouts };
}

// Boolean [T,T] mask (1 = may attend), same rule as formatting.branch_attention_mask.
export function mask(branch, bidirectionalState) {
  const T = branch.length, m = new Uint8Array(T * T);
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    const causal = j <= i, stateKey = branch[j] === 0;
    const ok = branch[i] === 0 ? (bidirectionalState ? stateKey : causal && stateKey)
                               : stateKey || (branch[i] === branch[j] && causal);
    m[i * T + j] = ok ? 1 : 0;
  }
  return m;
}

// Score every question of a packed request with an onnxruntime-web session; returns {key: {option: p}}.
export async function score(ort, session, packed, bidirectionalState) {
  const T = packed.ids.length, i64 = a => BigInt64Array.from(a, BigInt);
  const base = {
    input_ids: new ort.Tensor("int64", i64(packed.ids), [1, T]),
    position_ids: new ort.Tensor("int64", i64(packed.pos), [1, T]),
    mask: new ort.Tensor("bool", mask(packed.branch, bidirectionalState), [1, T, T]),
  };
  const out = {};
  for (const l of packed.layouts) {
    const feeds = { ...base, decide: new ort.Tensor("int64", i64([l.decide]), [1]),
                    options: new ort.Tensor("int64", i64(l.optionEnds), [l.optionEnds.length]) };
    if (session.inputNames.includes("bind")) {
      const w = new Float32Array(T); const [a, b] = l.statePositions || [[], []];
      if (a.length && b.length) { a.forEach(p => { w[p] += 1 / a.length; }); b.forEach(p => { w[p] -= 1 / b.length; }); }
      feeds.bind = new ort.Tensor("float32", w, [T]);
    }
    const logits = (await session.run(feeds)).logits.data, mx = Math.max(...logits);
    const e = Array.from(logits, v => Math.exp(v - mx)), z = e.reduce((s, v) => s + v, 0);
    out[l.key] = Object.fromEntries(l.keys.map((k, i) => [k, e[i] / z]));
  }
  return out;
}
