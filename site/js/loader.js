// Load onnxruntime-web, the tokenizer and an ONNX model, reporting download progress.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";
import { Tokenizer } from "./s1.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.numThreads = 1;  // GitHub Pages cannot send the COOP/COEP headers threads require

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(), chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); got += value.length; onProgress?.(got / total);
  }
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

export async function loadModel(modelUrl, onProgress) {
  const [tokJson, bytes] = await Promise.all([
    fetch("model/tokenizer.json").then(r => r.json()),
    fetchWithProgress(modelUrl, onProgress),
  ]);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return { ort, session, tok: new Tokenizer(tokJson) };
}
