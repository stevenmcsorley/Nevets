# Verified environment

The lab machine combination that all recorded results were produced with (checked 26 September 2026). Upgrade deliberately and re-run the parity checks below after any change.

| Component | Version | Notes |
|---|---|---|
| OS / GPU | Windows 11 Pro, NVIDIA RTX 3060 12 GB | WDDM driver: VRAM overflow spills to system RAM instead of raising OOM (see `reports/pt/throughput.md`); the owner is switching "CUDA Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" |
| Python | 3.12 (venv at `.venv`) | |
| PyTorch | **2.14.0+cu132** | bf16 autocast, fused AdamW |
| Triton | **triton-windows 3.5.1.post24** | community Windows build, required for `torch.compile`; pinned in `pyproject.toml` extra `pt` |
| tokenizers | 0.23.2 | legacy `data/tokenizer.json` (16k) and PT `tokenizers/pt_32k.json` (SHA-256 `229a91f7…`) |
| onnx / onnxruntime | 1.23.0 / 1.30.0 | browser: onnxruntime-web 1.22.0 (CDN) |
| python-chess | 1.11.2 | Stockfish 19 binary under `tools/` (not in git) |

Install: `pip install -e .[dev,pt,export]` inside the venv.

## Parity checks to re-run after upgrades
- **Compiled vs eager** (`torch.compile` via triton-windows), recorded in `reports/pt/compile_parity.json`: pt_35m loss diff 2.8e-4, hidden relative diff 0.56%, gradient cosine 0.99996; pt_150m 4.2e-4, 0.89%, 0.99992. This is bf16 rounding-level agreement: compilation fuses kernels and reorders floating-point reductions, so bitwise equality is not expected.
- **PyTorch vs ONNX (browser)**: `scripts/export_onnx.py` reports; `tests/test_js_parity.py` for the JS tokenizer and packing.
