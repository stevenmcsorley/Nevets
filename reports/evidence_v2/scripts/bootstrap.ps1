$ErrorActionPreference = "Stop"
Write-Host "SystemOne Lab bootstrap (Windows/NVIDIA)"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python 3.11-3.14 is required." }
python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,15), sys.version"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Write-Warning "nvidia-smi not found; GPU training may not be available." }
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Current stable line as of 2026-09-24. cu132 is preferred for modern NVIDIA drivers.
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
pip install -e ".[dev]"
python scripts/doctor.py
Write-Host "Bootstrap complete. Activate later with: .\.venv\Scripts\Activate.ps1"
