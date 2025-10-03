# Repository Guidelines

## Project Structure & Module Organization
- Core script: `hybrid_lstm_transformer_crypto.py` (training, eval, inference, FreqAI wrapper).
- Docker: `docker/` (GPU Dockerfiles), Compose: `docker-compose.yml`.
- Ops scripts: `scripts/` (e.g., NVIDIA L4 + Docker setup).
- Outputs: `./artifacts/` (created at runtime). Versioning: `CHANGELOG.md`.

## Build, Test, and Development Commands
- Local run (regression):
  - `python hybrid_lstm_transformer_crypto.py --epochs 10`
- Smoke test (quick):
  - `python hybrid_lstm_transformer_crypto.py --smoke_test`
- Multi‑step horizon:
  - `python hybrid_lstm_transformer_crypto.py --horizon 3 --epochs 30`
- Docker (GPU, L4):
  - `docker compose up --build crypto-train-gpu-l4`
- FreqAI via Freqtrade (GPU):
  - Ensure `../user_data/config.json` points to `module_path: "hybrid_lstm_transformer_crypto"`, `class_name: "HybridTimeseriesFreqAIModel"`.
  - `docker compose up --build freqai-train-gpu-l4`

## Coding Style & Naming Conventions
- Python 3.9+, PEP 8, 4‑space indentation, type hints required for new/edited functions.
- Names: modules `lower_snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.
- Keep changes minimal and focused; avoid introducing new top‑level dependencies unless necessary.
- Prefer small, testable helpers over long functions. Add concise docstrings.

## Testing Guidelines
- No formal test suite yet. Use `--smoke_test` for quick verification.
- If adding tests, use `pytest` with files named `tests/test_*.py`; mock network I/O (ccxt/yfinance) where possible.
- Ensure deterministic behavior by seeding (`--seed`) and avoiding time‑dependent code in tests.

## Commit & Pull Request Guidelines
- Follow Conventional Commits seen in history: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `perf:`, scopes allowed (e.g., `feat(freqai):`).
- Commits: small, descriptive messages; one logical change per commit.
- PRs: include summary, motivation, linked issues, run instructions, and screenshots/plots (e.g., `prediction_plot.png`) when relevant.

## Security & Configuration Tips
- Do not commit secrets or API keys. `.dockerignore` and `.gitignore` already exclude artifacts.
- For GPU runs, ensure NVIDIA drivers + `nvidia-container-toolkit` are installed (see `scripts/setup_nvidia_l4_cuda_docker_ubuntu.sh`).
