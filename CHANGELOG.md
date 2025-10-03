# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.1.2] - 2025-10-03
### Changed
- `scripts/setup_nvidia_l4_cuda_docker_ubuntu.sh`: Switch installer to `INSTALL_MODE=binary` for LTS branch (535) and update notes.
 - `freqtrade_user_data/config.json`: Conform FreqAI config to required keys (`model_classname`, `train_period_days`, `backtest_period_days`, `feature_parameters`, `data_split_parameters`, `purge_old_models`).
 - `docker-compose.yml`: Pass `--freqaimodel HybridTimeseriesFreqAIModel` explicitly.

### Added
- `freqtrade_user_data/freqaimodels/HybridTimeseriesFreqAIModel.py`: Thin wrapper exposing our model class to FreqAI discovery.

## [0.1.1] - 2025-10-03
### Added
- `scripts/setup_nvidia_l4_cuda_docker_ubuntu.sh` — Host setup script for Ubuntu (22.04/24.04) on GCP to install NVIDIA L4 (535 LTS) drivers, CUDA toolkit, Docker, and NVIDIA Container Toolkit, with docker group configuration.
 - `docker/Dockerfile.freqtrade.gpu.x86` — GPU-enabled image with Freqtrade + FreqAI on top of PyTorch CUDA runtime.
 - `docker-compose.yml` service `freqai-train-gpu-l4` — Runs Freqtrade backtesting with GPU, imports our custom FreqAI model via `PYTHONPATH`.
 - `freqtrade_user_data/config.json` — Minimal FreqAI config referencing our model.
 - `freqtrade_user_data/strategies/FreqAIHybridExample.py` — Example strategy consuming FreqAI predictions.
 - `.env` — Defaults for compose (epochs, horizon, batch_size, timerange, etc.).

## [0.1.0] - 2025-10-03
### Added
- `hybrid_lstm_transformer_crypto.py` — Hybrid LSTM + Transformer encoder for crypto time-series forecasting (BTC OHLCV).
  - Data: Fetch BTC/USDT hourly via ccxt; fallback to yfinance and pandas_datareader with logging.
  - Indicators: RSI(14), EMA(12), EMA(26); NaN handling.
  - Windows: Sliding windows (configurable window, stride, horizon 1..5).
  - Model: Stacked LSTM → PositionalEncoding → TransformerEncoder → pooling → regression/classification head.
  - Training: Mixed precision on CUDA, ReduceLROnPlateau, early stopping, gradient clipping, reproducibility.
  - Dataloaders: pin_memory on CUDA, persistent_workers, optional variable lengths + masks.
  - Evaluation: Test MAE/RMSE, predicted vs actual plot; saves best checkpoint and metrics JSON.
  - Inference: `predict()` scales, masks, inverse-transforms; batch inference.
  - Attention: Optional self-attention capture and heatmap.
  - FreqAI: Example BasePyTorchModel-compatible wrapper and config snippet.

### DevOps
- `docker/Dockerfile.gpu.x86` — GPU-enabled Dockerfile (x86, NVIDIA L4/SM 8.9) using PyTorch CUDA runtime.
- `docker-compose.yml` — Compose service `crypto-train-gpu-l4` to train with GPU and map artifacts.
# Changelog
# Changelog
# Changelog

## [0.1.3] - 2025-10-03
### Changed
- `docker/Dockerfile.freqtrade.gpu.x86`: Bump Freqtrade to `2025.9` (with `freqai,plot` extras) and switch base image to Python 3.11 (`mosaicml/pytorch:2.3.1_cu121-python3.11-ubuntu20.04-aws`).
