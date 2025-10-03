# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.1.26] - 2025-10-03
### Added
- feat(strategy): Volatility-aware entry/exit for `FreqAIHybridExample` using ATR(14) + EMA(200) trend filter.
  - Indicators: Add `atr`, `ema200`, and derive `atr_pct = atr/close`, `pred_ret = (&-prediction-close)/close`.
  - Entry (regression): `pred_ret > fee_buffer + 0.5*atr_pct` AND `close > ema200` (gated by `do_predict`).
  - Exit (regression): `pred_ret < -(fee_buffer + 0.5*atr_pct)` (gated by `do_predict`).
  - Classification thresholds unchanged.
  - Enable basic trailing stop: 1% activation offset, 0.5% trail.

## [0.1.27] - 2025-10-04
### Added
- feat(strategy): Enable short entries/exits for `FreqAIHybridExample`.
  - Set `can_short=True`.
  - Mirrored regression logic around ATR-adjusted thresholds and EMA200 trend: short when `pred_ret < -(fee_buffer + 0.5*atr_pct)` and `close < ema200`; exit short when the opposite holds.
  - Mirrored classification logic using probability thresholds (0.45/0.55).

## [0.1.25] - 2025-10-03
### Performance
- perf(freqai): Add a fast GPU profile for backtesting/training.
  - `freqtrade_user_data/config.json`:
    - Increase `batch_size` to 128 and set `num_workers` to 4 (match typical 4 vCPU hosts); enable `compile_model` for `torch.compile` speedups.
    - Reduce `epochs` to 10 with `early_stopping_patience=4` for quicker iterations.
    - Shrink sequence and dataset: `window_size=48`, `stride=2`.
    - Smaller model: `lstm_hidden=96`, `lstm_layers=1`, `d_model=96`, `nhead=2`, `ff_dim=256`.
    - Trim indicator load: `indicator_periods_candles` to `[14]`.
  - `docker-compose.yml`:
    - Narrow default `TIMERANGE` to `20240101-20240201` for faster local cycles and reduce `BLOCKS` to `1`.
    - Apply the same shortened `TIMERANGE` to `freqai-backtest-gpu-l4`.
  - `hybrid_lstm_transformer_crypto.py`:
    - Parallelize FreqAI inference DataLoader with `num_workers=cfg.num_workers`, enable CUDA `pin_memory` and `persistent_workers`.

## [0.1.4] - 2025-10-03
### Fixed
- freqai(config): Move `pairlists` to top-level in `freqtrade_user_data/config.json` and keep `pair_whitelist`/`pair_blacklist` under `exchange` to satisfy Freqtrade schema and fix startup validation error.

## [0.1.5] - 2025-10-03
### Fixed
- freqai(config): Add mandatory `entry_pricing` and `exit_pricing` sections to `freqtrade_user_data/config.json` to satisfy stricter schema/validation and prevent KeyError on startup.

## [0.1.6] - 2025-10-03
### Fixed
- freqai(strategy): Update strategy to import `IStrategy` from `freqtrade.strategy.interface` and drop deprecated `IFreqaiInterface` import/usage. Aligns with current Freqtrade FreqAI integration where `self.freqai` is injected by the framework.

## [0.1.7] - 2025-10-03
### Fixed
- freqai(compose): Prepend `freqtrade download-data` in `freqai-train-gpu-l4` service to fetch OHLCV before backtesting, preventing "No data found" errors.

## [0.1.8] - 2025-10-03
### Fixed
- freqai(compose): Simplify command to single-line `bash -lc` without line-continuation to avoid YAML folding issues causing `--config`/`&&` parsing errors.

## [0.1.9] - 2025-10-03
### Fixed
- freqai(imports): Import `BasePyTorchModel` from `freqtrade.freqai.base_models.BasePyTorchModel` to avoid subclassing the module object, fixing `TypeError: module() takes at most 2 arguments (3 given)` during FreqAI model class definition.

## [0.1.10] - 2025-10-03
### Fixed
- freqai(resolver): Make `freqtrade_user_data/freqaimodels/HybridTimeseriesFreqAIModel.py` define a local subclass of the core model so `__module__` matches the file, satisfying Freqtrade's resolver check and allowing the class to be discovered.

## [0.1.11] - 2025-10-03
### Chore
- freqai(compose):
  - Set `PYTHONWARNINGS=ignore::ImportWarning` to hide noisy SixMetaPathImporter warnings.
  - Prepend `pip install -U six` in `freqai-train-gpu-l4` to upgrade `six` at runtime.
  - Optionally install `stable-baselines3` and `sb3-contrib` (non-blocking) to suppress RL model import warnings.

## [0.1.12] - 2025-10-03
### Fixed
- freqai(compose): Switch to YAML list-form `command` with `bash -lc` and a folded script to avoid shell quoting issues that caused `syntax error: unexpected end of file`.

## [0.1.13] - 2025-10-03
### Fixed
- freqai(model): Inherit from `BasePyTorchRegressor` and implement `data_convertor` to satisfy abstract interface (`train` provided by base, `fit/predict` implemented here).

## [0.1.14] - 2025-10-03
### Fixed
- freqai(strategy): Simplify target to `shift(-label_period)` (remove rolling mean) to reduce label NaNs during training.
- freqai(compose): Download a wider data range via `DOWNLOAD_TIMERANGE` (defaults to start a year earlier) to ensure enough pre-backtest history for FreqAI training windows.

## [0.1.15] - 2025-10-03
### Added
- scripts: `scripts/freqtrade_download_blocks.sh` to download OHLCV in fixed-size blocks (default 30 days).
### Changed
- freqai(compose): Use block-based download by calling the new script with env-configurable `TIMEFRAMES`, `DOWNLOAD_START`, `DOWNLOAD_END`, and `BLOCK_DAYS` (default 30).

## [0.1.16] - 2025-10-03
### Added
- scripts: `scripts/freqtrade_download_recent_blocks.sh` — simplest mode using only `BLOCKS` (N×30 days) starting from today backward.
### Changed
- freqai(compose): Switch to the recent-block script driven by `BLOCKS` and `BLOCK_DAYS` (defaults 2 and 30) to reduce configuration complexity.

## [0.1.17] - 2025-10-03
### Added
- scripts: `scripts/freqtrade_download_prestart_blocks.sh` — downloads N×block_days ending at the backtest start (from `TIMERANGE`) to ensure FreqAI has pre-history for training.
### Changed
- freqai(compose): Always download the full `TIMERANGE` and then pre-start blocks based on `BLOCKS`/`BLOCK_DAYS` (default `BLOCKS=3`) to guarantee coverage before training.

## [0.1.18] - 2025-10-03
### Fixed
- scripts: Robust date math for block downloaders — convert YYYYMMDD to ISO (YYYY-MM-DD) before `date -d` operations to ensure correct ranges on Ubuntu base images.

## [0.1.19] - 2025-10-03
### Fixed
- freqai(model): Replace call to removed `dk.make_test_data_point` with supported `dk.filter_features(..., training_filter=False)` and `dk.feature_pipeline.transform(..., outlier_check=True)` in `HybridTimeseriesFreqAIModel.predict()`. Build sliding windows across the backtest slice, batch-infer, align predictions back to full length, and return `do_predict`/`DI_values` from the pipeline. Fixes `AttributeError: 'FreqaiDataKitchen' object has no attribute 'make_test_data_point'` during backtesting.

## [0.1.20] - 2025-10-03
### Fixed
- freqai(model): Unwrap saved Trainer wrapper to the underlying `nn.Module` inside `predict()` and normalize device handling to `torch.device`. This fixes `AttributeError` at `model.eval()` when backtesting loads the saved object. Also guard access to the optional `di` pipeline step to avoid noisy warnings when `DI_threshold` is not configured.

## [0.1.21] - 2025-10-03
### Added
- compose(webui): Add `freqtrade-webui` service exposing FreqUI on `http://localhost:8080`, mounting `freqtrade_user_data` so the UI can access the same configuration. Enable API server in `freqtrade_user_data/config.json`. Backtesting service now exports trades JSON to `/freqtrade/user_data/backtest_results/freqai_trades.json` for easy viewing via FreqUI Backtesting page (Upload Results).

## [0.1.22] - 2025-10-03
### Fixed
- webui: Install Freqtrade UI assets at container start (`freqtrade install-ui`) and persist them in a named volume (`freqtrade_ui_data`). Fixes "Freqtrade UI not installed" error when opening the Web UI.

## [0.1.23] - 2025-10-03
### Fixed
- strategy(freqai): Guard ADX/RSI/EMA feature generation for short slices in `FreqAIHybridExample.feature_engineering_expand_all` to avoid `ValueError: negative dimensions are not allowed` when the UI requests small data windows (e.g., charts). ADX now returns NaN for insufficient length; other indicators fall back safely.

## [0.1.24] - 2025-10-03
### Changed
- freqai(model): Rename class to `HybridTimeseriesFreqAIModel_tinhn` and add a discovery wrapper at `freqtrade_user_data/freqaimodels/HybridTimeseriesFreqAIModel_tinhn.py`. Updated config (`freqai.model_classname`) and compose `--freqaimodel` flags accordingly. Kept backward-compatible alias in the legacy wrapper.

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
- `docker-compose.yml`: Tag the FreqAI training service with an image name and reserve NVIDIA GPUs via `deploy.resources`.
- `freqtrade_user_data/config.json`: Add `pair_whitelist`/`pair_blacklist` to satisfy StaticPairList requirements.
- `hybrid_lstm_transformer_crypto.py`: Reworked `HybridTimeseriesFreqAIModel` to use official FreqAI fit/predict signatures, convert feature DataFrames into windows, and align predictions/DI output with FreqAI expectations.
### Fixed
- fix(webui): Remove unsupported `--freqaimodel` from `webserver` command and set top-level `freqaimodel` in `freqtrade_user_data/config.json` so WebUI can load the FreqAI model during `pair_history` without CLI args.
- fix(freqai): Bump `freqai.identifier` to `BTC_fast` to avoid feature-list mismatch with previously trained models after feature changes (e.g., trimmed indicator periods). Prevents OperationalException during WebUI Pair History.
