#!/usr/bin/env python3
"""
Hybrid LSTM + Transformer encoder for crypto time-series forecasting (BTC OHLCV).

Features
- Data: Fetch BTC/USDT hourly via ccxt; fallback to yfinance, then pandas_datareader.
- Indicators: RSI(14), EMA(12), EMA(26). NaN-safe.
- Windows: Sliding windows with configurable window size, stride, and horizon (1..5).
- Model: Stacked LSTM -> PositionalEncoding -> TransformerEncoder -> pooling -> head.
- Training: Mixed precision on CUDA, ReduceLROnPlateau, early stopping, grad clipping.
- Dataloaders: pin_memory on CUDA, persistent_workers, optional variable seq lens + masks.
- Evaluation: MAE/RMSE on test, prediction plot, save best checkpoint + metrics JSON.
- Inference: predict() handles scaling, masking, batch inference, inverse scaling.
- Attention (optional): Capture self-attention weights and plot heatmap.
- FreqAI: Example BasePyTorchModel-compatible class and config snippet.

Run
  python hybrid_lstm_transformer_crypto.py --epochs 2 --smoke_test

Optional deps install
  pip install ccxt yfinance pandas_datareader

"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler


# ---------------------------------------------
# Utils: seeds, device, precision
# ---------------------------------------------

def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
    return device


# ---------------------------------------------
# Data fetching with fallbacks
# ---------------------------------------------

def _print_optional_dep_instructions() -> None:
    print("Optional dependencies: install with")
    print("  pip install ccxt yfinance pandas_datareader")


def fetch_btc_ccxt(since_ms: int, max_rows: int = 50000) -> Optional[pd.DataFrame]:
    try:
        import ccxt  # type: ignore
    except Exception as e:
        print(f"ccxt not available: {e}")
        _print_optional_dep_instructions()
        return None

    print("Attempting to fetch via ccxt (Binance, 1h candles)...")
    try:
        exchange = ccxt.binance({"enableRateLimit": True, "options": {"adjustForTimeDifference": True}})
        symbol = "BTC/USDT"
        timeframe = "1h"
        limit = 1000
        all_rows: List[List[float]] = []
        since = since_ms
        # Safety for rate limit
        rate_limit_sec = (getattr(exchange, "rateLimit", 1200) or 1200) / 1000.0
        while True:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            all_rows.extend(batch)
            since = batch[-1][0] + 1
            # Avoid hitting API too fast
            time.sleep(rate_limit_sec)
            if len(all_rows) >= max_rows:
                break
        if not all_rows:
            print("No data returned by ccxt.")
            return None
        df = pd.DataFrame(all_rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.set_index("datetime").sort_index()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        print(f"ccxt rows fetched: {len(df)}")
        return df
    except Exception as e:
        print(f"ccxt fetch failed: {e}")
        return None


def fetch_btc_yfinance() -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        print(f"yfinance not available: {e}")
        _print_optional_dep_instructions()
        return None

    print("Attempting to fetch via yfinance (BTC-USD, 1h candles)...")
    try:
        # Try full history at 1h. Yahoo may restrict length; try 'max' period first.
        df = yf.download("BTC-USD", interval="1h", period="max", auto_adjust=False, progress=False)
        if df is None or df.empty or len(df) < 10000:
            # fallback: daily then upsample to 1h
            print("yfinance 1h insufficient; falling back to daily and resampling to 1H...")
            df_daily = yf.download("BTC-USD", interval="1d", period="max", auto_adjust=False, progress=False)
            if df_daily is None or df_daily.empty:
                return None
            df = df_daily.resample("1H").ffill()
        df = df.rename(columns={c: c.title() for c in df.columns})
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        print(f"yfinance rows fetched: {len(df)}")
        return df
    except Exception as e:
        print(f"yfinance fetch failed: {e}")
        return None


def fetch_btc_pandas_datareader() -> Optional[pd.DataFrame]:
    try:
        from pandas_datareader import data as pdr  # type: ignore
    except Exception as e:
        print(f"pandas_datareader not available: {e}")
        _print_optional_dep_instructions()
        return None

    print("Attempting to fetch via pandas_datareader (Yahoo daily, resample to 1H)...")
    try:
        df = pdr.get_data_yahoo("BTC-USD", start="2020-01-01")
        if df is None or df.empty:
            return None
        df = df.rename(columns={c: c.title() for c in df.columns})
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        # Resample from daily to 1H via forward fill
        df = df.resample("1H").ffill()
        print(f"pandas_datareader rows fetched: {len(df)}")
        return df
    except Exception as e:
        print(f"pandas_datareader fetch failed: {e}")
        return None


def fetch_btc_data(preferred: str = "auto", min_rows: int = 10000) -> Tuple[pd.DataFrame, str]:
    """Fetch BTC OHLCV hourly data since 2020.
    Returns (df, source).
    """
    since_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    since_ms = int(since_dt.timestamp() * 1000)

    sources = ["ccxt", "yfinance", "pandas_datareader"]
    if preferred != "auto" and preferred in sources:
        sources = [preferred] + [s for s in sources if s != preferred]

    df: Optional[pd.DataFrame] = None
    used = ""
    for src in sources:
        if src == "ccxt":
            df = fetch_btc_ccxt(since_ms)
            used = "ccxt"
        elif src == "yfinance":
            df = fetch_btc_yfinance()
            used = "yfinance"
        elif src == "pandas_datareader":
            df = fetch_btc_pandas_datareader()
            used = "pandas_datareader"
        if df is not None and not df.empty and len(df) >= min_rows:
            print(f"Using data source: {used}")
            return df, used
        else:
            print(f"Source {src} insufficient or failed.")

    if df is not None and not df.empty:
        print(f"Using data source: {used} (warning: only {len(df)} rows; < {min_rows})")
        return df, used
    raise RuntimeError("Failed to fetch BTC data from ccxt, yfinance, and pandas_datareader.")


# ---------------------------------------------
# Feature engineering: RSI, EMA12, EMA26
# ---------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)
    # EMA
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    # RSI (Wilder)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi.fillna(50.0)
    # Drop initial NaNs from EMAs
    df = df.dropna()
    return df


# ---------------------------------------------
# Windowing and scaling
# ---------------------------------------------

def build_windows(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    stride: int,
    classification: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Build sliding windows X, y from a feature-engineered DataFrame.
    Returns X (N,T,F), y (N,h) for regression OR y_class (N,) for classification,
    feature names, and close array used for labels reference.
    """
    features = ["Close", "Volume", "rsi_14", "ema_12", "ema_26"]
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")

    data = df[features].astype(float).values
    close = df["Close"].astype(float).values
    N = len(df)
    X_list: List[np.ndarray] = []
    y_reg_list: List[np.ndarray] = []
    y_cls_list: List[int] = []

    max_start = N - window_size - horizon
    for start in range(0, max_start + 1, stride):
        x = data[start : start + window_size]
        future_close = close[start + window_size : start + window_size + horizon]
        if classification:
            # Up/down classification for next step only
            cur_last_close = close[start + window_size - 1]
            label = 1 if (future_close[0] - cur_last_close) >= 0 else 0
            y_cls_list.append(label)
        else:
            y_reg_list.append(future_close.copy())
        X_list.append(x)

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, window_size, len(features)))
    if classification:
        y = np.array(y_cls_list, dtype=np.int64)
    else:
        y = np.stack(y_reg_list, axis=0) if y_reg_list else np.empty((0, horizon))
    return X, y, features, close


def time_split_indices(n: int, train_frac: float = 0.8, val_frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def build_windows_from_frames(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    window_size: int,
    stride: int,
    classification: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Create sliding windows from feature/label dataframes delivered by FreqAI.

    Returns X (N, T, F), y (N, label_dim) or (N,) for classification, and a list of
    indices (typically timestamps) corresponding to the window end positions.
    """

    if len(feature_df) != len(label_df):
        raise ValueError("Feature and label dataframes must share the same length")
    if len(feature_df) < window_size:
        raise ValueError("Not enough rows to create a single window")

    # Align indices and drop NaNs jointly
    combined = pd.concat([feature_df, label_df], axis=1, join="inner")
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    feature_cols = feature_df.columns
    label_cols = label_df.columns
    features = combined[feature_cols].astype(np.float32).to_numpy()
    labels = combined[label_cols].to_numpy()
    idx_list = combined.index.to_list()

    max_start = len(combined) - window_size
    X_list: List[np.ndarray] = []
    y_list: List[Any] = []
    end_indices: List[Any] = []

    for start in range(0, max_start + 1, stride):
        end = start + window_size
        X_list.append(features[start:end])
        target_row = labels[end - 1]
        y_list.append(target_row)
        end_indices.append(idx_list[end - 1])

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, window_size, features.shape[-1]), dtype=np.float32)
    if classification:
        cls_targets: List[int] = []
        for v in y_list:
            arr = np.atleast_1d(np.asarray(v))
            cls_targets.append(int(arr[0]))
        y = np.asarray(cls_targets, dtype=np.int64)
    else:
        y_array = np.asarray(y_list, dtype=np.float32)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        y = y_array
    return X, y, end_indices


def fit_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_minmax: bool = False,
    classification: bool = False,
) -> Tuple[Any, Any]:
    Scaler = MinMaxScaler if use_minmax else RobustScaler
    x_scaler = Scaler()
    # Fit on features aggregated across time
    B, T, F = X_train.shape
    x_scaler.fit(X_train.reshape(B * T, F))
    if classification:
        y_scaler = None
    else:
        y_scaler = Scaler()
        y_scaler.fit(y_train)
    return x_scaler, y_scaler


def apply_scalers(
    X: np.ndarray, y: np.ndarray, x_scaler: Any, y_scaler: Any, classification: bool
) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return X, y
    B, T, F = X.shape
    Xs = x_scaler.transform(X.reshape(B * T, F)).reshape(B, T, F)
    if classification or y_scaler is None or y.size == 0:
        ys = y
    else:
        ys = y_scaler.transform(y)
    return Xs, ys


# ---------------------------------------------
# Dataset and collate for variable lengths
# ---------------------------------------------

class TimeSeriesWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_lengths: bool = False,
        min_seq_len: int = 20,
    ) -> None:
        self.variable_lengths = variable_lengths
        self.min_seq_len = min_seq_len
        self.X = X.astype(np.float32)
        self.y = y
        self.lengths: Optional[np.ndarray] = None
        if variable_lengths and len(X) > 0:
            B, T, _ = X.shape
            rng = np.random.default_rng(123)
            lengths = rng.integers(low=min_seq_len, high=T + 1, size=B)
            self.lengths = lengths.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x = self.X[idx]
        if self.lengths is not None:
            L = int(self.lengths[idx])
            x = x[:L]
        else:
            L = x.shape[0]
        y = self.y[idx]
        return torch.from_numpy(x), torch.as_tensor(y), L


def collate_with_padding(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    # Batch list of (x, y, length). Pad to max_len with 0.0
    xs, ys, lengths = zip(*batch)
    max_len = max(lengths)
    feat_dim = xs[0].shape[-1]
    batch_size = len(xs)
    x_padded = xs[0].new_zeros((batch_size, max_len, feat_dim))
    key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    for i, (x, L) in enumerate(zip(xs, lengths)):
        x_padded[i, :L] = x
        key_padding_mask[i, :L] = False  # False = keep, True = mask
    y_tensor = torch.stack(ys)
    return {"x": x_padded, "y": y_tensor, "key_padding_mask": key_padding_mask, "lengths": torch.as_tensor(lengths)}


# ---------------------------------------------
# Model components: PositionalEncoding, Custom Transformer layer, Hybrid model
# ---------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        batch_first: bool = True,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu
        self.last_attn_weights: Optional[torch.Tensor] = None
        self.capture_attention: bool = False

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        need_weights = self.capture_attention
        src2, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        if attn_weights is not None:
            self.last_attn_weights = attn_weights.detach().cpu()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class HybridModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        horizon: int = 1,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        bidirectional: bool = False,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 1,
        ff_dim: int = 512,
        transformer_dropout: float = 0.1,
        pooling: str = "last",
        classification: bool = False,
        capture_attention: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.classification = classification
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(lstm_out_dim, d_model)
        self.posenc = PositionalEncoding(d_model, dropout=0.0)

        layers = []
        for _ in range(num_transformer_layers):
            layer = CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=transformer_dropout,
                batch_first=True,
            )
            layers.append(layer)
        self.enc_layers = nn.ModuleList(layers)
        self.enc_dropout = nn.Dropout(transformer_dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        if classification:
            self.head = nn.Linear(d_model, 2)  # up/down
        else:
            self.head = nn.Linear(d_model, horizon)

        # Attention capture flag (propagate to layers)
        self.capture_attention = capture_attention

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        out = self.proj(out)
        out = self.posenc(out)

        attn_weights: List[torch.Tensor] = []
        for layer in self.enc_layers:
            assert isinstance(layer, CustomTransformerEncoderLayer)
            layer.capture_attention = self.capture_attention
            out = layer(out, src_key_padding_mask=key_padding_mask)
            if self.capture_attention and layer.last_attn_weights is not None:
                attn_weights.append(layer.last_attn_weights)  # (B, nhead, T, T)

        out = self.enc_dropout(out)
        out = self.layer_norm(out)
        if self.pooling == "mean":
            if key_padding_mask is not None:
                # mask padded positions
                mask = (~key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
                summed = (out * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            else:
                pooled = out.mean(dim=1)
        else:
            # last token pooling (last valid position if mask available)
            if key_padding_mask is not None:
                lengths = (~key_padding_mask).sum(dim=1) - 1
                idx = lengths.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, out.size(-1))
                pooled = out.gather(1, idx).squeeze(1)
            else:
                pooled = out[:, -1, :]

        logits = self.head(pooled)
        return logits, (attn_weights if self.capture_attention else None)


# ---------------------------------------------
# Trainer utilities
# ---------------------------------------------

@dataclass
class TrainConfig:
    window_size: int = 60
    horizon: int = 1
    stride: int = 1
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    use_minmax: bool = False
    num_workers: int = 2
    variable_lengths: bool = False
    min_seq_len: int = 20
    pooling: str = "last"
    classification: bool = False
    capture_attention: bool = False
    bidirectional: bool = False
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    d_model: int = 128
    nhead: int = 4
    num_transformer_layers: int = 1
    ff_dim: int = 512
    transformer_dropout: float = 0.1
    compile_model: bool = False


class Trainer:
    def __init__(
        self,
        model: HybridModel,
        device: torch.device,
        config: TrainConfig,
        classification: bool = False,
    ) -> None:
        self.model = model.to(device)
        if config.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
                print("Model compiled with torch.compile().")
            except Exception as e:
                print(f"torch.compile failed (ignored): {e}")
        self.device = device
        self.config = config
        self.classification = classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        self.best_val_loss = float("inf")
        self.best_state: Optional[Dict[str, Any]] = None
        self.epochs_no_improve = 0

    def _step(self, batch: Dict[str, torch.Tensor], train: bool = True) -> Tuple[float, float, float]:
        x = batch["x"].to(self.device, non_blocking=True)
        y = batch["y"].to(self.device, non_blocking=True)
        key_padding_mask = batch.get("key_padding_mask")
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(self.device, non_blocking=True)
        # Mixed precision
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            preds, _ = self.model(x, key_padding_mask=key_padding_mask)
            if self.classification:
                loss = F.cross_entropy(preds, y.long())
                # Metrics: use accuracy as MAE placeholder, RMSE not applicable
                with torch.no_grad():
                    acc = (preds.argmax(dim=-1) == y.long()).float().mean().item()
                mae = 1.0 - acc
                rmse = 0.0
            else:
                loss = F.mse_loss(preds, y)
                with torch.no_grad():
                    mae = F.l1_loss(preds, y).item()
                    rmse = torch.sqrt(F.mse_loss(preds, y)).item()

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.config.grad_clip is not None and self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return loss.item(), mae, rmse

    def train_epochs(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        max_epochs: int,
    ) -> Dict[str, Any]:
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "train_rmse": [], "val_rmse": [], "lr": []}
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            train_loss_vals: List[float] = []
            train_mae_vals: List[float] = []
            train_rmse_vals: List[float] = []
            for batch in train_loader:
                loss, mae, rmse = self._step(batch, train=True)
                train_loss_vals.append(loss)
                train_mae_vals.append(mae)
                train_rmse_vals.append(rmse)

            self.model.eval()
            val_loss_vals: List[float] = []
            val_mae_vals: List[float] = []
            val_rmse_vals: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    loss, mae, rmse = self._step(batch, train=False)
                    val_loss_vals.append(loss)
                    val_mae_vals.append(mae)
                    val_rmse_vals.append(rmse)

            train_loss = float(np.mean(train_loss_vals)) if train_loss_vals else float("inf")
            val_loss = float(np.mean(val_loss_vals)) if val_loss_vals else float("inf")
            train_mae = float(np.mean(train_mae_vals)) if train_mae_vals else float("inf")
            val_mae = float(np.mean(val_mae_vals)) if val_mae_vals else float("inf")
            train_rmse = float(np.mean(train_rmse_vals)) if train_rmse_vals else float("inf")
            val_rmse = float(np.mean(val_rmse_vals)) if val_rmse_vals else float("inf")
            lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_mae"].append(train_mae)
            history["val_mae"].append(val_mae)
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)
            history["lr"].append(lr)

            print(
                f"Epoch {epoch:03d} | LR {lr:.2e} | Train: loss {train_loss:.4f}, MAE {train_mae:.4f}, RMSE {train_rmse:.4f} | "
                f"Val: loss {val_loss:.4f}, MAE {val_mae:.4f}, RMSE {val_rmse:.4f}"
            )

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {"model": self.model.state_dict()}
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.config.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["model"])  # type: ignore[arg-type]
        return history


# ---------------------------------------------
# Inference utilities
# ---------------------------------------------

def predict(
    model: HybridModel,
    device: torch.device,
    X: np.ndarray,
    x_scaler: Any,
    y_scaler: Optional[Any],
    batch_size: int = 64,
    variable_lengths: bool = False,
) -> np.ndarray:
    model.eval()
    B, T, F = X.shape
    Xs = x_scaler.transform(X.reshape(B * T, F)).reshape(B, T, F).astype(np.float32)
    ds = TimeSeriesWindowDataset(Xs, np.zeros((B, model.horizon), dtype=np.float32), variable_lengths=variable_lengths)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_with_padding)
    preds_list: List[np.ndarray] = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            key_padding_mask = batch.get("key_padding_mask")
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.to(device, non_blocking=True)
            y_hat, _ = model(x, key_padding_mask=key_padding_mask)
            preds_list.append(y_hat.detach().cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    if not model.classification and y_scaler is not None:
        preds = y_scaler.inverse_transform(preds)
    return preds


# ---------------------------------------------
# Plotting helpers
# ---------------------------------------------

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str = "Predictions vs Actual") -> None:
    plt.figure(figsize=(12, 5))
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        # Plot the first horizon step for simplicity
        plt.plot(y_true[:, 0], label="Actual (t+1)")
        plt.plot(y_pred[:, 0], label="Pred (t+1)")
    else:
        plt.plot(y_true.squeeze(), label="Actual")
        plt.plot(y_pred.squeeze(), label="Pred")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_attention_heatmap(attn: torch.Tensor, out_path: str) -> None:
    # attn: (B, nhead, T, T) capture one head of first sample
    a = attn[0, 0].numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(a, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="attention")
    plt.title("Self-Attention (Layer 1, Head 1)")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------
# FreqAI integration (optional class)
# ---------------------------------------------

try:
    # Use PyTorch regressor base which provides default train(), we implement fit()/predict()
    from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor  # type: ignore
    from freqtrade.freqai.torch.PyTorchDataConvertor import (  # type: ignore
        DefaultPyTorchDataConvertor,
        PyTorchDataConvertor,
    )
    from freqtrade.freqai.data_kitchen import FreqaiDataKitchen  # type: ignore
except Exception:
    class BasePyTorchModel:  # type: ignore
        """Fallback stub if freqtrade is not installed. Install with:
        pip install freqtrade
        """

        pass

    FreqaiDataKitchen = Any  # type: ignore


class HybridTimeseriesFreqAIModel_tinhn(BasePyTorchRegressor):  # type: ignore
    """FreqAI-compatible wrapper around the hybrid LSTM+Transformer model.

    The adapter converts FreqAI feature/label dataframes into sliding windows,
    handles scaling, trains the PyTorch model, and returns predictions in the
    format expected by FreqAI.
    """

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        self.cfg: Optional[TrainConfig] = None
        self.model: Optional[HybridModel] = None
        self.x_scaler: Optional[Any] = None
        self.y_scaler: Optional[Any] = None
        self.feature_columns: Optional[List[str]] = None
        self.label_columns: Optional[List[str]] = None
        self.device: Optional[torch.device] = None

    # Provide data converter for BasePyTorch* pipeline
    @property
    def data_convertor(self) -> "PyTorchDataConvertor":  # type: ignore[override]
        # Regression default: float targets; shape kept as-is by DefaultPyTorchDataConvertor
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def _resolve_config(self) -> TrainConfig:
        params = self.freqai_info.get("model_training_parameters", {}) if hasattr(self, "freqai_info") else {}
        return TrainConfig(
            window_size=int(params.get("window_size", 60)),
            horizon=int(params.get("horizon", 1)),
            stride=int(params.get("stride", 1)),
            batch_size=int(params.get("batch_size", 32)),
            epochs=int(params.get("epochs", 20)),
            lr=float(params.get("lr", 1e-3)),
            grad_clip=float(params.get("grad_clip", 1.0)),
            early_stopping_patience=int(params.get("early_stopping_patience", 10)),
            use_minmax=bool(params.get("use_minmax", False)),
            num_workers=int(params.get("num_workers", 2)),
            variable_lengths=bool(params.get("variable_lengths", False)),
            min_seq_len=int(params.get("min_seq_len", 20)),
            pooling=str(params.get("pooling", "last")),
            classification=bool(params.get("classification", False)),
            capture_attention=bool(params.get("capture_attention", False)),
            bidirectional=bool(params.get("bidirectional", False)),
            lstm_hidden=int(params.get("lstm_hidden", 128)),
            lstm_layers=int(params.get("lstm_layers", 2)),
            lstm_dropout=float(params.get("lstm_dropout", 0.1)),
            d_model=int(params.get("d_model", 128)),
            nhead=int(params.get("nhead", 4)),
            num_transformer_layers=int(params.get("num_transformer_layers", 1)),
            ff_dim=int(params.get("ff_dim", 512)),
            transformer_dropout=float(params.get("transformer_dropout", 0.1)),
            compile_model=bool(params.get("compile_model", False)),
        )

    @staticmethod
    def _ensure_dataframe(obj: Any) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, np.ndarray):
            return pd.DataFrame(obj)
        raise TypeError("Expected pandas DataFrame or numpy array for FreqAI data inputs")

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs: Any) -> Any:  # type: ignore[override]
        cfg = self._resolve_config()
        self.cfg = cfg

        train_features = self._ensure_dataframe(data_dictionary.get("train_features"))
        train_labels = self._ensure_dataframe(data_dictionary.get("train_labels"))
        val_features = data_dictionary.get("val_features")
        val_labels = data_dictionary.get("val_labels")

        if val_features is None or val_labels is None:
            split_point = int(len(train_features) * 0.8)
            val_features = train_features.iloc[split_point:].copy()
            val_labels = train_labels.iloc[split_point:].copy()
            train_features = train_features.iloc[:split_point].copy()
            train_labels = train_labels.iloc[:split_point].copy()
        else:
            val_features = self._ensure_dataframe(val_features)
            val_labels = self._ensure_dataframe(val_labels)

        self.feature_columns = train_features.columns.tolist()
        label_columns = train_labels.columns.tolist()
        self.label_columns = label_columns

        X_train, y_train, _ = build_windows_from_frames(
            train_features,
            train_labels,
            window_size=cfg.window_size,
            stride=cfg.stride,
            classification=cfg.classification,
        )
        X_val, y_val, _ = build_windows_from_frames(
            val_features,
            val_labels,
            window_size=cfg.window_size,
            stride=cfg.stride,
            classification=cfg.classification,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Insufficient data to build training/validation windows for FreqAI model.")

        dummy_y = y_train if not cfg.classification else np.zeros((len(X_train), cfg.horizon), dtype=np.float32)
        x_scaler, y_scaler = fit_scalers(X_train, dummy_y, use_minmax=cfg.use_minmax, classification=cfg.classification)
        X_train_s, y_train_s = apply_scalers(X_train, y_train, x_scaler, y_scaler, cfg.classification)
        X_val_s, y_val_s = apply_scalers(X_val, y_val, x_scaler, y_scaler, cfg.classification)

        dataset_kwargs = dict(variable_lengths=cfg.variable_lengths, min_seq_len=cfg.min_seq_len)
        ds_tr = TimeSeriesWindowDataset(X_train_s, y_train_s, **dataset_kwargs)
        ds_val = TimeSeriesWindowDataset(X_val_s, y_val_s, variable_lengths=False)

        device = get_device()
        self.device = device

        loader_kwargs = dict(
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_with_padding,
        )
        if device.type == "cuda":
            loader_kwargs.update(dict(pin_memory=True, persistent_workers=(cfg.num_workers > 0)))

        train_loader = torch.utils.data.DataLoader(ds_tr, **loader_kwargs)
        val_loader = torch.utils.data.DataLoader(ds_val, **{**loader_kwargs, "shuffle": False})

        model = HybridModel(
            input_size=X_train.shape[-1],
            horizon=cfg.horizon,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            lstm_dropout=cfg.lstm_dropout,
            bidirectional=cfg.bidirectional,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_transformer_layers=cfg.num_transformer_layers,
            ff_dim=cfg.ff_dim,
            transformer_dropout=cfg.transformer_dropout,
            pooling=cfg.pooling,
            classification=cfg.classification,
            capture_attention=cfg.capture_attention,
        )

        trainer = Trainer(model, device, cfg, classification=cfg.classification)
        trainer.train_epochs(train_loader, val_loader, cfg.epochs)

        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler if not cfg.classification else None

        # Persist references for saving/restoring via FreqAI
        self.freqai_trainer = trainer  # type: ignore[attr-defined]
        self.dk = dk
        return trainer

    def _prepare_inference_windows(self, feature_df: pd.DataFrame) -> Tuple[np.ndarray, List[Any]]:
        assert self.cfg is not None
        cfg = self.cfg
        if self.feature_columns is not None:
            feature_df = feature_df[self.feature_columns]
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
        feature_df = feature_df.sort_index()
        if feature_df.shape[1] == 0:
            raise ValueError("Feature dataframe is empty; cannot prepare inference window.")
        if len(feature_df) < cfg.window_size:
            raise ValueError("Not enough rows to create inference window. Increase history or reduce window_size.")

        placeholder_labels = pd.DataFrame(
            {"__placeholder_label__": feature_df.iloc[:, 0].to_numpy()},
            index=feature_df.index,
        )
        X, _, indices = build_windows_from_frames(
            feature_df,
            placeholder_labels,
            window_size=cfg.window_size,
            stride=1,
            classification=False,
        )
        return X[-1:], indices[-1:]

    def predict(self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs: Any) -> Tuple[pd.DataFrame, np.ndarray]:  # type: ignore[override]
        """Predict over a full backtest/live slice using DK pipelines.

        - Uses DataKitchen to select the trained feature set and compute do_predict/DI masks.
        - Creates sliding windows across the full slice and aligns predictions to the
          original dataframe length (leading positions without a full window are zero).
        - Returns a DataFrame with the same number of rows as ``unfiltered_df`` and a
          numpy array for ``do_predict`` compatible with FreqAI expectations.
        """
        if self.model is None or self.x_scaler is None or self.cfg is None:
            raise RuntimeError("Model has not been fitted before predict call.")

        cfg = self.cfg
        base_model = self.model
        # unwrap saved trainer wrappers (BasePyTorchRegressor saves wrapper objects via DataDrawer)
        torch_model = base_model.model if hasattr(base_model, "model") else base_model  # type: ignore[assignment]
        # Normalize device type to torch.device
        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device or get_device()

        # 1) Build feature dataframe via DK (ensures same feature list as training)
        try:
            dk.find_features(unfiltered_df)
        except Exception:
            # If already set from training, proceed
            pass

        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        dk.data_dictionary["prediction_features"] = filtered_df.copy()

        # 2) Apply DK feature pipeline to compute outliers/DI (keeps row count)
        outliers: np.ndarray
        di_values: np.ndarray
        try:
            pred_feats, outliers, _ = dk.feature_pipeline.transform(
                dk.data_dictionary["prediction_features"], outlier_check=True
            )
            dk.data_dictionary["prediction_features"] = pred_feats
            # Only query DI step if configured to avoid noisy warnings from the pipeline
            di_threshold = (
                self.freqai_info.get("feature_parameters", {}).get("DI_threshold", 0)
                if hasattr(self, "freqai_info") else 0
            )
            if di_threshold:
                try:
                    di_step = dk.feature_pipeline["di"]  # may warn if absent
                except Exception:
                    di_step = None
                if di_step is not None:
                    di_values = di_step.di_values  # type: ignore[attr-defined]
                else:
                    di_values = np.zeros(len(pred_feats), dtype=np.float32)
            else:
                di_values = np.zeros(len(pred_feats), dtype=np.float32)
        except Exception:
            # Fallback: if pipeline unavailable, keep raw features and mark all as predictible
            pred_feats = dk.data_dictionary["prediction_features"].copy()
            outliers = np.ones(len(pred_feats), dtype=np.int_)
            di_values = np.zeros(len(pred_feats), dtype=np.float32)

        dk.do_predict = outliers
        dk.DI_values = di_values

        # 3) Window the full slice and run the model in batches
        feature_df = pred_feats
        total_len = len(feature_df)

        if total_len < cfg.window_size:
            # Not enough data to form a single window -> return zeros matching length
            if cfg.classification:
                column_name = self.label_columns[0] if self.label_columns else "&-pred_up_prob"
                pred_df = pd.DataFrame({column_name: np.zeros(total_len, dtype=np.float32)}, index=feature_df.index)
            else:
                columns = self.label_columns or ["&-prediction"]
                pred_df = pd.DataFrame(np.zeros((total_len, len(columns)), dtype=np.float32), columns=columns, index=feature_df.index)
            return pred_df, dk.do_predict

        # Placeholder labels just to reuse the window builder and keep indices
        placeholder_labels = pd.DataFrame(
            {"__placeholder_label__": np.zeros(total_len, dtype=np.float32)},
            index=feature_df.index,
        )
        X_windows, _, end_indices = build_windows_from_frames(
            feature_df,
            placeholder_labels,
            window_size=cfg.window_size,
            stride=1,
            classification=False,
        )

        # Scale & forward pass (batch inference)
        dummy_y = np.zeros((len(X_windows), cfg.horizon), dtype=np.float32)
        X_scaled, _ = apply_scalers(X_windows, dummy_y, self.x_scaler, self.y_scaler, cfg.classification)

        torch_model.eval()
        preds_list: List[np.ndarray] = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            # Use the dataset/collate to keep behavior consistent with training
            ds = TimeSeriesWindowDataset(X_scaled, np.zeros((len(X_scaled), cfg.horizon), dtype=np.float32), variable_lengths=cfg.variable_lengths)
            loader = torch.utils.data.DataLoader(ds, batch_size=max(1, cfg.batch_size), shuffle=False, num_workers=0, collate_fn=collate_with_padding)
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                key_padding_mask = batch.get("key_padding_mask")
                if key_padding_mask is not None:
                    key_padding_mask = key_padding_mask.to(device, non_blocking=True)
                y_hat, _ = torch_model(x, key_padding_mask=key_padding_mask)
                preds_list.append(y_hat.detach().cpu().numpy())
        preds_np = np.concatenate(preds_list, axis=0) if preds_list else np.zeros((0, cfg.horizon), dtype=np.float32)

        # 4) Align predictions back to full length (leading rows get 0)
        if cfg.classification:
            probs = torch.softmax(torch.from_numpy(preds_np), dim=-1).numpy() if len(preds_np) else np.zeros((0, 2), dtype=np.float32)
            prob_up = probs[:, 1] if probs.shape[-1] > 1 else np.zeros(len(probs), dtype=np.float32)
            column_name = self.label_columns[0] if self.label_columns else "&-pred_up_prob"
            full_vals = np.zeros(total_len, dtype=np.float32)
            # end_indices reference the last index for each window
            if len(end_indices) > 0:
                full_indexer = pd.Index(end_indices)
                # Map predictions to their corresponding positions
                pd.Series(prob_up, index=full_indexer).reindex(feature_df.index, fill_value=0.0)
                full_series = pd.Series(full_vals, index=feature_df.index)
                full_series.loc[full_indexer] = prob_up
                pred_df = pd.DataFrame({column_name: full_series.values}, index=feature_df.index)
            else:
                pred_df = pd.DataFrame({column_name: full_vals}, index=feature_df.index)
            return pred_df, dk.do_predict
        else:
            # Inverse model-specific scaler first (second stage)
            if self.y_scaler is not None and len(preds_np) > 0:
                preds_np = self.y_scaler.inverse_transform(preds_np)

            columns = self.label_columns or ["&-prediction"]
            full_preds = np.zeros((total_len, len(columns)), dtype=np.float32)
            if len(end_indices) > 0 and len(preds_np) > 0:
                # Place window-end predictions at their positions
                idx_pos = pd.Index(feature_df.index)
                end_pos = idx_pos.get_indexer(end_indices)
                valid = end_pos >= 0
                full_preds[end_pos[valid]] = preds_np[: np.count_nonzero(valid)]
            pred_df = pd.DataFrame(full_preds, columns=columns, index=feature_df.index)

            # Then inverse the DK label pipeline to get back to original label scale if present
            try:
                pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
            except Exception:
                # If label pipeline not available, keep current scale
                pass

            return pred_df, dk.do_predict


FREQAI_CONFIG_SNIPPET = """
"model_training_parameters": {
  "window_size": 60,
  "horizon": 1,
  "stride": 1,
  "batch_size": 32,
  "epochs": 30,
  "lr": 0.001,
  "grad_clip": 1.0,
  "early_stopping_patience": 10,
  "use_minmax": false,
  "num_workers": 2,
  "variable_lengths": false,
  "min_seq_len": 20,
  "pooling": "last",
  "classification": false,
  "capture_attention": false,
  "bidirectional": false,
  "lstm_hidden": 128,
  "lstm_layers": 2,
  "lstm_dropout": 0.1,
  "d_model": 128,
  "nhead": 4,
  "num_transformer_layers": 1,
  "ff_dim": 512,
  "transformer_dropout": 0.1,
  "compile_model": false
}
"""


# ---------------------------------------------
# Main training/evaluation script
# ---------------------------------------------

def run(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = get_device()

    # Fetch data
    df, source = fetch_btc_data(preferred=args.source)
    print(f"Data source used: {source}")
    df = compute_indicators(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Data after indicators and NaN handling: {len(df)} rows")

    # Build windows
    X_all, y_all, feature_names, close = build_windows(
        df, window_size=args.window_size, horizon=args.horizon, stride=args.stride, classification=args.classification
    )
    if len(X_all) == 0:
        raise RuntimeError("No windows produced. Check parameters and data size.")

    # Time-based split on window indices
    idx_tr, idx_val, idx_te = time_split_indices(len(X_all), train_frac=0.8, val_frac=0.1)
    X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
    X_val, y_val = X_all[idx_val], y_all[idx_val]
    X_te, y_te = X_all[idx_te], y_all[idx_te]
    print(f"Windows: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

    # Optionally shrink for smoke test
    if args.smoke_test:
        X_tr, y_tr = X_tr[:1024], y_tr[:1024]
        X_val, y_val = X_val[:256], y_val[:256]
        X_te, y_te = X_te[:256], y_te[:256]
        print("Smoke test mode: using small subsets.")

    # Fit scalers on train only
    x_scaler, y_scaler = fit_scalers(X_tr, y_tr if not args.classification else np.zeros((len(X_tr), args.horizon)), use_minmax=args.use_minmax, classification=args.classification)
    X_tr_s, y_tr_s = apply_scalers(X_tr, y_tr, x_scaler, y_scaler, args.classification)
    X_val_s, y_val_s = apply_scalers(X_val, y_val, x_scaler, y_scaler, args.classification)
    X_te_s, y_te_s = apply_scalers(X_te, y_te, x_scaler, y_scaler, args.classification)

    # Datasets / Loaders
    ds_tr = TimeSeriesWindowDataset(X_tr_s, y_tr_s, variable_lengths=args.variable_lengths, min_seq_len=args.min_seq_len)
    ds_val = TimeSeriesWindowDataset(X_val_s, y_val_s, variable_lengths=False)
    ds_te = TimeSeriesWindowDataset(X_te_s, y_te_s, variable_lengths=False)

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_with_padding)
    if device.type == "cuda":
        loader_kwargs.update(dict(pin_memory=True, persistent_workers=(args.num_workers > 0)))

    train_loader = torch.utils.data.DataLoader(ds_tr, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(ds_val, **{**loader_kwargs, "shuffle": False})
    test_loader = torch.utils.data.DataLoader(ds_te, **{**loader_kwargs, "shuffle": False})

    # Model
    model = HybridModel(
        input_size=X_tr.shape[-1],
        horizon=args.horizon,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        bidirectional=args.bidirectional,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.num_transformer_layers,
        ff_dim=args.ff_dim,
        transformer_dropout=args.transformer_dropout,
        pooling=args.pooling,
        classification=args.classification,
        capture_attention=args.capture_attention,
    )
    cfg = TrainConfig(
        window_size=args.window_size,
        horizon=args.horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        use_minmax=args.use_minmax,
        num_workers=args.num_workers,
        variable_lengths=args.variable_lengths,
        min_seq_len=args.min_seq_len,
        pooling=args.pooling,
        classification=args.classification,
        capture_attention=args.capture_attention,
        bidirectional=args.bidirectional,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.num_transformer_layers,
        ff_dim=args.ff_dim,
        transformer_dropout=args.transformer_dropout,
        compile_model=args.compile_model,
    )

    trainer = Trainer(model, device, cfg, classification=args.classification)
    history = trainer.train_epochs(train_loader, val_loader, cfg.epochs)

    # Save best model
    model_path = os.path.join(args.out_dir, "best_model.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved best model to {model_path}")

    # Save scalers
    with open(os.path.join(args.out_dir, "x_scaler.pkl"), "wb") as f:
        pickle.dump(x_scaler, f)
    if y_scaler is not None:
        with open(os.path.join(args.out_dir, "y_scaler.pkl"), "wb") as f:
            pickle.dump(y_scaler, f)

    # Test evaluation
    model.eval()
    test_loss_vals: List[float] = []
    test_mae_vals: List[float] = []
    test_rmse_vals: List[float] = []
    with torch.no_grad():
        for batch in test_loader:
            loss, mae, rmse = trainer._step(batch, train=False)
            test_loss_vals.append(loss)
            test_mae_vals.append(mae)
            test_rmse_vals.append(rmse)
    test_loss = float(np.mean(test_loss_vals)) if test_loss_vals else float("inf")
    test_mae = float(np.mean(test_mae_vals)) if test_mae_vals else float("inf")
    test_rmse = float(np.mean(test_rmse_vals)) if test_rmse_vals else float("inf")
    print(f"Test: loss {test_loss:.4f}, MAE {test_mae:.4f}, RMSE {test_rmse:.4f}")

    # Prediction and inverse scaling (regression)
    y_hat_list: List[np.ndarray] = []
    y_true_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device, non_blocking=True)
            kpm = batch.get("key_padding_mask")
            if kpm is not None:
                kpm = kpm.to(device, non_blocking=True)
            y = batch["y"].cpu().numpy()
            y_true_list.append(y)
            preds, _ = model(x, key_padding_mask=kpm)
            y_hat_list.append(preds.detach().cpu().numpy())
    y_pred_scaled = np.concatenate(y_hat_list, axis=0)
    y_true_scaled = np.concatenate(y_true_list, axis=0)
    if not args.classification and y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_scaler.inverse_transform(y_true_scaled)
    else:
        y_pred = y_pred_scaled
        y_true = y_true_scaled

    # Plot predictions (first horizon step)
    plot_path = os.path.join(args.out_dir, "prediction_plot.png")
    if args.classification:
        # Plot predicted probability of UP for classification
        probs = torch.softmax(torch.from_numpy(y_pred), dim=-1).numpy()[:, 1]
        plt.figure(figsize=(12, 5))
        plt.plot(probs, label="Pred UP prob")
        plt.title("Classification: Probability of Up Move (t+1)")
        plt.legend(); plt.tight_layout(); plt.savefig(plot_path); plt.close()
    else:
        plot_predictions(y_true, y_pred, plot_path, title="Regression: Pred vs Actual (t+1)")
    print(f"Saved prediction plot to {plot_path}")

    # Optional attention visualization from first encoder layer
    if args.capture_attention and len(model.enc_layers) > 0:
        first_layer = model.enc_layers[0]
        if isinstance(first_layer, CustomTransformerEncoderLayer) and first_layer.last_attn_weights is not None:
            attn_plot_path = os.path.join(args.out_dir, "attention_heatmap.png")
            plot_attention_heatmap(first_layer.last_attn_weights, attn_plot_path)
            print(f"Saved attention heatmap to {attn_plot_path}")

    # Save metrics + hyperparams
    metrics = {
        "source": source,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "history": history,
        "hyperparameters": {
            "feature_names": feature_names,
            "window_size": args.window_size,
            "horizon": args.horizon,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "early_stopping_patience": args.early_stopping_patience,
            "use_minmax": args.use_minmax,
            "num_workers": args.num_workers,
            "variable_lengths": args.variable_lengths,
            "min_seq_len": args.min_seq_len,
            "pooling": args.pooling,
            "classification": args.classification,
            "capture_attention": args.capture_attention,
            "bidirectional": args.bidirectional,
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
            "lstm_dropout": args.lstm_dropout,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_transformer_layers": args.num_transformer_layers,
            "ff_dim": args.ff_dim,
            "transformer_dropout": args.transformer_dropout,
            "compile_model": args.compile_model,
        },
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid LSTM+Transformer crypto forecaster")
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--horizon", type=int, default=1, choices=list(range(1, 6)))
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--use_minmax", action="store_true", help="Use MinMaxScaler instead of RobustScaler")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--variable_lengths", action="store_true")
    p.add_argument("--min_seq_len", type=int, default=20)
    p.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
    p.add_argument("--classification", action="store_true", help="Up/Down classification for t+1. Uses CrossEntropy.")
    p.add_argument("--capture_attention", action="store_true")
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--lstm_hidden", type=int, default=128)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--lstm_dropout", type=float, default=0.1)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_transformer_layers", type=int, default=1)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--transformer_dropout", type=float, default=0.1)
    p.add_argument("--compile_model", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=".")
    p.add_argument("--source", type=str, default="auto", choices=["auto", "ccxt", "yfinance", "pandas_datareader"])
    p.add_argument("--smoke_test", action="store_true", help="Run a tiny 1-2 epoch training on a small subset")
    args = p.parse_args(argv)

    if args.classification and args.horizon > 1:
        print("[Warning] Classification mode uses only t+1. Horizon>1 will be ignored in head/labels.")
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as e:
        print(f"ERROR: {e}")
        print("If missing optional deps, install with: pip install ccxt yfinance pandas_datareader")
        sys.exit(1)
