from __future__ import annotations

"""
FreqAIHybridImproved5mShort

- Timeframe: 5m (uses 1h as informative timeframe)
- Supports long and short
- Volatility-aware thresholds via ATR% on 5m
- 1h EMA/RSI trend filter (informative timeframe)
- Smoothed predicted return to reduce flip-flop

Requires FreqAI to inject prediction columns (e.g., '&-prediction', '&-pred_up_prob').
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (
    IntParameter,
    DecimalParameter,
    merge_informative_pair,
)

try:
    import talib.abstract as ta  # type: ignore

    _HAS_TALIB = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_TALIB = False


class FreqAIHybridImproved5mShort(IStrategy):
    timeframe = "5m"
    informative_timeframe = "1h"
    process_only_new_candles = True

    # Allow shorting
    can_short: bool = True

    # Basic ROI/stop
    minimal_roi = {"0": 0.01}
    stoploss = -0.10

    # Trailing configuration
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.004  # 0.4%
    trailing_stop_positive_offset = 0.012  # activate after 1.2%

    # ~2500 x 5m covers 200h + ATR warmup
    startup_candle_count: int = 2500

    # --- Hyperoptable parameters ---
    ema_period = IntParameter(150, 250, default=200, space="buy")
    rsi_threshold = IntParameter(48, 55, default=50, space="buy")
    pred_ema_span = IntParameter(3, 9, default=5, space="buy")

    k_atr = DecimalParameter(0.30, 1.00, default=0.50, decimals=2, space="buy")
    fee_buffer = DecimalParameter(0.0005, 0.0030, default=0.0010, decimals=4, space="buy")
    min_pred_move = DecimalParameter(0.0000, 0.0050, default=0.0000, decimals=4, space="buy")

    prob_up_gate = DecimalParameter(0.50, 0.70, default=0.55, decimals=2, space="buy")
    prob_down_gate = DecimalParameter(0.50, 0.70, default=0.55, decimals=2, space="sell")
    prob_exit_gate = DecimalParameter(0.30, 0.50, default=0.45, decimals=2, space="sell")

    plot_config = {
        "main_plot": {
            "ema_trend_1h": {},
        },
        "subplots": {
            "pred_ret": {"pred_ret": {}, "pred_ret_ema": {}},
            "atr_pct": {"atr_pct": {}},
            "rsi_1h": {"rsi_1h": {}},
        },
    }

    # --- TA helpers ---
    @staticmethod
    def _ema(s: Series, n: int) -> Series:
        if _HAS_TALIB:
            try:
                return ta.EMA(s, timeperiod=n)
            except Exception:
                pass
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _rsi(s: Series, n: int = 14) -> Series:
        if _HAS_TALIB:
            try:
                return ta.RSI(s, timeperiod=n)
            except Exception:
                pass
        d = s.diff()
        up = d.clip(lower=0.0)
        dn = (-d).clip(lower=0.0)
        roll_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
        roll_dn = dn.ewm(alpha=1.0 / n, adjust=False).mean()
        rs = roll_up / roll_dn.replace(0, np.nan)
        out = 100.0 - (100.0 / (1.0 + rs))
        return out.fillna(0.0)

    @staticmethod
    def _wilder_ema(s: Series, n: int) -> Series:
        return s.ewm(alpha=1.0 / float(n), adjust=False).mean()

    @classmethod
    def _atr(cls, df: DataFrame, n: int = 14) -> Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return cls._wilder_ema(tr, n)

    @staticmethod
    def _first_col(df: DataFrame, needles: Any) -> str:
        items = list(needles) if isinstance(needles, (list, tuple)) else [needles]
        for col in df.columns:
            for n in items:
                if n in col:
                    return col
        return ""

    def informative_pairs(self) -> List[Tuple[str, str]]:
        if self.dp:
            return [(p, self.informative_timeframe) for p in self.dp.current_whitelist()]
        return []

    def _inject_freqai(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        if hasattr(self, "freqai"):
            return self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        # Let FreqAI inject predictions and do_predict/DI flags
        dataframe = self._inject_freqai(dataframe, metadata)

        pred_col = self._first_col(dataframe, ["&-prediction", "prediction"])
        do_pred_col = self._first_col(dataframe, ["do_predict", "do_pred"])
        prob_up_col = self._first_col(dataframe, ["&-pred_up_prob", "pred_up_prob", "up_prob", "prob_up"])
        prob_dn_col = self._first_col(dataframe, ["&-pred_dn_prob", "pred_dn_prob", "down_prob", "prob_down"])

        if pred_col:
            dataframe["pred_ret"] = (dataframe[pred_col] - dataframe["close"]) / dataframe["close"]
            dataframe["pred_ret_ema"] = dataframe["pred_ret"].ewm(
                span=int(self.pred_ema_span.value), adjust=False
            ).mean()
        else:
            dataframe["pred_ret"] = 0.0
            dataframe["pred_ret_ema"] = 0.0

        # Volatility guard (ATR on 5m)
        atr = self._atr(dataframe, 14)
        dataframe["atr_pct"] = (atr / dataframe["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dataframe["thr"] = float(self.fee_buffer.value) + float(self.k_atr.value) * dataframe["atr_pct"]
        dataframe["gate_mag"] = dataframe["pred_ret_ema"].abs()

        # Informative timeframe (1h) for trend filters
        if self.dp:
            inf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
            inf["ema_trend"] = self._ema(inf["close"], int(self.ema_period.value))
            inf["rsi"] = self._rsi(inf["close"], 14)
            # Use positional args for broader Freqtrade compatibility
            dataframe = merge_informative_pair(
                dataframe,
                inf,
                self.timeframe,
                self.informative_timeframe,
                ffill=True,
            )
            dataframe.rename(
                columns={
                    f"ema_trend_{self.informative_timeframe}": "ema_trend_1h",
                    f"rsi_{self.informative_timeframe}": "rsi_1h",
                },
                inplace=True,
            )
        else:
            # Fallback if no DataProvider (unit tests / degenerate cases)
            dataframe["ema_trend_1h"] = self._ema(dataframe["close"], int(self.ema_period.value))
            dataframe["rsi_1h"] = self._rsi(dataframe["close"], 14)

        # Probabilities (optional)
        has_prob_up = bool(prob_up_col)
        has_prob_dn = bool(prob_dn_col)
        if has_prob_up:
            dataframe["prob_up"] = dataframe[prob_up_col].clip(0.0, 1.0)
        else:
            dataframe["prob_up"] = 0.5
        if has_prob_dn:
            dataframe["prob_down"] = dataframe[prob_dn_col].clip(0.0, 1.0)
        else:
            dataframe["prob_down"] = 1.0 - dataframe["prob_up"]

        # Precompute probability gates so entries still work when probabilities are absent
        # If no prob columns, treat gates as passed (True)
        dataframe["prob_gate_long_ok"] = (
            dataframe["prob_up"] >= float(self.prob_up_gate.value)
            if has_prob_up
            else True
        )
        dataframe["prob_gate_short_ok"] = (
            dataframe["prob_down"] >= float(self.prob_down_gate.value)
            if has_prob_dn
            else True
        )

        # Exit probability triggers (only when probabilities are present)
        dataframe["prob_exit_long_hit"] = (
            dataframe["prob_up"] < float(self.prob_exit_gate.value)
            if has_prob_up
            else False
        )
        dataframe["prob_exit_short_hit"] = (
            dataframe["prob_down"] < float(self.prob_exit_gate.value)
            if has_prob_dn
            else False
        )

        # do_predict gating
        if do_pred_col:
            dataframe["do_pred"] = dataframe[do_pred_col].fillna(0)
        else:
            dataframe["do_pred"] = 1

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        cond_long = (
            (dataframe["do_pred"] == 1)
            & (dataframe["pred_ret_ema"] > dataframe["thr"])
            & (dataframe["gate_mag"] > float(self.min_pred_move.value))
            & (dataframe["close"] > dataframe["ema_trend_1h"])
            & (dataframe["rsi_1h"] > int(self.rsi_threshold.value))
            & (dataframe["prob_gate_long_ok"])
            & (dataframe["volume"] > 0)
        )

        cond_short = (
            (dataframe["do_pred"] == 1)
            & (dataframe["pred_ret_ema"] < -dataframe["thr"])
            & (dataframe["gate_mag"] > float(self.min_pred_move.value))
            & (dataframe["close"] < dataframe["ema_trend_1h"])
            & (dataframe["rsi_1h"] < 50)
            & (dataframe["prob_gate_short_ok"])
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[cond_long, ["enter_long", "enter_tag"]] = (1, "L_pred>thr_trend")
        dataframe.loc[cond_short, ["enter_short", "enter_tag"]] = (1, "S_pred<thr_trend")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        exit_long = (
            ((dataframe["do_pred"] == 1) & (dataframe["pred_ret_ema"] < -dataframe["thr"]))
            | (dataframe["close"] < dataframe["ema_trend_1h"])
            | (dataframe["rsi_1h"] < 50)
            | (dataframe["prob_exit_long_hit"])  # only true if probs present
        ) & (dataframe["volume"] > 0)

        exit_short = (
            ((dataframe["do_pred"] == 1) & (dataframe["pred_ret_ema"] > dataframe["thr"]))
            | (dataframe["close"] > dataframe["ema_trend_1h"])
            | (dataframe["rsi_1h"] > 50)
            | (dataframe["prob_exit_short_hit"])  # only true if probs present
        ) & (dataframe["volume"] > 0)

        dataframe.loc[exit_long, ["exit_long", "exit_tag"]] = (1, "L_edge_lost")
        dataframe.loc[exit_short, ["exit_short", "exit_tag"]] = (1, "S_edge_lost")
        return dataframe

    # --------- FreqAI required hooks ---------
    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """Define target '&-prediction' as close shifted -label_period.

        FreqAI reads this to build labels. label_period (candles) comes from
        freqai.feature_parameters.label_period_candles in the config.
        """
        label_period = int(self.freqai_info.get("feature_parameters", {}).get("label_period_candles", 24))
        df = dataframe.copy()
        df["&-prediction"] = df["close"].shift(-label_period)
        return df

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """Create period-dependent features that FreqAI expands across indicator periods.

        Attempts TA-Lib where available; falls back to pandas variants and safe NaNs
        for indicators that require longer windows.
        """
        df = dataframe.copy()
        # EMA
        try:
            if _HAS_TALIB:
                df["%-ema"] = ta.EMA(df["close"], timeperiod=period)
            else:
                df["%-ema"] = df["close"].ewm(span=max(1, period), adjust=False).mean()
        except Exception:
            df["%-ema"] = df["close"].ewm(span=max(1, period), adjust=False).mean()

        # RSI
        try:
            if _HAS_TALIB:
                df["%-rsi"] = ta.RSI(df["close"], timeperiod=period)
            else:
                # Wilder RSI fallback
                delta = df["close"].diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.ewm(alpha=1 / max(1, period), adjust=False).mean()
                roll_down = down.ewm(alpha=1 / max(1, period), adjust=False).mean()
                rs = roll_up / roll_down.replace(0, pd.NA)
                df["%-rsi"] = 100 - (100 / (1 + rs))
        except Exception:
            df["%-rsi"] = pd.Series(np.nan, index=df.index)

        # ADX (optional; safe NaN if missing or too short)
        try:
            if _HAS_TALIB and len(df) >= max(2, period):
                df["%-adx"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=period)
            else:
                df["%-adx"] = pd.Series(np.nan, index=df.index)
        except Exception:
            df["%-adx"] = pd.Series(np.nan, index=df.index)

        return df

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        df = dataframe.copy()
        df["%-pct_change"] = df["close"].pct_change()
        df["%-volume"] = df["volume"]
        return df

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        df = dataframe.copy()
        if "date" in df.columns:
            df["%-day_of_week"] = df["date"].dt.dayofweek / 6.0
            df["%-hour_of_day"] = df["date"].dt.hour / 23.0
        return df
