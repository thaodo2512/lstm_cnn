from __future__ import annotations

"""
FreqAIHybridImproved5mShort v2

- Timeframe: 5m with 1h informative regime filter (EMA fast/slow + slope)
- Long only when 1h regime is up; short only when 1h regime is down
- Remove ROI cap: let trailing + exits take profit
- Earlier exits: flip on zero-cross of predicted return and ATR-based guard on 1h
- Stricter entries: require bigger predicted move and probability gates

Requires FreqAI to inject prediction columns (e.g., '&-prediction', '&-pred_up_prob').
"""

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, merge_informative_pair
from datetime import datetime

try:
    import talib.abstract as ta  # type: ignore
    _HAS_TALIB = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_TALIB = False


class FreqAIHybridImproved5mShort(IStrategy):
    timeframe = "5m"
    informative_timeframe = "1h"
    process_only_new_candles = True
    can_short: bool = True

    # ROI OFF (use trailing + exits)
    minimal_roi = {"0": 0.99}

    # Wider trailing so winners can run
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.006          # 0.6%
    trailing_stop_positive_offset = 0.018   # activate after 1.8%

    stoploss = -0.10
    startup_candle_count: int = 3000

    # Make exit signals effective
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # --- Hyperoptable params ---
    ema_period = IntParameter(150, 250, default=200, space="buy")
    ema_fast_1h = IntParameter(40, 80, default=50, space="buy")
    ema_slow_1h = IntParameter(180, 320, default=200, space="buy")

    rsi_threshold = IntParameter(50, 58, default=52, space="buy")  # a bit stricter
    pred_ema_span = IntParameter(3, 9, default=5, space="buy")

    k_atr = DecimalParameter(0.30, 1.20, default=0.60, decimals=2, space="buy")
    k_exit_atr = DecimalParameter(0.5, 2.0, default=0.70, decimals=2, space="sell")

    fee_buffer = DecimalParameter(0.0008, 0.0020, default=0.0012, decimals=4, space="buy")
    min_pred_move = DecimalParameter(0.0015, 0.0040, default=0.0025, decimals=4, space="buy")

    prob_up_gate = DecimalParameter(0.60, 0.72, default=0.66, decimals=2, space="buy")
    prob_down_gate = DecimalParameter(0.60, 0.72, default=0.66, decimals=2, space="sell")
    prob_exit_gate = DecimalParameter(0.45, 0.55, default=0.48, decimals=2, space="sell")

    slope_gate = DecimalParameter(0.0, 0.0020, default=0.0002, decimals=5, space="buy")

    plot_config = {
        "main_plot": {"ema_trend_1h": {}, "ema_fast_1h": {}, "ema_slow_1h": {}},
        "subplots": {
            "pred_ret": {"pred_ret": {}, "pred_ret_ema": {}},
            "atr_pct": {"atr_pct": {}},
            "rsi_1h": {"rsi_1h": {}},
        },
    }

    @property
    def protections(self) -> List[Dict[str, Any]]:
        """Define exchange-agnostic protections (moved from deprecated config).

        Enable via `--enable-protections` in backtesting/hyperopt.
        """
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 12},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 720,
                "stop_duration_candles": 144,
                "max_allowed_drawdown": 0.20,
            },
        ]

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
            wl = self.dp.current_whitelist()
            return [(p, self.informative_timeframe) for p in wl]
        return []

    def _inject_freqai(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        if hasattr(self, "freqai"):
            return self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe = self._inject_freqai(dataframe, metadata)

        pred_col = self._first_col(dataframe, ["&-prediction", "prediction"])
        do_pred_col = self._first_col(dataframe, ["do_predict", "do_pred"])
        prob_up_col = self._first_col(dataframe, ["&-pred_up_prob", "pred_up_prob", "up_prob", "prob_up"])
        prob_dn_col = self._first_col(dataframe, ["&-pred_dn_prob", "pred_dn_prob", "down_prob", "prob_down"])

        if pred_col:
            # Treat model output as future price and convert to return
            dataframe["pred_ret"] = (dataframe[pred_col] - dataframe["close"]) / dataframe["close"]
            dataframe["pred_ret_ema"] = dataframe["pred_ret"].ewm(
                span=int(self.pred_ema_span.value), adjust=False
            ).mean()
        else:
            dataframe["pred_ret"] = 0.0
            dataframe["pred_ret_ema"] = 0.0

        # 5m ATR%
        atr_5m = self._atr(dataframe, 14)
        dataframe["atr_pct"] = (atr_5m / dataframe["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dataframe["thr"] = float(self.fee_buffer.value) + float(self.k_atr.value) * dataframe["atr_pct"]
        # Chop filter: require some volatility
        dataframe["atr_ok"] = dataframe["atr_pct"] > 0.003
        dataframe["gate_mag"] = dataframe["pred_ret_ema"].abs()

        # 1h regime and buffers
        if self.dp:
            inf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
            inf["ema_trend"] = self._ema(inf["close"], int(self.ema_period.value))
            inf["ema_fast"] = self._ema(inf["close"], int(self.ema_fast_1h.value))
            inf["ema_slow"] = self._ema(inf["close"], int(self.ema_slow_1h.value))
            inf["ema_slow_slope"] = inf["ema_slow"].pct_change().fillna(0.0)
            inf["rsi"] = self._rsi(inf["close"], 14)
            inf["atr_1h"] = self._atr(inf, 14)

            dataframe = merge_informative_pair(
                dataframe, inf, self.timeframe, self.informative_timeframe, ffill=True
            )
            dataframe.rename(
                columns={
                    f"ema_trend_{self.informative_timeframe}": "ema_trend_1h",
                    f"ema_fast_{self.informative_timeframe}": "ema_fast_1h",
                    f"ema_slow_{self.informative_timeframe}": "ema_slow_1h",
                    f"ema_slow_slope_{self.informative_timeframe}": "ema_slow_slope_1h",
                    f"rsi_{self.informative_timeframe}": "rsi_1h",
                    f"atr_1h_{self.informative_timeframe}": "atr_1h",
                },
                inplace=True,
            )
        else:
            dataframe["ema_trend_1h"] = self._ema(dataframe["close"], int(self.ema_period.value))
            dataframe["ema_fast_1h"] = self._ema(dataframe["close"], int(self.ema_fast_1h.value))
            dataframe["ema_slow_1h"] = self._ema(dataframe["close"], int(self.ema_slow_1h.value))
            dataframe["ema_slow_slope_1h"] = dataframe["ema_slow_1h"].pct_change().fillna(0.0)
            dataframe["rsi_1h"] = self._rsi(dataframe["close"], 14)
            dataframe["atr_1h"] = self._atr(dataframe, 14)

        # Probabilities
        has_prob_up = bool(prob_up_col)
        has_prob_dn = bool(prob_dn_col)
        dataframe["prob_up"] = dataframe[prob_up_col].clip(0.0, 1.0) if has_prob_up else 0.5
        dataframe["prob_down"] = dataframe[prob_dn_col].clip(0.0, 1.0) if has_prob_dn else (1.0 - dataframe["prob_up"])

        dataframe["prob_gate_long_ok"] = (
            dataframe["prob_up"] >= float(self.prob_up_gate.value) if has_prob_up else True
        )
        dataframe["prob_gate_short_ok"] = (
            dataframe["prob_down"] >= float(self.prob_down_gate.value) if has_prob_dn else True
        )
        dataframe["prob_exit_long_hit"] = (
            dataframe["prob_up"] < float(self.prob_exit_gate.value) if has_prob_up else False
        )
        dataframe["prob_exit_short_hit"] = (
            dataframe["prob_down"] < float(self.prob_exit_gate.value) if has_prob_dn else False
        )

        dataframe["do_pred"] = dataframe[do_pred_col].fillna(0) if do_pred_col else 1

        # Regime flags
        dataframe["regime_long"] = (
            (dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]) & (dataframe["ema_slow_slope_1h"] > float(self.slope_gate.value))
        )
        dataframe["regime_short"] = (
            (dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]) & (dataframe["ema_slow_slope_1h"] < -float(self.slope_gate.value))
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # Stricter entry: require prediction to exceed threshold by 25% and min magnitude
        cond_long = (
            (dataframe["do_pred"] == 1)
            & (dataframe["pred_ret_ema"] > 1.25 * dataframe["thr"])
            & (dataframe["gate_mag"] > float(self.min_pred_move.value))
            & (dataframe["rsi_1h"] > int(self.rsi_threshold.value))
            & (dataframe["regime_long"])
            & (dataframe["prob_gate_long_ok"])
            & (dataframe["atr_ok"])  # avoid tiny-range chop
            & (dataframe["volume"] > 0)
        )

        cond_short = (
            (dataframe["do_pred"] == 1)
            & (dataframe["pred_ret_ema"] < -1.25 * dataframe["thr"])
            & (dataframe["gate_mag"] > float(self.min_pred_move.value))
            & (dataframe["rsi_1h"] < 50)
            & (dataframe["regime_short"])
            & (dataframe["prob_gate_short_ok"])
            & (dataframe["atr_ok"])  # avoid tiny-range chop
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[cond_long, ["enter_long", "enter_tag"]] = (1, "L_pred>thr_regime")
        dataframe.loc[cond_short, ["enter_short", "enter_tag"]] = (1, "S_pred<thr_regime")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        # Earlier flip: exit on zero-cross of pred_ret_ema or weaker threshold, plus ATR guard on 1h
        half_thr = 0.25 * dataframe["thr"]
        k_exit = float(self.k_exit_atr.value)

        exit_long = (
            ((dataframe["do_pred"] == 1) & (dataframe["pred_ret_ema"] < 0))  # flip
            | (dataframe["pred_ret_ema"] < -half_thr)                        # momentum against us
            | (dataframe["rsi_1h"] < 48)
            | (dataframe["close"] < dataframe["ema_trend_1h"] - k_exit * dataframe["atr_1h"])
            | (dataframe["prob_exit_long_hit"])
        ) & (dataframe["volume"] > 0)

        exit_short = (
            ((dataframe["do_pred"] == 1) & (dataframe["pred_ret_ema"] > 0))
            | (dataframe["pred_ret_ema"] > half_thr)
            | (dataframe["rsi_1h"] > 52)
            | (dataframe["close"] > dataframe["ema_trend_1h"] + k_exit * dataframe["atr_1h"])
            | (dataframe["prob_exit_short_hit"])
        ) & (dataframe["volume"] > 0)

        dataframe.loc[exit_long, ["exit_long", "exit_tag"]] = (1, "L_exit_flip")
        dataframe.loc[exit_short, ["exit_short", "exit_tag"]] = (1, "S_exit_flip")
        return dataframe

    # --------- Time-based exit tied to label horizon ---------
    def _tf_minutes(self) -> int:
        """Return timeframe length in minutes for the current strategy timeframe."""
        tf = str(self.timeframe)
        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return 60 * int(tf[:-1])
        if tf.endswith("d"):
            return 1440 * int(tf[:-1])
        return 5

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> Optional[str]:
        """Time-stop around label horizon to prevent grind.

        Exits after ~1.5x label-period candles since entry.
        """
        try:
            lp = int(self.freqai_info.get("feature_parameters", {}).get("label_period_candles", 24))
        except Exception:
            lp = 24
        max_candles = int(1.5 * lp)
        age_minutes = int((current_time - trade.open_date_utc).total_seconds() / 60)
        age_candles = age_minutes // max(1, self._tf_minutes())
        if age_candles >= max_candles:
            return "time_stop"
        return None

    # --------- FreqAI hooks ---------
    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """Predict future price then convert to return in-strategy. Keep as-is for now."""
        label_period = int(self.freqai_info.get("feature_parameters", {}).get("label_period_candles", 24))
        df = dataframe.copy()
        df["&-prediction"] = df["close"].shift(-label_period)
        return df

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        df = dataframe.copy()
        try:
            if _HAS_TALIB:
                df["%-ema"] = ta.EMA(df["close"], timeperiod=period)
            else:
                df["%-ema"] = df["close"].ewm(span=max(1, period), adjust=False).mean()
        except Exception:
            df["%-ema"] = df["close"].ewm(span=max(1, period), adjust=False).mean()

        try:
            if _HAS_TALIB:
                df["%-rsi"] = ta.RSI(df["close"], timeperiod=period)
            else:
                delta = df["close"].diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.ewm(alpha=1 / max(1, period), adjust=False).mean()
                roll_down = down.ewm(alpha=1 / max(1, period), adjust=False).mean()
                rs = roll_up / roll_down.replace(0, pd.NA)
                df["%-rsi"] = 100 - (100 / (1 + rs))
        except Exception:
            df["%-rsi"] = pd.Series(np.nan, index=df.index)

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
