from __future__ import annotations

# Ichimoku + FreqAI hybrid strategy
# - Moves indicators into FreqAI feature hooks
# - Uses FreqAI predictions to gate entries
# - Trend filters via Ichimoku cloud and EMA fans

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta  # type: ignore
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa

pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt  # type: ignore
from functools import reduce
import os


class ichiV1(IStrategy):
    """Ichimoku + FreqAI hybrid strategy.

    Notes
    - Subclasses IStrategy (FreqAI integrates via self.freqai.start()).
    - Predictions column name follows the label name set in set_freqai_targets (here: '&-s_close').
    - If FreqAI does not provide '&-s_close_mean'/'&-s_close_std', a rolling mean/std fallback is used.
    """

    # Buy hyperspace params (can be overridden via env, see _env_* helpers)
    buy_params = {
        "buy_trend_above_senkou_level": 1,
        # Loosened default to help validate signals; override with ICHI_BULLISH_LEVEL
        "buy_trend_bullish_level": 2,
        # Require fewer consecutive increases; override with ICHI_FAN_SHIFT
        "buy_fan_magnitude_shift_value": 1,
        # Slightly easier acceleration gate; override with ICHI_FAN_GAIN
        "buy_min_fan_magnitude_gain": 1.001,
        # "buy_min_fan_magnitude_gain": 1.008,  # very safe (Win% ~90%), fewer trades
    }

    # Sell hyperspace params
    sell_params = {
        # Use EMA trend cross for exits
        "sell_trend_indicator": "trend_close_2h",
    }

    # ROI table
    minimal_roi = {
        "0": 0.059,
        "10": 0.037,
        "41": 0.012,
        "114": 0,
    }

    # Stoploss
    stoploss = -0.275

    # Optimal timeframe for the strategy
    timeframe = "5m"

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = False
    # trailing_stop_positive = 0.002
    # trailing_stop_positive_offset = 0.025
    # trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    # Enable shorting (requires futures-capable exchange/config)
    can_short: bool = True

    trend_period_map = {
        "trend_close_5m": 1,
        "trend_close_15m": 3,
        "trend_close_30m": 6,
        "trend_close_1h": 12,
        "trend_close_2h": 24,
        "trend_close_4h": 48,
        "trend_close_6h": 72,
        "trend_close_8h": 96,
    }

    freqai_info = {
        "model_per_pair": True,
        "feature_parameters": {
            "include_timeframes": ["5m", "15m", "30m", "1h", "2h", "4h"],
            "include_shifted_candles": 3,
            "include_corr_pairlist": [],
            "indicator_periods_candles": [1, 3, 6, 12, 24, 48, 72, 96],
            "label_period_candles": 24,
        },
        "data_split_parameters": {
            "test_size": 0.25,
        },
        "fit_live_predictions_candles": 300,
    }

    plot_config = {
        "main_plot": {
            # fill area between senkou_a and senkou_b
            "%%-senkou_a": {
                "color": "green",  # optional
                "fill_to": "%%-senkou_b",
                "fill_label": "Ichimoku Cloud",  # optional
                "fill_color": "rgba(255,76,46,0.2)",  # optional
            },
            # plot senkou_b, too. Not only the area to it.
            "%%-senkou_b": {},
            "%%-trend_close_period_1": {"color": "#FF5733"},
            "%%-trend_close_period_3": {"color": "#FF8333"},
            "%%-trend_close_period_6": {"color": "#FFB533"},
            "%%-trend_close_period_12": {"color": "#FFE633"},
            "%%-trend_close_period_24": {"color": "#E3FF33"},
            "%%-trend_close_period_48": {"color": "#C4FF33"},
            "%%-trend_close_period_72": {"color": "#61FF33"},
            "%%-trend_close_period_96": {"color": "#33FF7D"},
        },
        "subplots": {
            "fan_magnitude": {"%%-fan_magnitude": {}},
            "fan_magnitude_gain": {"%%-fan_magnitude_gain": {}},
        },
    }

    # -------------------- Env helpers --------------------
    @staticmethod
    def _env_float(key: str, default: float) -> float:
        try:
            v = os.getenv(key)
            return float(v) if v not in (None, "") else default
        except Exception:
            return default

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        try:
            v = os.getenv(key)
            return int(v) if v not in (None, "") else default
        except Exception:
            return default

    @staticmethod
    def _env_bool(key: str, default: bool) -> bool:
        v = os.getenv(key)
        if v is None or v == "":
            return default
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _env_str(key: str, default: str) -> str:
        v = os.getenv(key)
        return v if v not in (None, "") else default

    # -------------------- FreqAI feature hooks --------------------
    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        # Write explicit period-suffixed columns to avoid naming ambiguity across periods
        dataframe[f"%%-trend_close_period_{period}"] = ta.EMA(
            dataframe["close"], timeperiod=period
        )
        dataframe[f"%%-trend_open_period_{period}"] = ta.EMA(
            dataframe["open"], timeperiod=period
        )
        dataframe[f"%%-atr_period_{period}"] = ta.ATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        if "date" in dataframe.columns:
            dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
            dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25

        ichimoku = ftt.ichimoku(
            dataframe,
            conversion_line_period=20,
            base_line_periods=60,
            laggin_span=120,
            displacement=30,
        )
        dataframe["%%-chikou_span"] = ichimoku["chikou_span"]
        dataframe["%%-tenkan_sen"] = ichimoku["tenkan_sen"]
        dataframe["%%-kijun_sen"] = ichimoku["kijun_sen"]
        dataframe["%%-senkou_a"] = ichimoku["senkou_span_a"]
        dataframe["%%-senkou_b"] = ichimoku["senkou_span_b"]
        dataframe["%%-leading_senkou_span_a"] = ichimoku["leading_senkou_span_a"]
        dataframe["%%-leading_senkou_span_b"] = ichimoku["leading_senkou_span_b"]
        dataframe["%%-cloud_green"] = ichimoku["cloud_green"]
        dataframe["%%-cloud_red"] = ichimoku["cloud_red"]

        # Ensure required EMA fan columns exist even if expand_all hasn't populated them yet
        def _ema(series: pd.Series, n: int) -> pd.Series:
            try:
                return ta.EMA(series, timeperiod=n)
            except Exception:
                return series.ewm(span=max(1, n), adjust=False).mean()

        for p in [1, 3, 6, 12, 24, 48, 72, 96]:
            # Prefer underscore naming; also support legacy hyphen names by aliasing
            ccol = f"%%-trend_close_period_{p}"
            ocol = f"%%-trend_open_period_{p}"
            if ccol not in dataframe.columns:
                legacy = f"%%-trend_close-period_{p}"
                if legacy in dataframe.columns:
                    dataframe[ccol] = dataframe[legacy]
                else:
                    dataframe[ccol] = _ema(dataframe["close"], p)
            if ocol not in dataframe.columns:
                legacy = f"%%-trend_open-period_{p}"
                if legacy in dataframe.columns:
                    dataframe[ocol] = dataframe[legacy]
                else:
                    dataframe[ocol] = _ema(dataframe["open"], p)

        # Fan magnitude and acceleration (safe if columns were just created)
        # Guard against zero/NaN denominator
        den = dataframe["%%-trend_close_period_96"].replace(0, 1e-8)
        dataframe["%%-fan_magnitude"] = dataframe["%%-trend_close_period_12"] / den
        dataframe["%%-fan_magnitude_gain"] = (
            dataframe["%%-fan_magnitude"] / dataframe["%%-fan_magnitude"].shift(1)
        )

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        # Predict forward return over the label_period (mean-close ratio - 1)
        lp = int(self.freqai_info["feature_parameters"]["label_period_candles"])
        dataframe["&-s_close"] = (
            dataframe["close"].shift(-lp).rolling(lp).mean() / dataframe["close"] - 1
        )
        return dataframe

    # -------------------- Strategy hooks --------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self._env_bool("ICHI_USE_HA", True):
            heikinashi = qtpylib.heikinashi(dataframe)
            dataframe["open"] = heikinashi["open"]
            # dataframe['close'] = heikinashi['close']
            dataframe["high"] = heikinashi["high"]
            dataframe["low"] = heikinashi["low"]

        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df["enter_long"] = 0
        df["enter_short"] = 0

        conditions = []

        # Integrate FreqAI prediction
        require_dp = self._env_bool("ICHI_REQUIRE_DOPREDICT", True)
        if require_dp and "do_predict" in df.columns:
            conditions.append(df["do_predict"] == 1)

        # Test mode: minimal gates to force trades for validation
        if self._env_bool("ICHI_TEST_LOOSE", False):
            thr = self._env_float("ICHI_TEST_THR", 0.0)
            if "&-s_close" in df.columns:
                long_cond = df["&-s_close"] > thr
                short_cond = df["&-s_close"] < -thr
            else:
                # Fallback if prediction column missing: simple momentum
                long_cond = df["close"] > df["close"].shift(1)
                short_cond = df["close"] < df["close"].shift(1)
            if require_dp and "do_predict" in df.columns:
                long_cond = long_cond & (df["do_predict"] == 1)
                short_cond = short_cond & (df["do_predict"] == 1)
            df.loc[long_cond.fillna(False), "enter_long"] = 1
            if self._env_bool("ICHI_ENABLE_SHORT", True):
                df.loc[short_cond.fillna(False), "enter_short"] = 1
            return df

        # Fallback for target statistics if not provided by FreqAI
        if "&-s_close_mean" not in df.columns or "&-s_close_std" not in df.columns:
            pred = df.get("&-s_close", pd.Series(float("nan"), index=df.index))
            roll = pred.rolling(100, min_periods=5)
            df["&-s_close_mean"] = roll.mean()
            df["&-s_close_std"] = roll.std(ddof=0)

        # Thresholded target
        std_mult = self._env_float("ICHI_STD_MULT", 0.5)
        df["target_roi"] = df["&-s_close_mean"] + df["&-s_close_std"] * std_mult
        if "&-s_close" in df.columns:
            conditions.append(df["&-s_close"] > df["target_roi"])

        # Trending market above cloud
        level = self._env_int(
            "ICHI_CLOUD_LEVEL", int(self.buy_params["buy_trend_above_senkou_level"])
        )
        periods = [1, 3, 6, 12, 24, 48, 72, 96]
        for p in periods[: max(0, min(level, len(periods)))]:
            conditions.append(df[f"%%-trend_close_period_{p}"] > df["%%-senkou_a"])
            conditions.append(df[f"%%-trend_close_period_{p}"] > df["%%-senkou_b"])

        # Trends bullish (close EMA above open EMA)
        bull_level = self._env_int(
            "ICHI_BULLISH_LEVEL", int(self.buy_params["buy_trend_bullish_level"])
        )
        for p in periods[: max(0, min(bull_level, len(periods)))]:
            conditions.append(df[f"%%-trend_close_period_{p}"] > df[f"%%-trend_open_period_{p}"])

        # Fan magnitude acceleration
        fan_gain = float(
            self._env_float(
                "ICHI_FAN_GAIN", float(self.buy_params["buy_min_fan_magnitude_gain"]) 
            )
        )
        # Clamp to avoid equality/zero-division edge cases
        fan_gain = max(1.0001, fan_gain)
        conditions.append(df["%%-fan_magnitude_gain"] >= fan_gain)
        conditions.append(df["%%-fan_magnitude"] > 1)
        fan_shift = int(
            self._env_int(
                "ICHI_FAN_SHIFT", int(self.buy_params["buy_fan_magnitude_shift_value"]) 
            )
        )
        fm = df["%%-fan_magnitude"].ffill().fillna(1.0)
        for x in range(fan_shift):
            conditions.append(fm.shift(x + 1) < fm)

        if conditions:
            df.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        # ---------------- Short entry (optional via ICHI_ENABLE_SHORT) ----------------
        if self._env_bool("ICHI_ENABLE_SHORT", True):
            short_conditions = []
            if require_dp and "do_predict" in df.columns:
                short_conditions.append(df["do_predict"] == 1)

            # Use same computed target_roi magnitude, require negative pred less than -target
            if "&-s_close" in df.columns:
                short_conditions.append(df["&-s_close"] < -df["target_roi"])

            # Below cloud by level
            for p in periods[: max(0, min(level, len(periods)))]:
                short_conditions.append(df[f"%%-trend_close_period_{p}"] < df["%%-senkou_a"])
                short_conditions.append(df[f"%%-trend_close_period_{p}"] < df["%%-senkou_b"])

            # Bearish EMAs (close EMA below open EMA)
            for p in periods[: max(0, min(bull_level, len(periods)))]:
                short_conditions.append(df[f"%%-trend_close_period_{p}"] < df[f"%%-trend_open_period_{p}"])

            # Fan decreasing and below 1
            short_conditions.append(df["%%-fan_magnitude"] < 1)
            # Mirror gain threshold: require contraction
            short_conditions.append(df["%%-fan_magnitude_gain"] <= (1.0 / fan_gain))
            for x in range(fan_shift):
                short_conditions.append(fm.shift(x + 1) > fm)

            if short_conditions:
                df.loc[reduce(lambda x, y: x & y, short_conditions), "enter_short"] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df["exit_long"] = 0
        df["exit_short"] = 0

        sell_indicator = self._env_str(
            "ICHI_SELL_TREND", self.sell_params["sell_trend_indicator"]
        )
        period = int(
            self.trend_period_map.get(
                sell_indicator, self.trend_period_map[self.sell_params["sell_trend_indicator"]]
            )
        )
        cond = qtpylib.crossed_below(
            df["%%-trend_close_period_1"], df[f"%%-trend_close_period_{period}"]
        )
        df.loc[cond.fillna(False), "exit_long"] = 1
        # Short exit (optional via ICHI_ENABLE_SHORT): crossed above inverse
        if self._env_bool("ICHI_ENABLE_SHORT", True):
            cond_s = qtpylib.crossed_above(
                df["%%-trend_close_period_1"], df[f"%%-trend_close_period_{period}"]
            )
            df.loc[cond_s.fillna(False), "exit_short"] = 1
        return df
