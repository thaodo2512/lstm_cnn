from __future__ import annotations

# Ichimoku + FreqAI hybrid strategy
# - Moves indicators into FreqAI feature hooks
# - Uses FreqAI predictions to gate entries
# - Trend filters via Ichimoku cloud and EMA fans

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import json
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

    # ROI table (trend-following, avoid fast decay to ~0)
    minimal_roi = {
        # Keep some profit expectation early, decay slowly over hours/days
        "0": 0.02,     # 2.0% immediately
        "360": 0.012,  # after 6h
        "1440": 0.004, # after 24h
        "4320": 0.0,   # after 3d
    }

    # Stoploss (tighten to cap downside; trailing will manage winners)
    stoploss = -0.03

    # Optimal timeframe for the strategy
    timeframe = "5m"

    startup_candle_count = 96
    # FreqAI requires new-candle processing in live mode
    process_only_new_candles = True

    trailing_stop = True
    trailing_stop_positive = 0.004
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    # Let ROI/trailing govern profit-taking, avoid early losses via exit_signal
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    # Enable shorting (requires futures-capable exchange/config)
    # Disabled by default; can re-enable via env gates below
    can_short: bool = False

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

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Make short capability runtime-togglable via env.
        # If you set ICHI_ENABLE_SHORT=true in .env / compose env, shorts are allowed.
        self.can_short = self._env_bool("ICHI_ENABLE_SHORT", False)

    @property
    def protections(self):
        # Define protections in-strategy (config key deprecated in recent Freqtrade)
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 12},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 144,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.06,
            },
        ]

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
        """Add Ichimoku and ensure EMA fan features exist.

        Uses periods from freqai_info.feature_parameters.indicator_periods_candles
        if available; falls back to [1, 3, 6, 12, 24, 48, 72, 96].
        """
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

        # Derive dynamic period list
        periods = (
            list(self.freqai_info.get("feature_parameters", {}).get("indicator_periods_candles", []))
            if hasattr(self, "freqai_info") else []
        )
        if not periods:
            periods = [1, 3, 6, 12, 24, 48, 72, 96]
        periods = sorted(set(int(p) for p in periods))

        for p in periods:
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
        # Use nearest available periods to defaults fast=12, slow=96
        def _nearest(target: int, avail: list[int]) -> int:
            if target in avail:
                return target
            return min(avail, key=lambda x: abs(x - target)) if avail else target

        p_fast = _nearest(12, periods)
        p_slow = _nearest(96, periods)
        # Avoid identical periods (choose max as slow, min as fast)
        if p_fast == p_slow and periods:
            p_fast = min(periods)
            p_slow = max(periods)
        fast_col = f"%%-trend_close_period_{p_fast}"
        slow_col = f"%%-trend_close_period_{p_slow}"
        # Guard against zero/NaN denominator
        den = dataframe[slow_col].replace(0, 1e-8)
        dataframe["%%-fan_magnitude"] = dataframe[fast_col] / den
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
        # Log full runtime config once per pair/timeframe for debugging
        self._log_config_once(metadata)
        if self._env_bool("ICHI_USE_HA", True):
            heikinashi = qtpylib.heikinashi(dataframe)
            dataframe["open"] = heikinashi["open"]
            # Optionally use HA close as well (disabled by default)
            if self._env_bool("ICHI_USE_HA_CLOSE", False):
                dataframe["close"] = heikinashi["close"]
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
        periods = (
            list(self.freqai_info.get("feature_parameters", {}).get("indicator_periods_candles", []))
            if hasattr(self, "freqai_info") else []
        )
        if not periods:
            periods = [1, 3, 6, 12, 24, 48, 72, 96]
        periods = sorted(set(int(p) for p in periods))
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

        # Optional volume filter
        if self._env_bool("ICHI_VOLUME_FILTER", False):
            vol_win = max(1, self._env_int("ICHI_VOL_WINDOW", 20))
            vol_mult = max(0.0, float(self._env_float("ICHI_VOL_MULT", 1.0)))
            vol_ma = df["volume"].rolling(vol_win, min_periods=1).mean()
            conditions.append(df["volume"] > vol_ma * vol_mult)

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

        # EMA trend cross (legacy)
        # Map requested periods to nearest available to avoid KeyErrors when
        # indicator_periods_candles does not include exact values like 1 or 24
        periods = (
            list(self.freqai_info.get("feature_parameters", {}).get("indicator_periods_candles", []))
            if hasattr(self, "freqai_info") else []
        )
        if not periods:
            periods = [1, 3, 6, 12, 24, 48, 72, 96]
        periods = sorted(set(int(p) for p in periods))
        def _nearest(target: int, avail: list[int]) -> int:
            if target in avail:
                return target
            return min(avail, key=lambda x: abs(x - target)) if avail else target
        p_fast = _nearest(1, periods)
        p_slow = _nearest(period, periods)
        ema_cross_down = qtpylib.crossed_below(
            df[f"%%-trend_close_period_{p_fast}"], df[f"%%-trend_close_period_{p_slow}"]
        )

        # Ichimoku exits: Tenkan crosses below Kijun OR close below Kijun
        tenkan = df.get("%%-tenkan_sen")
        kijun = df.get("%%-kijun_sen")
        ichi_cross_down = (
            qtpylib.crossed_below(tenkan, kijun) if tenkan is not None and kijun is not None else pd.Series(False, index=df.index)
        )
        close_below_kijun = (
            (df["close"] < kijun) if kijun is not None else pd.Series(False, index=df.index)
        )

        # Exit mode: ema | ichi | kijun | any | all
        exit_mode = self._env_str("ICHI_EXIT_MODE", "any").strip().lower()
        if exit_mode == "ema":
            cond_long = ema_cross_down
        elif exit_mode in ("ichi", "tenkan_kijun_cross"):
            cond_long = (ichi_cross_down | close_below_kijun)
        elif exit_mode == "kijun":
            cond_long = close_below_kijun
        elif exit_mode == "all":
            cond_long = (ema_cross_down & ichi_cross_down & close_below_kijun)
        else:  # any
            cond_long = (ema_cross_down | ichi_cross_down | close_below_kijun)
        df.loc[cond_long.fillna(False), "exit_long"] = 1

        # Short exit (optional via ICHI_ENABLE_SHORT): crossed above inverses
        if self._env_bool("ICHI_ENABLE_SHORT", True):
            ema_cross_up = qtpylib.crossed_above(
                df[f"%%-trend_close_period_{p_fast}"], df[f"%%-trend_close_period_{p_slow}"]
            )
            ichi_cross_up = (
                qtpylib.crossed_above(tenkan, kijun) if tenkan is not None and kijun is not None else pd.Series(False, index=df.index)
            )
            close_above_kijun = (
                (df["close"] > kijun) if kijun is not None else pd.Series(False, index=df.index)
            )
            cond_short = (ema_cross_up | ichi_cross_up | close_above_kijun)
            df.loc[cond_short.fillna(False), "exit_short"] = 1
        # Optional debug export for inspection
        self._maybe_debug_export(df, metadata)
        return df

    # ---------------- Debug helpers ----------------
    @staticmethod
    def _safe_name(text: str) -> str:
        try:
            return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text))
        except Exception:
            return "unknown"

    _debug_written_keys: set = set()

    def _maybe_debug_export(self, dataframe: DataFrame, metadata: dict) -> None:
        """If ICHI_DEBUG_EXPORT=true, write a compact CSV with predictions, gates, and
        final signals to /freqtrade/user_data/backtest_results.
        """
        flag = os.getenv("ICHI_DEBUG_EXPORT", "").strip().lower()
        if flag not in {"1", "true", "yes", "on"}:
            return
        try:
            pair = metadata.get("pair", "unknown_pair")
            timeframe = getattr(self, "timeframe", "tf")
            key = f"{pair}|{timeframe}"
            # Write once per pair/timeframe per run
            if key in self._debug_written_keys:
                return

            cols = []
            # price/volume
            cols += [c for c in ["open", "high", "low", "close", "volume"] if c in dataframe.columns]
            # predictions
            cols += [c for c in ["&-s_close", "&-s_close_mean", "&-s_close_std", "target_roi", "do_predict"] if c in dataframe.columns]
            # ichimoku
            cols += [c for c in [
                "%-pct-change",
                "%%-tenkan_sen", "%%-kijun_sen", "%%-senkou_a", "%%-senkou_b",
                "%%-cloud_green", "%%-cloud_red",
            ] if c in dataframe.columns]
            # trend / fan
            for p in [1,3,6,12,24,48,72,96]:
                tc = f"%%-trend_close_period_{p}"
                to = f"%%-trend_open_period_{p}"
                if tc in dataframe.columns: cols.append(tc)
                if to in dataframe.columns: cols.append(to)
            cols += [c for c in ["%%-fan_magnitude", "%%-fan_magnitude_gain"] if c in dataframe.columns]
            # signals
            cols += [c for c in ["enter_long", "enter_short", "exit_long", "exit_short"] if c in dataframe.columns]

            debug_df = dataframe[cols].copy() if cols else dataframe.copy()

            # Limit rows if requested
            try:
                max_rows = int(os.getenv("ICHI_DEBUG_MAX_ROWS", "0"))
            except Exception:
                max_rows = 0
            if max_rows and len(debug_df) > max_rows:
                debug_df = debug_df.tail(max_rows)

            from pathlib import Path
            out_dir = Path("/freqtrade/user_data/backtest_results")
            out_dir.mkdir(parents=True, exist_ok=True)
            pair_s = self._safe_name(pair)
            tf_s = self._safe_name(timeframe)
            start = debug_df.index.min()
            end = debug_df.index.max()
            start_s = self._safe_name(str(start)[:19].replace(" ", "T")) if start is not None else "start"
            end_s = self._safe_name(str(end)[:19].replace(" ", "T")) if end is not None else "end"
            out_path = out_dir / f"ichi_debug_{pair_s}_{tf_s}_{start_s}_{end_s}.csv"
            debug_df.to_csv(out_path)
            self._debug_written_keys.add(key)
        except Exception:
            # Never interrupt strategy execution due to debugging
            return

    # ---------------- Config logging ----------------
    _config_logged_keys: set = set()

    def _log_config_once(self, metadata: dict) -> None:
        """Print a compact view of ichiV1 configuration and env overrides once.

        Toggle via env ICHI_LOG_CONFIG (default: true). Avoids log spam by logging
        once per pair/timeframe.
        """
        if not self._env_bool("ICHI_LOG_CONFIG", True):
            return
        try:
            pair = metadata.get("pair", "unknown_pair")
            timeframe = getattr(self, "timeframe", "tf")
            key = f"{pair}|{timeframe}"
            if key in self._config_logged_keys:
                return

            # Pull dynamic periods for fan/trend
            periods = (
                list(self.freqai_info.get("feature_parameters", {}).get("indicator_periods_candles", []))
                if hasattr(self, "freqai_info") else []
            )
            if not periods:
                periods = [1, 3, 6, 12, 24, 48, 72, 96]

            env_overrides = {
                "ICHI_STD_MULT": self._env_float("ICHI_STD_MULT", 0.5),
                "ICHI_CLOUD_LEVEL": self._env_int("ICHI_CLOUD_LEVEL", int(self.buy_params["buy_trend_above_senkou_level"])),
                "ICHI_BULLISH_LEVEL": self._env_int("ICHI_BULLISH_LEVEL", int(self.buy_params["buy_trend_bullish_level"])),
                "ICHI_FAN_SHIFT": self._env_int("ICHI_FAN_SHIFT", int(self.buy_params["buy_fan_magnitude_shift_value"])),
                "ICHI_FAN_GAIN": self._env_float("ICHI_FAN_GAIN", float(self.buy_params["buy_min_fan_magnitude_gain"])),
                "ICHI_REQUIRE_DOPREDICT": self._env_bool("ICHI_REQUIRE_DOPREDICT", True),
                "ICHI_USE_HA": self._env_bool("ICHI_USE_HA", True),
                "ICHI_USE_HA_CLOSE": self._env_bool("ICHI_USE_HA_CLOSE", False),
                "ICHI_ENABLE_SHORT": self._env_bool("ICHI_ENABLE_SHORT", False),
                "ICHI_TEST_LOOSE": self._env_bool("ICHI_TEST_LOOSE", False),
                "ICHI_TEST_THR": self._env_float("ICHI_TEST_THR", 0.0),
                "ICHI_SELL_TREND": self._env_str("ICHI_SELL_TREND", self.sell_params["sell_trend_indicator"]),
                "ICHI_EXIT_MODE": self._env_str("ICHI_EXIT_MODE", "any"),
                "ICHI_VOLUME_FILTER": self._env_bool("ICHI_VOLUME_FILTER", False),
                "ICHI_VOL_WINDOW": self._env_int("ICHI_VOL_WINDOW", 20),
                "ICHI_VOL_MULT": self._env_float("ICHI_VOL_MULT", 1.0),
            }

            freqai_cfg = getattr(self, "freqai_info", {})
            cfg = {
                "pair": pair,
                "timeframe": timeframe,
                "startup_candle_count": self.startup_candle_count,
                "stoploss": self.stoploss,
                "trailing": {
                    "enabled": bool(self.trailing_stop),
                    "positive": getattr(self, "trailing_stop_positive", None),
                    "offset": getattr(self, "trailing_stop_positive_offset", None),
                    "only_offset_reached": getattr(self, "trailing_only_offset_is_reached", None),
                },
                "minimal_roi": self.minimal_roi,
                "can_short": bool(self.can_short),
                "buy_params": self.buy_params,
                "sell_params": self.sell_params,
                "indicator_periods": periods,
                "env_overrides": env_overrides,
                "freqai_info": {
                    k: freqai_cfg.get(k)
                    for k in ("feature_parameters", "data_split_parameters", "fit_live_predictions_candles")
                    if isinstance(freqai_cfg, dict)
                },
            }
            payload = json.dumps(cfg, default=str, indent=2)
            self.logger.info("ichiV1 config (%%s %%s) =>\n%%s", pair, timeframe, payload)
            self._config_logged_keys.add(key)
        except Exception:
            # Never block execution due to logging issues
            return
