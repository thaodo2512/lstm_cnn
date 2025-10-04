from __future__ import annotations

# Ichimoku + FreqAI hybrid strategy (long-only)
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
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair  # type: ignore
import numpy as np


class ichiV1(IStrategy):
    """Ichimoku + FreqAI hybrid strategy.

    Notes
    - Subclasses IStrategy (FreqAI integrates via self.freqai.start()).
    - Predictions column name follows the label name set in set_freqai_targets (here: '&-s_close').
    - If FreqAI does not provide '&-s_close_mean'/'&-s_close_std', a rolling mean/std fallback is used.
    """

    # Buy hyperspace params
    buy_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        # NOTE: Good value (Win% ~70%), a lot of trades
        "buy_min_fan_magnitude_gain": 1.002,
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
            "%%-trend_close-period_1": {"color": "#FF5733"},
            "%%-trend_close-period_3": {"color": "#FF8333"},
            "%%-trend_close-period_6": {"color": "#FFB533"},
            "%%-trend_close-period_12": {"color": "#FFE633"},
            "%%-trend_close-period_24": {"color": "#E3FF33"},
            "%%-trend_close-period_48": {"color": "#C4FF33"},
            "%%-trend_close-period_72": {"color": "#61FF33"},
            "%%-trend_close-period_96": {"color": "#33FF7D"},
        },
        "subplots": {
            "fan_magnitude": {"%%-fan_magnitude": {}},
            "fan_magnitude_gain": {"%%-fan_magnitude_gain": {}},
        },
    }

    # -------------------- FreqAI feature hooks --------------------
    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%%-trend_close-period"] = ta.EMA(dataframe["close"], timeperiod=period)
        dataframe["%%-trend_open-period"] = ta.EMA(dataframe["open"], timeperiod=period)
        dataframe["%%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
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
            ccol = f"%%-trend_close-period_{p}"
            ocol = f"%%-trend_open-period_{p}"
            if ccol not in dataframe.columns:
                dataframe[ccol] = _ema(dataframe["close"], p)
            if ocol not in dataframe.columns:
                dataframe[ocol] = _ema(dataframe["open"], p)

        # Fan magnitude and acceleration (safe if columns were just created)
        dataframe["%%-fan_magnitude"] = (
            dataframe["%%-trend_close-period_12"] / dataframe["%%-trend_close-period_96"]
        )
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

        conditions = []

        # Integrate FreqAI prediction
        if "do_predict" in df.columns:
            conditions.append(df["do_predict"] == 1)

        # Fallback for target statistics if not provided by FreqAI
        if "&-s_close_mean" not in df.columns or "&-s_close_std" not in df.columns:
            pred = df.get("&-s_close", pd.Series(np.nan, index=df.index))
            roll = pred.rolling(100, min_periods=10)
            df["&-s_close_mean"] = roll.mean()
            df["&-s_close_std"] = roll.std(ddof=0)

        # Thresholded target
        df["target_roi"] = df["&-s_close_mean"] + df["&-s_close_std"] * 1.5
        if "&-s_close" in df.columns:
            conditions.append(df["&-s_close"] > df["target_roi"])

        # Trending market above cloud
        level = int(self.buy_params["buy_trend_above_senkou_level"])
        if level >= 1:
            conditions.append(df["%%-trend_close-period_1"] > df["%%-senkou_a"])
            conditions.append(df["%%-trend_close-period_1"] > df["%%-senkou_b"])
        if level >= 2:
            conditions.append(df["%%-trend_close-period_3"] > df["%%-senkou_a"])
            conditions.append(df["%%-trend_close-period_3"] > df["%%-senkou_b"])
        if level >= 3:
            conditions.append(df["%%-trend_close-period_6"] > df["%%-senkou_a"])
            conditions.append(df["%%-trend_close-period_6"] > df["%%-senkou_b"])
        if level >= 4:
            conditions.append(df["%%-trend_close-period_12"] > df["%%-senkou_a"])
            conditions.append(df["%%-trend_close-period_12"] > df["%%-senkou_b"])
        if level >= 5:
            conditions.append(df["%%-trend_close-period_24"] > df["%%-senkou_a"])
            conditions.append(df["%%-trend_close-period_24"] > df["%%-senkou_b"])
        if level >= 6:
            conditions.append(df["%%-trend_close-period_48"] > df["%%-senkou_a"])
            conditions.append(df["%%-trend_close-period_48"] > df["%%-senkou_b"])
        if level >= 7:
            conditions.append(df["%%-trend_close-period_72"] > df["%%-senkou_a"])
            conditions.append(df["%%-trend_close-period_72"] > df["%%-senkou_b"])
        if level >= 8:
            conditions.append(df["%%-trend_close-period_96"] > df["%%-senkou_a"])
            conditions.append(df["%%-trend_close-period_96"] > df["%%-senkou_b"])

        # Trends bullish (close EMA above open EMA)
        bull_level = int(self.buy_params["buy_trend_bullish_level"])
        if bull_level >= 1:
            conditions.append(df["%%-trend_close-period_1"] > df["%%-trend_open-period_1"])
        if bull_level >= 2:
            conditions.append(df["%%-trend_close-period_3"] > df["%%-trend_open-period_3"])
        if bull_level >= 3:
            conditions.append(df["%%-trend_close-period_6"] > df["%%-trend_open-period_6"])
        if bull_level >= 4:
            conditions.append(df["%%-trend_close-period_12"] > df["%%-trend_open-period_12"])
        if bull_level >= 5:
            conditions.append(df["%%-trend_close-period_24"] > df["%%-trend_open-period_24"])
        if bull_level >= 6:
            conditions.append(df["%%-trend_close-period_48"] > df["%%-trend_open-period_48"])
        if bull_level >= 7:
            conditions.append(df["%%-trend_close-period_72"] > df["%%-trend_open-period_72"])
        if bull_level >= 8:
            conditions.append(df["%%-trend_close-period_96"] > df["%%-trend_open-period_96"])

        # Fan magnitude acceleration
        conditions.append(df["%%-fan_magnitude_gain"] >= self.buy_params["buy_min_fan_magnitude_gain"])
        conditions.append(df["%%-fan_magnitude"] > 1)
        for x in range(int(self.buy_params["buy_fan_magnitude_shift_value"])):
            conditions.append(df["%%-fan_magnitude"].shift(x + 1) < df["%%-fan_magnitude"])

        if conditions:
            df.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df["exit_long"] = 0

        trend_indicator = self.sell_params["sell_trend_indicator"]
        period = int(self.trend_period_map[trend_indicator])
        cond = qtpylib.crossed_below(
            df["%%-trend_close-period_1"], df[f"%%-trend_close-period_{period}"]
        )
        df.loc[cond.fillna(False), "exit_long"] = 1
        return df
