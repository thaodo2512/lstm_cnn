from __future__ import annotations

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class FreqAIHybridExample(IStrategy):
    """
    Minimal example strategy that consumes FreqAI predictions.

    Expects FreqAI to add either:
      - 'prediction' (regression, next-step price), or
      - 'pred_up_prob' (classification probability for up move)
    """

    timeframe = "1h"
    minimal_roi = {"0": 0.02}
    stoploss = -0.1

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Predictions are injected by FreqAI. Nothing custom here.
        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["buy"] = 0
        if "prediction" in df.columns:
            df.loc[(df["prediction"] > df["close"] * 1.001), "buy"] = 1
        elif "pred_up_prob" in df.columns:
            df.loc[(df["pred_up_prob"] > 0.55), "buy"] = 1
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["sell"] = 0
        if "prediction" in df.columns:
            df.loc[(df["prediction"] < df["close"] * 0.999), "sell"] = 1
        elif "pred_up_prob" in df.columns:
            df.loc[(df["pred_up_prob"] < 0.45), "sell"] = 1
        return df

