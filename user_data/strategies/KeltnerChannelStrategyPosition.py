# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
from ast import In
from audioop import mul
from random import vonmisesvariate
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame  # noqa
from datetime import datetime  # noqa
from typing import Optional, Union, List  # noqa
from functools import reduce

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
    RealParameter,
)
from freqtrade.optimize.space import Categorical, Dimension, SKDecimal  # Import necessary classes
from freqtrade.exchange import timeframe_to_minutes

# --------------------------------
# Add your lib to import here
# from freqtrade.strategy import IStrategy, merge_informative_pair


class KeltnerChannelStrategyPosition(IStrategy):
    """
    Keltner Channel strategy using only buy/sell signals without stoploss and protections
    """

    INTERFACE_VERSION = 3

    stoploss = -0.50

    # Trailing stoploss
    trailing_stop = False
    can_short: bool = False
    process_only_new_candles = True
    # Strategy parameters
    kc_window = IntParameter(low=5, high=60, default=14, space="buy")
    kc_mult = DecimalParameter(low=1, high=4, decimals=2, default=2.1, space="buy")
    kc_atrs = IntParameter(low=5, high=60, default=11, space="buy")

    startup_candle_count: int = 60


    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 30,
                "trade_limit": 4,
                "stop_duration_candles": 0,
                "max_allowed_drawdown": 1.0,
            },
        ]

    # Keep only the Keltner Channel calculation methods
    @staticmethod
    def true_range(bars: pd.DataFrame) -> pd.Series:
        high_safe = bars["high"]
        low_safe = bars["low"]
        close_shifted_safe = bars["close"].shift(1)
        hl = high_safe - low_safe
        hc = abs(high_safe - close_shifted_safe)
        lc = abs(low_safe - close_shifted_safe)
        tr = pd.DataFrame({"hl": hl, "hc": hc, "lc": lc}).max(axis=1)
        return tr.fillna(0)

    @staticmethod
    def rma(series: pd.Series, window: int) -> pd.Series:
        alpha = 1.0 / window
        return series.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def myatr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
        tr = KeltnerChannelStrategyPosition.true_range(bars)
        res = KeltnerChannelStrategyPosition.rma(tr, window)
        return pd.Series(res)

    @staticmethod
    def typical_price_ohlc4(bars: pd.DataFrame) -> pd.Series:
        return bars[["open", "high", "low", "close"]].sum(axis=1) / 4

    @staticmethod
    def rolling_weighted_mean(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def keltner_channel(
        bars: pd.DataFrame, window: int = 15, mult: float = 3.2, atrs: int = 30
    ) -> pd.DataFrame:
        typical_price = KeltnerChannelStrategyPosition.typical_price_ohlc4(bars)
        typical_mean = KeltnerChannelStrategyPosition.rolling_weighted_mean(typical_price, window)
        atrval = KeltnerChannelStrategyPosition.myatr(bars, atrs) * mult
        upper = typical_mean + atrval
        lower = typical_mean - atrval
        return pd.DataFrame(
            index=bars.index,
            data={"upper": upper.values, "mid": typical_mean.values, "lower": lower.values},
        )

    @staticmethod
    def crossed_above(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
        series2_aligned = (
            pd.Series(series2, index=series1.index)
            if isinstance(series2, (float, int))
            else series2.reindex(series1.index)
        )
        series1_shifted = series1.shift(1)
        series2_shifted = series2_aligned.shift(1)
        above = (series1 > series2_aligned) & (series1_shifted <= series2_shifted)
        return above.fillna(False)

    @staticmethod
    def crossed_below(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
        series2_aligned = (
            pd.Series(series2, index=series1.index)
            if isinstance(series2, (float, int))
            else series2.reindex(series1.index)
        )
        series1_shifted = series1.shift(1)
        series2_shifted = series2_aligned.shift(1)
        below = (series1 < series2_aligned) & (series1_shifted >= series2_shifted)
        return below.fillna(False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        kc = self.keltner_channel(
            dataframe,
            window=self.kc_window.value,
            mult=self.kc_mult.value,
            atrs=self.kc_atrs.value,
        )
        conditions.append(
            self.crossed_above(
                dataframe["close"], kc["upper"]
            )
        )
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        kc = self.keltner_channel(
            dataframe,
            window=self.kc_window.value,
            mult=self.kc_mult.value,
            atrs=self.kc_atrs.value,
        )
        conditions.append(
            self.crossed_below(
                dataframe["close"], kc["lower"]
            )
        )
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1

        return dataframe
