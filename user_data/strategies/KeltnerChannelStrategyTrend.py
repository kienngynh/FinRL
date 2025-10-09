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
from typing import Optional, Union, List, Dict, Any  # noqa
from functools import reduce
import json
from pathlib import Path

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
    RealParameter,
)
from freqtrade.optimize.space import (
    Categorical,
    Dimension,
    Integer,
    Real,
    SKDecimal,
)  # Import necessary classes
from freqtrade.exchange import timeframe_to_minutes

# --------------------------------
# Add your lib to import here
# from freqtrade.strategy import IStrategy, merge_informative_pair


class KeltnerChannelStrategyTrend(IStrategy):
    """
    Keltner Channel strategy using only buy/sell signals without stoploss and protections
    MODIFIED: This strategy now calculates signals based on market cap dominance,
    replicating the logic from trading_bot.py for accurate trend hyper-optimization.
    """

    INTERFACE_VERSION = 3

    process_only_new_candles = True

    # Keep only the essential buy/sell parameters
    kc_window = IntParameter(low=5, high=50, default=14, space="buy")
    kc_mult = DecimalParameter(low=1, high=4, decimals=1, default=2.1, space="buy")
    kc_atrs = IntParameter(low=5, high=50, default=11, space="buy")

    startup_candle_count: int = 60

    # --- Cached portfolio data ---
    _portfolio_config: Optional[Dict[str, Any]] = None
    _global_status: Optional[Dict[str, Any]] = None

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

    # --- Hyperopt space definition ---
    @staticmethod
    def hyperopt_space():
        return [
            Integer(5, 50, name="kc_window"),
            SKDecimal(1, 5, decimals=1, name="kc_mult"),
            Integer(5, 50, name="kc_atrs"),
        ]

    # --- Configuration Loading ---
    def _load_config_files(self):
        """Load portfolio and status configs if not already cached."""
        if self._portfolio_config is not None and self._global_status is not None:
            return

        try:
            portfolio_config_path = Path("./portfolio/configs/portfolio_config.json")
            if portfolio_config_path.exists():
                with portfolio_config_path.open() as f:
                    self._portfolio_config = json.load(f)
            else:
                # Log or raise an error if the config is essential
                return

            paths_cfg = self._portfolio_config.get("paths", {})
            global_status_path_str = paths_cfg.get("global_status_file_path")
            if global_status_path_str:
                global_status_path = Path(global_status_path_str)
                if global_status_path.exists():
                    with global_status_path.open() as f:
                        self._global_status = json.load(f)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Handle cases where files are missing or corrupt during a run
            # For hyperopt, it might be acceptable to fail gracefully
            print(f"Could not load required config files: {e}")
            self._portfolio_config = None
            self._global_status = None

    # --- Keltner Channel Calculation Methods (Unchanged) ---
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
        tr = KeltnerChannelStrategyTrend.true_range(bars)
        res = KeltnerChannelStrategyTrend.rma(tr, window)
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
        typical_price = KeltnerChannelStrategyTrend.typical_price_ohlc4(bars)
        typical_mean = KeltnerChannelStrategyTrend.rolling_weighted_mean(
            typical_price, window
        )
        atrval = KeltnerChannelStrategyTrend.myatr(bars, atrs) * mult
        upper = typical_mean + atrval
        lower = typical_mean - atrval
        return pd.DataFrame(
            index=bars.index,
            data={
                "upper": upper.values,
                "mid": typical_mean.values,
                "lower": lower.values,
            },
        )

    @staticmethod
    def crossed_above(
        series1: pd.Series, series2: Union[pd.Series, float, int]
    ) -> pd.Series:
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
    def crossed_below(
        series1: pd.Series, series2: Union[pd.Series, float, int]
    ) -> pd.Series:
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
        """
        Calculates market cap dominance and adds it to the dataframe.
        """
        self._load_config_files()
        if not self._portfolio_config or not self._global_status or not self.dp:
            print(
                "Missing config, status, or DataProvider. Skipping dominance calculation."
            )
            return dataframe

        target_ticker = metadata["pair"].split("/")[0]
        all_assets = self._portfolio_config.get("portfolio_assets", [])
        stablecoin_cfg = next((a for a in all_assets if a.get("level") == 0), None)
        if not stablecoin_cfg:
            return dataframe
        stablecoin_ticker = stablecoin_cfg["ticker"]

        # --- Step 1: Estimate historical market cap for each asset ---
        all_historical_mc_df = {}
        tradable_assets = [
            a for a in all_assets if a.get("level") != 0 and a.get("exchange") != "N/A"
        ]

        for asset in tradable_assets:
            ticker = asset["ticker"]
            pair_symbol = f"{ticker}/{stablecoin_ticker}"

            # Use data provider to get ohlcv for each asset
            ohlcv_df = self.dp.get_pair_dataframe(
                pair=pair_symbol, timeframe=self.timeframe
            )
            asset_status = self._global_status.get(ticker, {})
            last_mc = asset_status.get("last_market_cap")

            if ohlcv_df.empty or last_mc is None or last_mc <= 0:
                continue

            # In backtesting, "current price" is the last known price in the series
            current_price = ohlcv_df["close"].iloc[-1]
            if current_price <= 0:
                continue

            mc_df = pd.DataFrame(index=ohlcv_df.index)
            mc_df["open"] = (last_mc / current_price) * ohlcv_df["open"]
            mc_df["high"] = (last_mc / current_price) * ohlcv_df["high"]
            mc_df["low"] = (last_mc / current_price) * ohlcv_df["low"]
            mc_df["close"] = (last_mc / current_price) * ohlcv_df["close"]
            all_historical_mc_df[ticker] = mc_df

        if not all_historical_mc_df:
            return dataframe

        # --- Step 2: Calculate total historical market cap ---
        combined_mc_df = pd.concat(
            all_historical_mc_df.values(), axis=1, keys=all_historical_mc_df.keys()
        )
        total_mc_df = pd.DataFrame(index=combined_mc_df.index)
        total_mc_df["open"] = combined_mc_df.loc[:, (slice(None), "open")].sum(axis=1)
        total_mc_df["high"] = combined_mc_df.loc[:, (slice(None), "high")].sum(axis=1)
        total_mc_df["low"] = combined_mc_df.loc[:, (slice(None), "low")].sum(axis=1)
        total_mc_df["close"] = combined_mc_df.loc[:, (slice(None), "close")].sum(axis=1)
        total_mc_df.replace(0, np.nan, inplace=True)  # Avoid division by zero

        # --- Step 3: Calculate dominance for the target asset ---
        target_asset_mc_df = all_historical_mc_df.get(target_ticker)
        if target_asset_mc_df is None:
            return dataframe

        dominance_ohlcv_df = pd.DataFrame(index=target_asset_mc_df.index)
        dominance_ohlcv_df["open"] = (
            target_asset_mc_df["open"] / total_mc_df["open"]
        ) * 100
        dominance_ohlcv_df["high"] = (
            target_asset_mc_df["high"] / total_mc_df["high"]
        ) * 100
        dominance_ohlcv_df["low"] = (
            target_asset_mc_df["low"] / total_mc_df["low"]
        ) * 100
        dominance_ohlcv_df["close"] = (
            target_asset_mc_df["close"] / total_mc_df["close"]
        ) * 100

        # Add the results to the main dataframe for use in trend population methods
        dataframe["dom_open"] = dominance_ohlcv_df["open"]
        dataframe["dom_high"] = dominance_ohlcv_df["high"]
        dataframe["dom_low"] = dominance_ohlcv_df["low"]
        dataframe["dom_close"] = dominance_ohlcv_df["close"]
        # Use price volume for indicator compatibility
        dataframe["dom_volume"] = dataframe["volume"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Check if dominance calculation was successful
        if (
            "dom_close" not in dataframe.columns
            or dataframe["dom_close"].isnull().all()
        ):
            return dataframe

        # Create a temporary dataframe with dominance data for the indicator
        dominance_df = dataframe[
            ["dom_open", "dom_high", "dom_low", "dom_close", "dom_volume"]
        ].copy()
        dominance_df.rename(
            columns={
                "dom_open": "open",
                "dom_high": "high",
                "dom_low": "low",
                "dom_close": "close",
                "dom_volume": "volume",
            },
            inplace=True,
        )

        conditions = []
        kc = self.keltner_channel(
            dominance_df,  # Use dominance data
            window=self.kc_window.value,
            mult=self.kc_mult.value,
            atrs=self.kc_atrs.value,
        )
        # Signal is based on the dominance close crossing the dominance-based Keltner Channel
        conditions.append(self.crossed_above(dataframe["dom_close"], kc["upper"]))
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Check if dominance calculation was successful
        if (
            "dom_close" not in dataframe.columns
            or dataframe["dom_close"].isnull().all()
        ):
            return dataframe

        # Create a temporary dataframe with dominance data for the indicator
        dominance_df = dataframe[
            ["dom_open", "dom_high", "dom_low", "dom_close", "dom_volume"]
        ].copy()
        dominance_df.rename(
            columns={
                "dom_open": "open",
                "dom_high": "high",
                "dom_low": "low",
                "dom_close": "close",
                "dom_volume": "volume",
            },
            inplace=True,
        )

        conditions = []
        kc = self.keltner_channel(
            dominance_df,  # Use dominance data
            window=self.kc_window.value,
            mult=self.kc_mult.value,
            atrs=self.kc_atrs.value,
        )
        # Signal is based on the dominance close crossing the dominance-based Keltner Channel
        conditions.append(self.crossed_below(dataframe["dom_close"], kc["lower"]))
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1

        return dataframe
