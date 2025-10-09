### trading_bot.py
import asyncio
import datetime
import decimal
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from decimal import Decimal, InvalidOperation
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
from telegram import Bot

# --- Global Config Loading ---
DEFAULT_CONFIG_FILE_PATH = "./portfolio/configs/portfolio_config.json"


def _validate_required_sections(config_data, config_path):
    required_sections = [
        "bot_settings",
        "paths",
        "logging",
        "exchange_configurations",
        "exchanges_credentials",
        "telegram",
        "telegram_credentials",
        "portfolio_assets",
    ]
    for section in required_sections:
        if section not in config_data:
            print(
                f"ERROR: Config file {config_path} is missing the '{section}' section. "
                "Please ensure it exists. Exiting."
            )
            sys.exit(1)


def _validate_exchange_configurations(config_data):
    exchange_configs = config_data.get("exchange_configurations", {})
    all_credentials = config_data.get("exchanges_credentials", {})
    if not isinstance(exchange_configs, dict):
        print("ERROR: 'exchange_configurations' must be a dictionary. Exiting.")
        sys.exit(1)
    for ex_id, ex_conf in exchange_configs.items():
        if "credentials_key" not in ex_conf:
            print(
                f"ERROR: 'credentials_key' missing in 'exchange_configurations' for "
                f"'{ex_id}'. Exiting."
            )
            sys.exit(1)
        if ex_conf["credentials_key"] not in all_credentials:
            print(
                f"ERROR: Credentials key '{ex_conf['credentials_key']}' for exchange '{ex_id}' "
                "not found in 'exchanges_credentials'. Exiting."
            )
            sys.exit(1)


def _validate_paths(config_data):
    paths_cfg = config_data.get("paths", {})
    if (
        "portfolio_strategies_dir" not in paths_cfg
        or "global_status_file_path" not in paths_cfg
    ):
        print(
            "ERROR: Config file must contain 'portfolio_strategies_dir' and 'global_status_file_path' in the 'paths' section. Exiting."
        )
        sys.exit(1)


def _validate_portfolio_assets(config_data):
    portfolio_assets = config_data.get("portfolio_assets", [])
    if not isinstance(portfolio_assets, list) or not portfolio_assets:
        print(
            "ERROR: Config file must contain a non-empty 'portfolio_assets' list. Exiting."
        )
        sys.exit(1)
    for asset_cfg in portfolio_assets:
        required_keys = ["ticker", "level", "strategy_file", "strategy_name"]
        if not all(key in asset_cfg for key in required_keys):
            print(
                f"ERROR: Each item in 'portfolio_assets' must contain 'ticker', 'level', 'strategy_file', and 'strategy_name'. Problem with: {asset_cfg}. Exiting."
            )
            sys.exit(1)
        if asset_cfg["level"] != 0 and "exchange" not in asset_cfg:
            print(
                f"ERROR: Tradable asset (level != 0) '{asset_cfg.get('ticker')}' must have an 'exchange' key. Exiting."
            )
            sys.exit(1)


def load_app_config(config_path):
    try:
        with Path(config_path).open() as f:
            config_data = json.load(f)
        _validate_required_sections(config_data, config_path)
        _validate_exchange_configurations(config_data)
        _validate_paths(config_data)
        _validate_portfolio_assets(config_data)
        return config_data
    except FileNotFoundError:
        print(
            f"ERROR: Configuration file not found at {config_path}. "
            "Please create it based on the example. Exiting."
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"ERROR: Could not parse configuration file {config_path}: {e}. "
            "Check for JSON syntax errors. Exiting."
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"ERROR: Unexpected error loading configuration from {config_path}: {e}. Exiting."
        )
        sys.exit(1)


# Load configuration globally at the start.
APP_CONFIG = load_app_config(DEFAULT_CONFIG_FILE_PATH)

# --- Constants from Config (used by decorator and potentially other global scope items) ---
BOT_SETTINGS_CFG = APP_CONFIG.get("bot_settings", {})
MAX_RETRIES_CFG = BOT_SETTINGS_CFG.get("max_retries_network", 3)
RETRY_DELAY_SECONDS_CFG = BOT_SETTINGS_CFG.get("retry_delay_seconds_network", 5)


# Define which columns belong to status files for saving
STATUS_COLUMNS = [
    "price",
    "dominance",
    "last_market_cap",
    "last_market_cap_fetch_time",
    "position_last_check_bucket",
    "trend_last_check_bucket",
    "position_signal_prev_1",
    "position_signal_prev_2",
    "trend_signal_prev_1",
    "trend_signal_prev_2",
    "free",
    "used",
    "value_in_stable",
    "ratio",
    "target_value",
    "change",
    "done",
]


# --- Helper: Retry Decorator ---
def retry_on_network_error(
    _func=None, *, max_retries=MAX_RETRIES_CFG, delay_seconds=RETRY_DELAY_SECONDS_CFG
):
    def decorator(func):
        async def wrapper_async(self, *args, **kwargs):
            return await _retry_logic_async(
                self, func, args, kwargs, max_retries, delay_seconds
            )

        def wrapper_sync(self, *args, **kwargs):
            return _retry_logic_sync(
                self, func, args, kwargs, max_retries, delay_seconds
            )

        if asyncio.iscoroutinefunction(func):
            return wrapper_async
        else:
            return wrapper_sync

    if _func is None:
        return decorator
    else:
        if asyncio.iscoroutinefunction(_func):
            return decorator(_func)
        else:
            return decorator(_func)


async def _retry_logic_async(self, func, args, kwargs, max_retries, delay_seconds):
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await func(self, *args, **kwargs)
        except (
            ccxt.NetworkError,
            ccxt.RequestTimeout,
            ccxt.ExchangeNotAvailable,
            ccxt.OnMaintenance,
        ) as e:
            self.logger.warning(
                f"Network error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay_seconds}s..."
            )
            last_exception = e
            await asyncio.sleep(delay_seconds)
        except ConnectionError as e:
            self.logger.warning(
                f"Connection error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay_seconds}s..."
            )
            last_exception = e
            await asyncio.sleep(delay_seconds)
    return _handle_retry_failure(self, func, args, last_exception)


def _retry_logic_sync(self, func, args, kwargs, max_retries, delay_seconds):
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(self, *args, **kwargs)
        except (
            ccxt.NetworkError,
            ccxt.RequestTimeout,
            ccxt.ExchangeNotAvailable,
            ccxt.OnMaintenance,
        ) as e:
            self.logger.warning(
                f"Network error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay_seconds}s..."
            )
            last_exception = e
            time.sleep(delay_seconds)
        except ConnectionError as e:
            self.logger.warning(
                f"Connection error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay_seconds}s..."
            )
            last_exception = e
            time.sleep(delay_seconds)
    return _handle_retry_failure(self, func, args, last_exception)


def _handle_retry_failure(self, func, args, last_exception):
    self.logger.error(f"{func.__name__} failed after retries: {last_exception}")
    if "fetch_ohlcv" in func.__name__:
        return pd.DataFrame()
    if "fetch_market_cap" in func.__name__:
        if len(args) > 3 and isinstance(args[3], (dict, pd.Series)):
            return args[3].get("last_market_cap"), args[3].get(
                "last_market_cap_fetch_time"
            )
        else:
            self.logger.warning(
                f"Could not determine fallback for fetch_market_cap due to unexpected args: {args}"
            )
            return None, None
    raise last_exception


# --- IndicatorMixin and Keltner Channel (Unchanged) ---
class IndicatorMixin:
    def _check_fillna(self, serie, value=0):
        if hasattr(self, "_fillna") and self._fillna:
            serie = serie.fillna(value)
        return serie

    def _true_range(self, high, low, close):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

    def _crossed_above(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        series1_shifted = series1.shift(1)
        series2_shifted = series2.shift(1)
        above = (series1 > series2) & (series1_shifted <= series2_shifted)
        return above.fillna(False)

    def _crossed_below(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        series1_shifted = series1.shift(1)
        series2_shifted = series2.shift(1)
        below = (series1 < series2) & (series1_shifted >= series2_shifted)
        return below.fillna(False)


class AverageTrueRange(IndicatorMixin):
    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        true_range = self._true_range(self._high, self._low, self._close)
        # MODIFICATION: Removed 'min_periods' to match Freqtrade's calculation method.
        atr = true_range.ewm(alpha=1 / self._window, adjust=False).mean()
        self._atr = pd.Series(data=atr, index=true_range.index)

    def average_true_range(self) -> pd.Series:
        atr = self._check_fillna(self._atr, value=0)
        return pd.Series(atr, name="atr")


class KeltnerChannel(IndicatorMixin):
    def __init__(
        self,
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        window_atr: int = 10,
        window_mult: float = 2,
        fillna: bool = False,
    ):
        self._open = open
        self._high = high
        self._low = low
        self._close = close
        self._window = int(window)
        self._window_atr = int(window_atr)
        self._window_mult = window_mult
        self._fillna = fillna
        self._run()

    def _run(self):
        # Calculate typical price as (open + high + low + close) / 4.0
        typical_price = (self._open + self._high + self._low + self._close) / 4.0
        # Calculate middle band (tp) using EMA of typical price
        # MODIFICATION: Removed 'min_periods' to match Freqtrade's calculation method.
        self._tp = typical_price.ewm(span=self._window, adjust=False).mean()

        atr_indicator = AverageTrueRange(
            high=self._high,
            low=self._low,
            close=self._close,
            window=self._window_atr,
            fillna=self._fillna,
        )
        atr = atr_indicator.average_true_range()
        self._tp_high = self._tp + (self._window_mult * atr)
        self._tp_low = self._tp - (self._window_mult * atr)

    def keltner_channel_mband(self) -> pd.Series:
        return pd.Series(self._check_fillna(self._tp, value=-1), name="kc_mband")

    def keltner_channel_hband(self) -> pd.Series:
        return pd.Series(self._check_fillna(self._tp_high, value=-1), name="kc_hband")

    def keltner_channel_lband(self) -> pd.Series:
        return pd.Series(self._check_fillna(self._tp_low, value=-1), name="kc_lband")

    def keltner_channel_wband(self) -> pd.Series:
        wband = ((self._tp_high - self._tp_low) / self._tp.replace(0, np.nan)) * 100
        return pd.Series(self._check_fillna(wband, value=0), name="kc_wband")

    def keltner_channel_pband(self) -> pd.Series:
        denominator = self._tp_high - self._tp_low
        pband = (self._close - self._tp_low) / denominator.where(
            denominator != 0, np.nan
        )
        return pd.Series(self._check_fillna(pband, value=0.5), name="kc_pband")

    def keltner_channel_close_hband_indicator(self) -> pd.Series:
        hband_indicator = self._crossed_above(self._close, self._tp_high)
        return pd.Series(
            self._check_fillna(hband_indicator, value=False),
            name="kc_close_hband_indicator",
        )

    def keltner_channel_high_hband_indicator(self) -> pd.Series:
        hband_indicator = self._crossed_above(self._high, self._tp_high)
        return pd.Series(
            self._check_fillna(hband_indicator, value=False),
            name="kc_high_hband_indicator",
        )

    def keltner_channel_close_lband_indicator(self) -> pd.Series:
        lband_indicator = self._crossed_below(self._close, self._tp_low)
        return pd.Series(
            self._check_fillna(lband_indicator, value=False),
            name="kc_close_lband_indicator",
        )

    def keltner_channel_low_lband_indicator(self) -> pd.Series:
        lband_indicator = self._crossed_below(self._low, self._tp_low)
        return pd.Series(
            self._check_fillna(lband_indicator, value=False),
            name="kc_low_lband_indicator",
        )


# Modular Indicator Strategy Framework ---


class IndicatorStrategy(ABC):
    """
    Abstract base class for all indicator strategies.
    This defines the contract that any new indicator must follow.
    """

    @classmethod
    @abstractmethod
    def get_strategy_params(cls) -> list[str]:
        """
        Returns a list of parameter names (column names) used by this strategy.
        This includes configuration (e.g., 'position_kc_window') and stop-loss params.
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_config(cls, is_stablecoin: bool) -> dict:
        """
        Returns a dictionary with the default configuration for this strategy.
        """
        pass

    @classmethod
    @abstractmethod
    def get_param_data_types(cls) -> dict[str, str]:
        """
        Returns a dictionary mapping parameter names to their data types ('int', 'float', 'bool').
        """
        pass

    @classmethod
    @abstractmethod
    def prepare_signal_params(cls, row_data: pd.Series, param_prefix: str) -> dict:
        """
        Extracts and prepares the parameters needed for signal calculation from a portfolio row.

        Args:
            row_data (pd.Series): A single row from the portfolio DataFrame.
            param_prefix (str): The prefix for the parameters (e.g., 'position', 'trend').

        Returns:
            dict: A dictionary of parameters ready for `calculate_signals`.
        """
        pass

    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        Calculates the trading signals based on the indicator logic.
        """
        pass


class KeltnerChannelStrategy(IndicatorStrategy):
    """
    Implements the Keltner Channel strategy, encapsulating its logic,
    parameters, defaults, and data types.
    """

    @classmethod
    def get_strategy_params(cls) -> list[str]:
        """Returns the list of parameters specific to the Keltner Channel strategy."""
        return [
            "position_kc_window",
            "position_kc_atr",
            "position_kc_mult",
            "trend_kc_window",
            "trend_kc_atr",
            "trend_kc_mult",
        ]

    @classmethod
    def get_default_config(cls, is_stablecoin: bool) -> dict:
        """Provides default Keltner Channel parameters."""
        if is_stablecoin:
            return {
                "position_kc_window": 0,
                "position_kc_atr": 0,
                "position_kc_mult": 0.0,
                "trend_kc_window": 0,
                "trend_kc_atr": 0,
                "trend_kc_mult": 0.0,
            }
        return {
            "position_kc_window": 20,
            "position_kc_atr": 10,
            "position_kc_mult": 2.0,
            "trend_kc_window": 20,
            "trend_kc_atr": 10,
            "trend_kc_mult": 2.0,
        }

    @classmethod
    def get_param_data_types(cls) -> dict[str, str]:
        """Returns the data types for all Keltner Channel parameters."""
        return {
            "position_kc_window": "int",
            "position_kc_atr": "int",
            "position_kc_mult": "float",
            "trend_kc_window": "int",
            "trend_kc_atr": "int",
            "trend_kc_mult": "float",
        }

    @classmethod
    def prepare_signal_params(cls, row_data: pd.Series, param_prefix: str) -> dict:
        """Prepares Keltner Channel parameters from the portfolio row."""
        # The keys in this returned dict ('window', 'window_atr', etc.) are what the
        # `calculate_signals` method expects.
        return {
            "window": row_data[f"{param_prefix}_kc_window"],
            "window_atr": row_data[f"{param_prefix}_kc_atr"],
            "window_mult": row_data[f"{param_prefix}_kc_mult"],
        }

    def calculate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """Calculates signals using the Keltner Channel indicator."""
        # Extract parameters
        window = params.get("window")
        window_atr = params.get("window_atr")
        window_mult = params.get("window_mult")

        # Validate parameters
        if window is None or window < 1:
            raise ValueError("Invalid 'window': must be >= 1.")
        if window_atr is None or window_atr < 1:
            raise ValueError("Invalid 'window_atr': must be >= 1.")
        if window_mult is None or window_mult <= 0:
            raise ValueError("Invalid 'window_mult': must be > 0.")

        # Calculate signals
        kc = KeltnerChannel(
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=int(window),
            window_mult=float(window_mult),
            window_atr=int(window_atr),
            fillna=True,
        )
        df["buy_crossupper"] = kc.keltner_channel_close_hband_indicator()
        df["sell_crosslower"] = kc.keltner_channel_close_lband_indicator()

        # Combine signals
        df["position"] = 0.0
        df.loc[df["buy_crossupper"], "position"] = 1.0
        df.loc[df["sell_crosslower"], "position"] = -1.0
        df["position"] = df["position"].replace(0, np.nan).ffill().fillna(0)
        return df["position"]


class IndicatorFactory:
    """
    A factory for creating indicator strategy objects. This makes it easy to
    add new strategies without changing the core bot logic.

    To add a new strategy (e.g., 'RSI'):
    1. Create a new class `RSIStrategy(IndicatorStrategy)`.
    2. Implement the required abstract methods in `RSIStrategy`.
    3. Register it in the `_strategies` dictionary below:
       "rsi": RSIStrategy
    """

    _strategies = {
        "keltner_channel": KeltnerChannelStrategy,
        # --- Register new strategies here ---
    }

    @staticmethod
    def get_strategy_instance(name: str) -> IndicatorStrategy:
        """
        Returns an instance of the requested indicator strategy.
        The name is case-insensitive.
        """
        strategy_class = IndicatorFactory._strategies.get(name.lower())
        if not strategy_class:
            raise ValueError(f"Indicator strategy '{name}' not recognized.")
        return strategy_class()

    @staticmethod
    def get_strategy_class(name: str) -> type[IndicatorStrategy]:
        """
        Returns the class type of the requested indicator strategy.
        The name is case-insensitive.
        """
        strategy_class = IndicatorFactory._strategies.get(name.lower())
        if not strategy_class:
            raise ValueError(f"Indicator strategy '{name}' not recognized.")
        return strategy_class


# --- Helper Functions (Unchanged) ---
def standardized_amount(
    amount_float: float, precision_step: float, min_amount_limit: float
) -> decimal.Decimal:
    min_amount_limit = min_amount_limit or 0.0
    if amount_float < min_amount_limit:
        pass
    precision_step_str = format(precision_step, ".18f").rstrip("0").rstrip(".")
    if not precision_step_str or precision_step_str == "0":
        standardized_val = decimal.Decimal(str(amount_float))
    else:
        quantizer = decimal.Decimal(precision_step_str)
        if quantizer == decimal.Decimal(0):
            standardized_val = decimal.Decimal(str(amount_float))
        else:
            standardized_val = (
                decimal.Decimal(str(amount_float)) // quantizer
            ) * quantizer
    if standardized_val < decimal.Decimal(str(min_amount_limit)):
        return decimal.Decimal("0")
    return standardized_val


async def send_info(path: str, bot_token: str, chat_id: str, logger: logging.Logger):
    try:
        bot = Bot(token=bot_token)
        await bot.initialize()
        if Path(path).exists():
            with Path(path).open("rb") as doc_file:
                await bot.send_document(chat_id=chat_id, document=doc_file)
            logger.info(f"Sent {path} to Telegram.")
        else:
            logger.error(f"File not found for Telegram send: {path}")
    except Exception as e:
        logger.error(f"Error sending to Telegram: {e}", exc_info=True)


# --- Unified TradingBot Class ---
class TradingBot:
    def __init__(
        self,
        app_config: dict,
        log_file_path: str,
        logger: logging.Logger,
        exchange_clients: dict[str, ccxt.Exchange] = None,
    ):
        self._initialize_basic_attributes(
            app_config, log_file_path, logger, exchange_clients
        )
        self._initialize_paths()
        self._initialize_telegram()
        self._initialize_coingecko()
        self._log_initialization_mode()
        self._run_initialization_sequence()

    def _initialize_basic_attributes(
        self,
        app_config: dict,
        log_file_path: str,
        logger: logging.Logger,
        exchange_clients: dict[str, ccxt.Exchange],
    ):
        self.app_config = app_config
        self.log_file_path = log_file_path
        self.logger = logger
        self.exchange_clients = exchange_clients if exchange_clients else {}
        self.markets_by_exchange = {}
        self.primary_stable_coin_ticker = None
        self.portfolio_df = pd.DataFrame()
        self.stablecoin_balance_by_exchange = {}
        self.cycle_ohlcv_cache = {}
        self.last_total_stablecoin_balance = None

        self.bot_settings = self.app_config.get("bot_settings", {})
        self.dry_run = self.bot_settings.get("dry_run", True)
        self.indicator_history_size = self.bot_settings.get(
            "indicator_history_size", 1000
        )
        self.max_indicator_history_size = self.bot_settings.get(
            "max_indicator_history_size", 5000
        )
        self.min_order_value_usd = self.bot_settings.get("min_order_value_usd", 15.0)
        self.market_cap_stale_threshold_seconds = self.bot_settings.get(
            "market_cap_stale_threshold_seconds", 24 * 60 * 60
        )
        self.balance_adjustment_factor_buy = self.bot_settings.get(
            "balance_adjustment_factor_buy", 0.99
        )
        self.balance_adjustment_factor_sell = self.bot_settings.get(
            "balance_adjustment_factor_sell", 0.9999
        )
        self.initial_stablecoin_balance = self.bot_settings.get(
            "initial_stablecoin_balance", 1000.0
        )
        self.max_alt_coin_ratio = self.bot_settings.get("max_alt_coin_ratio", 1.0)
        self.rebalance_threshold_percentage = self.bot_settings.get(
            "rebalance_threshold_percentage", 0.01
        )
        self.signal_confirmation_cycles = self.bot_settings.get(
            "signal_confirmation_cycles", 1
        )

    def _initialize_paths(self):
        paths_cfg = self.app_config.get("paths", {})
        self.portfolio_strategies_dir = paths_cfg.get("portfolio_strategies_dir")
        self.global_status_file_path = paths_cfg.get("global_status_file_path")
        self.portfolio_export_path = paths_cfg.get("portfolio_export_path")
        self.ohlcv_export_dir = paths_cfg.get("ohlcv_export_dir")

        if not self.portfolio_strategies_dir or not self.global_status_file_path:
            self.logger.critical(
                "Portfolio strategies directory or global status file path not configured. Exiting."
            )
            sys.exit(1)

        from pathlib import Path

        Path(self.portfolio_strategies_dir).mkdir(parents=True, exist_ok=True)
        global_status_dir = Path(self.global_status_file_path).parent
        if global_status_dir:
            global_status_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_telegram(self):
        self.telegram_bot_token = None
        self.telegram_chat_id = None
        tg_main_cfg = self.app_config.get("telegram", {})
        if tg_main_cfg.get("enabled", False):
            tg_creds_key = tg_main_cfg.get("credentials_key")
            if tg_creds_key:
                creds = self.app_config.get("telegram_credentials", {}).get(
                    tg_creds_key
                )
                if creds and creds.get("apiKey") and creds.get("chatId"):
                    self.telegram_bot_token = creds["apiKey"]
                    self.telegram_chat_id = creds["chatId"]
                    self.logger.info(
                        f"Telegram notifications configured using credentials key: '{tg_creds_key}'."
                    )
                else:
                    self.logger.warning(
                        f"Telegram credentials for key '{tg_creds_key}' are incomplete or missing. Notifications disabled."
                    )
            else:
                self.logger.warning(
                    "Telegram is enabled, but 'credentials_key' not specified. Notifications disabled."
                )
        else:
            self.logger.info("Telegram notifications are disabled.")

    def _initialize_coingecko(self):
        try:
            self.cg = CoinGeckoAPI()
            self.logger.info("CoinGeckoAPI client initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize CoinGeckoAPI: {e}", exc_info=True)
            self.cg = None

    def _log_initialization_mode(self):
        if self.dry_run:
            self.logger.warning("=" * 50)
            self.logger.warning("BOT IS RUNNING IN DRY RUN MODE.")
            self.logger.warning("NO REAL TRADES WILL BE EXECUTED.")
            self.logger.warning("=" * 50)
        else:
            self.logger.warning("=" * 50)
            self.logger.warning("BOT IS RUNNING IN LIVE TRADING MODE.")
            self.logger.warning("REAL TRADES WILL BE EXECUTED. BE CAREFUL!")
            self.logger.warning("=" * 50)

    def _run_initialization_sequence(self):
        self.logger.info(
            f"{datetime.datetime.now(datetime.timezone.utc)}: Initializing TradingBot..."
        )

        self._load_portfolio_data()

        if self.portfolio_df.empty:
            self.logger.warning("Portfolio is empty. Initialization complete.")
            return

        self._identify_primary_stablecoin()
        self._perform_initial_signal_checks()
        self._load_exchange_markets()
        self._perform_initial_portfolio_distribution()
        self._save_portfolio_status()
        self.logger.info("TradingBot initialization complete. Portfolio status saved.")

    def _identify_primary_stablecoin(self):
        stable_coin_series_init = self.portfolio_df.loc[
            self.portfolio_df["level"] == 0, "ticker"
        ]
        if not stable_coin_series_init.empty:
            self.primary_stable_coin_ticker = stable_coin_series_init.iloc[0]
        else:
            self.logger.warning(
                "No stablecoin (Level 0) defined. Some functions might be impaired."
            )

    def _perform_initial_signal_checks(self):
        self.logger.info("Performing initial signal checks for all active assets...")
        for idx in self.portfolio_df.loc[self.portfolio_df["level"] != 0].index:
            asset_exchange_id = self.portfolio_df.loc[idx, "exchange"]
            if asset_exchange_id != "N/A" and self.get_client(asset_exchange_id):
                self.portfolio_df = self._check_position(
                    user_dataframe=self.portfolio_df, row_idx=idx
                )
                self.portfolio_df = self._check_trend(
                    user_dataframe=self.portfolio_df, row_idx=idx
                )
            elif asset_exchange_id != "N/A":
                self.logger.warning(
                    f"No client for exchange '{asset_exchange_id}' for asset {self.portfolio_df.loc[idx, 'ticker']}. Skipping initial signal check."
                )
        self.logger.info("Initial signal checks complete.")

    def _perform_initial_portfolio_distribution(self):
        self.logger.info("Performing initial portfolio distribution...")
        if self.markets_by_exchange and self.primary_stable_coin_ticker:
            self._distribute_portfolio()
            self.logger.info(
                "Setting 'Done' flag to True for all non-stablecoin assets after initial distribution."
            )
            self.portfolio_df.loc[
                (self.portfolio_df["level"] != 0)
                & (self.portfolio_df["exchange"] != "N/A"),
                "done",
            ] = True
        else:
            self.logger.warning(
                "Skipping initial distribution due to missing exchange markets or stablecoin definition."
            )

    # Helper Methods for Multi-Exchange ---
    def get_client(self, exchange_id: str) -> ccxt.Exchange | None:
        """Retrieves the CCXT client for a given exchange ID."""
        client = self.exchange_clients.get(exchange_id)
        if not client:
            # This log might be too frequent if an exchange is intentionally not configured.
            # Consider logging only if an operation *requires* it and it's missing.
            # self.logger.error(f"No CCXT client available for exchange_id: {exchange_id}")
            pass
        return client

    def get_markets_for_exchange(self, exchange_id: str) -> dict | None:
        """Retrieves cached market data for a given exchange ID."""
        markets = self.markets_by_exchange.get(exchange_id)
        if not markets:
            self.logger.warning(
                f"Market data not loaded or available for exchange_id: {exchange_id}"
            )
        return markets

    def _load_exchange_markets(self):
        """Loads and caches market data for all configured exchange clients."""
        self.logger.info("Loading exchange markets...")
        self.markets_by_exchange = {}
        if not self.exchange_clients:
            self.logger.warning(
                "No exchange clients initialized. Cannot load market data."
            )
            return

        for exchange_id, client_instance in self.exchange_clients.items():
            try:
                markets_data_list = client_instance.fetch_markets()
                self.markets_by_exchange[exchange_id] = {
                    m["symbol"]: m for m in markets_data_list
                }
                self.logger.debug(
                    f"Fetched {len(self.markets_by_exchange[exchange_id])} markets for {exchange_id}."
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to fetch markets for {exchange_id}: {e}. This exchange might be unavailable for trading."
                )

        if not self.markets_by_exchange:
            self.logger.error(
                "No exchange clients successfully loaded markets. Most operations will be skipped."
            )

    def _load_json_file(self, file_path, is_global_status=False):
        try:
            with Path(file_path).open() as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.info(f"File {file_path} not found.")
            if is_global_status:
                self.logger.info("Creating empty global status data.")
                data = {}
                return data
            return None
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Error decoding JSON from {file_path}: {e}. Using empty/default data."
            )
            return {} if is_global_status else None
        except Exception as e:
            self.logger.error(
                f"Unexpected error loading {file_path}: {e}", exc_info=True
            )
            return {} if is_global_status else None

    def _save_json_file(self, file_path, data):
        try:
            # The log shows an IsADirectoryError, which means the path exists and is a directory.
            # This check prevents the bot from crashing by handling the misconfiguration gracefully.
            if Path(file_path).is_dir():
                self.logger.error(
                    f"Error saving data: The provided path '{file_path}' is a directory, not a file. "
                    "Please update your configuration to point to a specific file (e.g., './portfolio/status/status.json').",
                    exc_info=False,
                )
                return  # Stop execution for this function to prevent the crash.

            dir_name = Path(file_path).parent
            if dir_name:
                dir_name.mkdir(parents=True, exist_ok=True)

            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif pd.isna(obj):
                    return None
                return obj

            with Path(file_path).open("w") as f:
                json.dump(data, f, indent=4, default=convert_numpy_types)
            self.logger.debug(f"Saved data to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}", exc_info=True)

    def _get_default_status_data(self, ticker: str, level: int):
        status = {
            "price": None,
            "dominance": None,
            "last_market_cap": None,
            "last_market_cap_fetch_time": None,
            "position_last_check_bucket": 0,
            "trend_last_check_bucket": 0,
            "position_signal_prev_1": 0.0,
            "position_signal_prev_2": 0.0,
            "trend_signal_prev_1": 0.0,
            "trend_signal_prev_2": 0.0,
            "free": 0.0,
            "used": 0.0,
            "value_in_stable": 0.0,
            "ratio": 0.0,
            "target_value": 0.0,
            "change": 0.0,
            "done": True,
        }
        if level == 0:
            status["price"] = 1.0
            status["free"] = self.initial_stablecoin_balance
            status["value_in_stable"] = self.initial_stablecoin_balance
            status["ratio"] = 1.0
            status["target_value"] = self.initial_stablecoin_balance
            status["done"] = True
        return status

    # Define core column groups for dynamic DataFrame creation
    CORE_STATIC_COLUMNS = [
        "ticker",
        "level",
        "exchange",
        "coingecko_id",
        "position_interval",
        "trend_interval",
        "strategy_name",
        "strategy_file",
    ]
    CORE_STATUS_COLUMNS = STATUS_COLUMNS  # STATUS_COLUMNS is defined globally

    # Define data types for the bot's own core columns
    CORE_COLUMN_TYPES = {
        # Static
        "level": "int",
        "position_interval": "int",
        "trend_interval": "int",
        # Status
        "price": "float",
        "dominance": "float",
        "last_market_cap": "float",
        "last_market_cap_fetch_time": "float",
        "position_last_check_bucket": "int",
        "trend_last_check_bucket": "int",
        "position_signal_prev_1": "float",
        "position_signal_prev_2": "float",
        "trend_signal_prev_1": "float",
        "trend_signal_prev_2": "float",
        "free": "float",
        "used": "float",
        "value_in_stable": "float",
        "ratio": "float",
        "target_value": "float",
        "change": "float",
        "done": "bool",
    }

    def _load_portfolio_data(self):
        self.logger.info("Loading portfolio data...")
        asset_configs = self.app_config.get("portfolio_assets", [])
        if not asset_configs:
            self.logger.warning(
                "No assets defined in 'portfolio_assets'. Portfolio will be empty."
            )
            self.portfolio_df = pd.DataFrame()
            return

        self._initialize_portfolio_metadata(asset_configs)
        all_asset_data, global_status_data_changed = self._load_asset_data(
            asset_configs
        )

        if global_status_data_changed:
            self._save_json_file(self.global_status_file_path, self.global_status_data)

        if not all_asset_data:
            self.logger.warning(
                "Portfolio data is empty after attempting to load assets."
            )
            self.portfolio_df = pd.DataFrame()
            return

        self._create_portfolio_dataframe(all_asset_data)

    def _initialize_portfolio_metadata(self, asset_configs):
        self.global_status_data = (
            self._load_json_file(self.global_status_file_path, is_global_status=True)
            or {}
        )
        self.primary_stable_coin_ticker = next(
            (ac.get("ticker") for ac in asset_configs if ac.get("level") == 0), None
        )
        if not self.primary_stable_coin_ticker:
            self.logger.warning(
                "No primary stablecoin (Level 0) found. Some defaults might not be optimal."
            )

    def _load_asset_data(self, asset_configs):
        all_asset_data = []
        global_status_data_changed = False
        for asset_config in asset_configs:
            ticker = asset_config.get("ticker")
            if not ticker:
                continue

            static_data, strategy_data, status_data = self._load_individual_asset_data(
                asset_config, ticker
            )
            if status_data is None:
                status_data = self._get_default_status_data(
                    ticker, asset_config.get("level", 0)
                )
                self.global_status_data[ticker] = status_data
                global_status_data_changed = True

            combined_data = {**static_data, **strategy_data, **status_data}
            all_asset_data.append(combined_data)
        return all_asset_data, global_status_data_changed

    def _load_individual_asset_data(self, asset_config, ticker):
        static_data = {k: v for k, v in asset_config.items()}
        strategy_file_path = (
            Path(self.portfolio_strategies_dir) / asset_config["strategy_file"]
        )
        strategy_data = self._load_json_file(strategy_file_path) or {}
        if not strategy_data:
            self.logger.info(
                f"Strategy file for {ticker} not found. Creating with defaults."
            )
            strategy_data = IndicatorFactory.get_strategy_class(
                asset_config["strategy_name"]
            ).get_default_config(asset_config.get("level", 0) == 0)
            self._save_json_file(strategy_file_path, strategy_data)

        status_data = self.global_status_data.get(ticker)
        return static_data, strategy_data, status_data

    def _create_portfolio_dataframe(self, all_asset_data):
        self.portfolio_df = pd.DataFrame(all_asset_data)
        expected_columns = (
            self.CORE_STATIC_COLUMNS
            + self.CORE_STATUS_COLUMNS
            + list(self.CORE_COLUMN_TYPES.keys())
        )
        for col in expected_columns:
            if col not in self.portfolio_df.columns:
                self.portfolio_df[col] = np.nan
                self.logger.warning(
                    f"Added missing column '{col}' to portfolio DataFrame."
                )

        for col, dtype in self.CORE_COLUMN_TYPES.items():
            self._apply_column_dtype(col, dtype)

        self.logger.info(f"Portfolio loaded with {len(self.portfolio_df)} assets.")
        if not self.portfolio_df.empty:
            self.logger.debug(
                f"Portfolio head:\n{self.portfolio_df.head().to_string()}"
            )

    def _apply_column_dtype(self, col, dtype):
        if col in self.portfolio_df.columns:
            try:
                if dtype == "int":
                    self.portfolio_df[col] = (
                        pd.to_numeric(self.portfolio_df[col], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                elif dtype == "float":
                    self.portfolio_df[col] = pd.to_numeric(
                        self.portfolio_df[col], errors="coerce"
                    )
                elif dtype == "bool":
                    self.portfolio_df[col] = self.portfolio_df[col].apply(
                        lambda x: bool(x) if pd.notna(x) else False
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to cast column '{col}' to type '{dtype}': {e}"
                )

    def _save_portfolio_status(self):
        if self.portfolio_df is None or self.portfolio_df.empty:
            self.logger.warning(
                "Portfolio DataFrame is empty. Nothing to save to global status."
            )
            return

        all_statuses_to_save = {}
        for idx, row in self.portfolio_df.iterrows():
            ticker = row["ticker"]
            current_asset_status = {}
            for col in STATUS_COLUMNS:
                if col in row:
                    value = row[col]
                    current_asset_status[col] = value
                else:
                    self.logger.warning(
                        f"Status column '{col}' not found in portfolio_df for {ticker} during save. Setting to None."
                    )
                    current_asset_status[col] = None
            all_statuses_to_save[ticker] = current_asset_status

        self._save_json_file(self.global_status_file_path, all_statuses_to_save)
        self.logger.info(
            f"Global portfolio status saved to {self.global_status_file_path}"
        )

    def _get_ccxt_timeframe(self, interval_seconds: int) -> str:
        timeframes = {
            60: "1m",
            180: "3m",
            300: "5m",
            900: "15m",
            1800: "30m",
            3600: "1h",
            7200: "2h",
            14400: "4h",
            21600: "6h",
            28800: "8h",
            43200: "12h",
            86400: "1d",
            604800: "1w",
        }
        for threshold, timeframe in sorted(timeframes.items()):
            if interval_seconds <= threshold:
                return timeframe
        self.logger.warning(
            f"Interval {interval_seconds}s not cleanly mapped, defaulting to '1d'."
        )
        return "1d"

    @retry_on_network_error
    def _fetch_ohlcv_ccxt(
        self,
        client: ccxt.Exchange,
        symbol_pair: str,
        timeframe: str,
        n_bars: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data from an exchange, handling pagination to retrieve
        a large number of bars (n_bars) if it exceeds the exchange's single-request limit.
        """
        if not client:
            self.logger.error("CCXT client not provided for fetching OHLCV.")
            return pd.DataFrame()

        cache_key = (client.id, symbol_pair, timeframe, n_bars)
        if cache_key in self.cycle_ohlcv_cache:
            self.logger.debug(
                f"CACHE HIT for {symbol_pair} ({timeframe}, {n_bars} bars) on {client.id}."
            )
            return self.cycle_ohlcv_cache[cache_key].copy()

        self.logger.debug(
            f"CACHE MISS for {symbol_pair} ({timeframe}, {n_bars} bars) on {client.id}. Fetching from exchange."
        )

        try:
            limit = client.options.get("fetchOHLCVLimit", 1000)
            all_ohlcv = []

            # Calculate an approximate start time, adding a small buffer for market gaps
            timeframe_duration_ms = client.parse_timeframe(timeframe) * 1000
            since = client.milliseconds() - timeframe_duration_ms * (n_bars + 5)

            while since < client.milliseconds():
                self.logger.debug(
                    f"Fetching chunk for {symbol_pair} since {client.iso8601(since)}"
                )
                ohlcv_chunk = client.fetch_ohlcv(
                    symbol_pair, timeframe, since=since, limit=limit
                )

                if not ohlcv_chunk:
                    # self.logger.info(
                    #     f"No more historical data available for {symbol_pair} after {client.iso8601(since)}."
                    # )
                    break

                # Avoid re-adding the same last candle from the previous chunk
                if all_ohlcv and ohlcv_chunk[0][0] == all_ohlcv[-1][0]:
                    ohlcv_chunk = ohlcv_chunk[1:]

                if not ohlcv_chunk:
                    break

                all_ohlcv.extend(ohlcv_chunk)
                since = (
                    ohlcv_chunk[-1][0] + 1
                )  # Next fetch starts after the last candle

            if not all_ohlcv:
                self.logger.warning(
                    f"No OHLCV data returned for {symbol_pair} ({timeframe}) from {client.id}."
                )
                return pd.DataFrame()

            df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("datetime", inplace=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)

            # Ensure we only return the most recent `n_bars`
            df = df.tail(n_bars)

            if df.empty:
                self.logger.warning(
                    f"OHLCV data for {symbol_pair} ({timeframe}) on {client.id} is empty after processing."
                )
            else:
                self.cycle_ohlcv_cache[cache_key] = df.copy()
                self.logger.debug(
                    f"Successfully fetched and processed {len(df)} bars for {symbol_pair}."
                )
            return df
        except ccxt.BadSymbol as e:
            self.logger.error(
                f"CCXT BadSymbol: {symbol_pair} may not be supported by {client.id}. Error: {e}"
            )
        except ccxt.ExchangeError as e:
            self.logger.error(
                f"CCXT ExchangeError fetching {symbol_pair} ({timeframe}) from {client.id}: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching OHLCV for {symbol_pair} ({timeframe}) from {client.id}: {e}",
                exc_info=True,
            )
        return pd.DataFrame()

    @retry_on_network_error  # This method is for CoinGecko, independent of exchange client
    def _fetch_market_cap(
        self, coin_gecko_id: str, ticker: str, current_row_data: pd.Series
    ) -> tuple[float | None, float | None]:
        if not self.cg:
            self.logger.error("CoinGecko client not available.")
            return current_row_data.get("last_market_cap"), current_row_data.get(
                "last_market_cap_fetch_time"
            )
        if not coin_gecko_id or pd.isna(coin_gecko_id):
            self.logger.warning(
                f"CoinGecko ID missing for {ticker}. Cannot fetch market cap."
            )
            return current_row_data.get("last_market_cap"), current_row_data.get(
                "last_market_cap_fetch_time"
            )
        try:
            self.logger.debug(
                f"Fetching market cap for {ticker} (ID: {coin_gecko_id}) from CoinGecko."
            )
            if isinstance(coin_gecko_id, list):
                coin_gecko_id = coin_gecko_id[0] if coin_gecko_id else ""
            if not coin_gecko_id:
                self.logger.warning(
                    f"CoinGecko ID became empty for {ticker}. Cannot fetch market cap."
                )
                return current_row_data.get("last_market_cap"), current_row_data.get(
                    "last_market_cap_fetch_time"
                )

            coin_data_full = self.cg.get_coins_markets(
                vs_currency="usd", ids=coin_gecko_id
            )
            if (
                coin_data_full
                and isinstance(coin_data_full, list)
                and len(coin_data_full) > 0
            ):
                market_cap = coin_data_full[0].get("market_cap")
                if market_cap is not None:
                    self.logger.info(
                        f"Live market cap for {ticker}: {market_cap:,.2f} USD"
                    )
                    return (
                        float(market_cap),
                        datetime.datetime.now(datetime.timezone.utc).timestamp(),
                    )
                else:
                    self.logger.warning(
                        f"Market cap not found in CoinGecko response for {ticker} (ID: {coin_gecko_id})."
                    )
            else:
                self.logger.warning(
                    f"Empty or invalid response from CoinGecko for market cap of {ticker} (ID: {coin_gecko_id}). Response: {coin_data_full}"
                )
        except Exception as e:
            self.logger.error(
                f"Error fetching market cap from CoinGecko for {ticker} (ID: {coin_gecko_id}): {e}",
                exc_info=True,
            )

        last_mc = current_row_data.get("last_market_cap")
        last_mc_time = current_row_data.get("last_market_cap_fetch_time")
        if pd.notna(last_mc) and pd.notna(last_mc_time):
            age_seconds = (
                datetime.datetime.now(datetime.timezone.utc).timestamp() - last_mc_time
            )
            if age_seconds < self.market_cap_stale_threshold_seconds:
                self.logger.info(
                    f"Using stored (non-stale) market cap for {ticker}: {last_mc:,.2f} USD (fetched {age_seconds / 3600:.1f} hours ago)."
                )
                return float(last_mc), last_mc_time
            else:
                self.logger.warning(
                    f"Stored market cap for {ticker} is STALE ({last_mc:,.2f} USD, fetched {age_seconds / 3600:.1f} hours ago). Using it as last resort."
                )
                return float(last_mc), last_mc_time
        self.logger.warning(
            f"No valid stored market cap for {ticker} to use as fallback."
        )
        return None, None

    def _check_position(
        self, user_dataframe: pd.DataFrame, row_idx: int
    ) -> pd.DataFrame:
        portfolio_df = user_dataframe
        try:
            row_data = portfolio_df.loc[row_idx]
            ticker = row_data["ticker"]
            asset_exchange_id = row_data["exchange"]
            strategy_name = row_data["strategy_name"]

            if not self.primary_stable_coin_ticker:
                self.logger.error(
                    f"Primary stable coin not set. Cannot check position for {ticker}."
                )
                return portfolio_df
            if ticker == self.primary_stable_coin_ticker or asset_exchange_id == "N/A":
                return portfolio_df

            client = self.get_client(asset_exchange_id)
            if not client:
                self.logger.warning(
                    f"No client for '{asset_exchange_id}' to check position for {ticker}."
                )
                return portfolio_df

            ccxt_pair = f"{ticker}/{self.primary_stable_coin_ticker}"
            ccxt_timeframe = self._get_ccxt_timeframe(row_data["position_interval"])
            self.logger.debug(
                f"Checking position for {ticker} on {asset_exchange_id} ({ccxt_pair}, {ccxt_timeframe})"
            )

            current_bars_to_fetch = self.indicator_history_size
            ohlcv_df = pd.DataFrame()
            position_signals = pd.Series(dtype=float)

            while current_bars_to_fetch <= self.max_indicator_history_size:
                self.logger.info(
                    f"[{ticker}] Checking position with {current_bars_to_fetch} bars..."
                )

                ohlcv_df = self._fetch_ohlcv_ccxt(
                    client=client,
                    symbol_pair=ccxt_pair,
                    timeframe=ccxt_timeframe,
                    n_bars=current_bars_to_fetch,
                )

                if ohlcv_df is None or ohlcv_df.empty:
                    self.logger.warning(
                        f"[{ticker}] OHLCV fetch failed or is empty with {current_bars_to_fetch} bars."
                    )
                    break

                end_of_history = len(ohlcv_df) < current_bars_to_fetch

                try:
                    strategy_class = IndicatorFactory.get_strategy_class(strategy_name)
                    params = strategy_class.prepare_signal_params(
                        row_data, param_prefix="position"
                    )
                    indicator_strategy = IndicatorFactory.get_strategy_instance(
                        strategy_name
                    )
                    position_signals = indicator_strategy.calculate_signals(
                        ohlcv_df.copy(), **params
                    )

                    if not position_signals.empty:
                        latest_signal = position_signals.iloc[-1]
                        self.logger.info(
                            f"[{ticker}] Signal with {current_bars_to_fetch} bars is: {latest_signal}"
                        )
                        if latest_signal != 0.0:
                            break  # Found non-neutral signal
                except Exception as e:
                    self.logger.error(
                        f"Error calculating position indicator for {ticker}: {e}",
                        exc_info=True,
                    )
                    break

                if end_of_history:
                    self.logger.info(
                        f"[{ticker}] Reached end of history ({len(ohlcv_df)} bars). Using final neutral signal."
                    )
                    break

                current_bars_to_fetch += self.indicator_history_size

            if ohlcv_df is not None and not ohlcv_df.empty:
                portfolio_df.loc[row_idx, "price"] = ohlcv_df.close.iloc[-1]

                if self.ohlcv_export_dir:
                    try:
                        Path(self.ohlcv_export_dir).mkdir(parents=True, exist_ok=True)
                        filename = f"{ticker}_{asset_exchange_id}_position_{ccxt_timeframe}.csv"
                        export_path = Path(self.ohlcv_export_dir) / filename
                        ohlcv_df.to_csv(export_path)
                        self.logger.debug(
                            f"Exported position OHLCV data for {ticker} to {export_path}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to export position OHLCV data for {ticker}: {e}"
                        )

                if not position_signals.empty and len(position_signals) >= 2:
                    portfolio_df.loc[row_idx, "position_signal_prev_1"] = (
                        position_signals.iloc[-2]
                    )
                    portfolio_df.loc[row_idx, "position_signal_prev_2"] = (
                        position_signals.iloc[-3] if len(position_signals) >= 3 else 0.0
                    )
                else:
                    self.logger.warning(
                        f"Not enough indicator data for {ticker} for PositionSignal_Prev1/2. Setting to 0."
                    )
                    portfolio_df.loc[row_idx, "position_signal_prev_1"] = 0.0
                    portfolio_df.loc[row_idx, "position_signal_prev_2"] = 0.0
            else:
                self.logger.warning(
                    f"OHLCV DataFrame empty for {ticker}. Price and PositionSignals not updated."
                )

            p_interval_val = row_data["position_interval"]
            portfolio_df.loc[row_idx, "position_last_check_bucket"] = (
                (
                    datetime.datetime.now(datetime.timezone.utc).timestamp()
                    // p_interval_val
                )
                if p_interval_val > 0
                else 0
            )
        except Exception as e:
            self.logger.error(
                f"Error in _check_position for {portfolio_df.loc[row_idx, 'ticker']}: {e}",
                exc_info=True,
            )
        try:
            self._validate_trend_position_signals(portfolio_df, row_idx)
        except Exception:
            # Be conservative: don't raise here, just log if logger exists
            if hasattr(self, "logger"):
                self.logger.debug(
                    f"Signal validation failed for row {row_idx}, continuing."
                )
        return portfolio_df

    def _check_trend(self, user_dataframe: pd.DataFrame, row_idx: int) -> pd.DataFrame:
        portfolio_df = user_dataframe
        try:
            row_data = portfolio_df.loc[row_idx]
            ticker = row_data["ticker"]
            asset_exchange_id = row_data["exchange"]
            coin_gecko_id = row_data["coingecko_id"]
            strategy_name = row_data["strategy_name"]

            self._check_trend_price_component(
                portfolio_df,
                row_idx,
                row_data,
                ticker,
                asset_exchange_id,
                strategy_name,
            )
            self._check_trend_market_cap(
                portfolio_df, row_idx, row_data, ticker, coin_gecko_id
            )

            t_interval_val = row_data["trend_interval"]
            portfolio_df.loc[row_idx, "trend_last_check_bucket"] = (
                (
                    datetime.datetime.now(datetime.timezone.utc).timestamp()
                    // t_interval_val
                )
                if t_interval_val > 0
                else 0
            )
        except Exception as e:
            self.logger.error(
                f"Error in _check_trend for {portfolio_df.loc[row_idx, 'ticker']}: {e}",
                exc_info=True,
            )
        return portfolio_df

    def _calculate_historical_dominance_ohlcv(
        self, target_ticker: str, timeframe: str, n_bars: int = 1000
    ) -> pd.DataFrame:
        """
        Calculates a historical OHLC DataFrame based on an asset's market cap dominance
        relative to the total market cap of all assets in the portfolio.
        """
        self.logger.info(
            f"Calculating historical dominance OHLCV for {target_ticker} with {n_bars} bars..."
        )

        # --- Step 1: Fetch historical price data for all tradable assets ---
        all_ohlcv_data = {}
        for idx, row in self.portfolio_df[self.portfolio_df["level"] != 0].iterrows():
            ticker = row["ticker"]
            exchange_id = row["exchange"]
            client = self.get_client(exchange_id)

            if not client or exchange_id == "N/A":
                self.logger.warning(
                    f"Skipping {ticker} for dominance calculation: no valid client or exchange."
                )
                continue

            pair_symbol = f"{ticker}/{self.primary_stable_coin_ticker}"
            self.logger.debug(
                f"Fetching OHLCV for {pair_symbol} on {exchange_id} for dominance calculation."
            )
            ohlcv_df = self._fetch_ohlcv_ccxt(client, pair_symbol, timeframe, n_bars)

            if ohlcv_df.empty:
                self.logger.warning(
                    f"Could not fetch OHLCV for {ticker}. It will be excluded from total market cap calculation."
                )
                continue

            all_ohlcv_data[ticker] = ohlcv_df

        if not all_ohlcv_data:
            self.logger.error(
                "Failed to fetch any OHLCV data. Cannot calculate dominance."
            )
            return pd.DataFrame()

        # --- Step 2: Estimate historical market cap for each asset ---
        all_historical_mc_df = {}
        for ticker, ohlcv_df in all_ohlcv_data.items():
            asset_row = self.portfolio_df[self.portfolio_df["ticker"] == ticker].iloc[0]
            last_mc = asset_row.get("last_market_cap")
            current_price = asset_row.get("price")

            if (
                pd.isna(last_mc)
                or last_mc <= 0
                or pd.isna(current_price)
                or current_price <= 0
            ):
                self.logger.warning(
                    f"Missing last_market_cap or price for {ticker}. Cannot estimate its historical market cap."
                )
                continue

            # Estimate historical MC based on price change
            # MC_hist = (last_mc / current_price) * price_hist
            mc_df = pd.DataFrame(index=ohlcv_df.index)
            mc_df["open"] = (last_mc / current_price) * ohlcv_df["open"]
            mc_df["high"] = (last_mc / current_price) * ohlcv_df["high"]
            mc_df["low"] = (last_mc / current_price) * ohlcv_df["low"]
            mc_df["close"] = (last_mc / current_price) * ohlcv_df["close"]
            all_historical_mc_df[ticker] = mc_df

        if not all_historical_mc_df:
            self.logger.error(
                "Could not estimate any historical market caps. Aborting dominance calculation."
            )
            return pd.DataFrame()

        # --- Step 3: Calculate total historical market cap of the portfolio ---
        # Concatenate all historical MC dataframes
        combined_mc_df = pd.concat(
            all_historical_mc_df.values(), axis=1, keys=all_historical_mc_df.keys()
        )

        # Sum up the market caps for each candle (open, high, low, close)
        total_mc_df = pd.DataFrame(index=combined_mc_df.index)
        total_mc_df["open"] = combined_mc_df.loc[:, (slice(None), "open")].sum(axis=1)
        total_mc_df["high"] = combined_mc_df.loc[:, (slice(None), "high")].sum(axis=1)
        total_mc_df["low"] = combined_mc_df.loc[:, (slice(None), "low")].sum(axis=1)
        total_mc_df["close"] = combined_mc_df.loc[:, (slice(None), "close")].sum(
            axis=1
        )

        # --- Step 4: Calculate dominance for the target asset ---
        target_asset_mc_df = all_historical_mc_df.get(target_ticker)
        if target_asset_mc_df is None:
            self.logger.error(
                f"Historical market cap for target asset {target_ticker} could not be calculated. Cannot compute dominance."
            )
            return pd.DataFrame()

        # Avoid division by zero
        total_mc_df.replace(0, np.nan, inplace=True)

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

        # Add volume from the original price chart for compatibility with indicators if needed
        dominance_ohlcv_df["volume"] = all_ohlcv_data[target_ticker]["volume"]

        dominance_ohlcv_df.dropna(inplace=True)

        self.logger.info(
            f"Successfully calculated historical dominance OHLCV for {target_ticker}."
        )
        self.logger.debug(
            f"Dominance for {target_ticker} (last 5 periods):\n{dominance_ohlcv_df.tail()}"
        )

        return dominance_ohlcv_df

    def _check_trend_price_component(
        self, portfolio_df, row_idx, row_data, ticker, asset_exchange_id, strategy_name
    ):
        if (
            not self.primary_stable_coin_ticker
            or ticker == self.primary_stable_coin_ticker
            or asset_exchange_id == "N/A"
        ):
            return

        ccxt_timeframe = self._get_ccxt_timeframe(row_data["trend_interval"])
        self.logger.debug(
            f"Calculating TREND based on DOMINANCE for {ticker} ({ccxt_timeframe})"
        )

        current_bars_to_fetch = self.indicator_history_size
        ohlcv_df_trend = pd.DataFrame()
        trend_signals = pd.Series(dtype=float)

        while current_bars_to_fetch <= self.max_indicator_history_size:
            self.logger.info(
                f"[{ticker}] Calculating TREND with {current_bars_to_fetch} bars of dominance data..."
            )
            ohlcv_df_trend = self._calculate_historical_dominance_ohlcv(
                target_ticker=ticker,
                timeframe=ccxt_timeframe,
                n_bars=current_bars_to_fetch,
            )

            if ohlcv_df_trend is None or ohlcv_df_trend.empty:
                self.logger.warning(
                    f"[{ticker}] Dominance OHLCV empty for TREND with {current_bars_to_fetch} bars."
                )
                break

            end_of_history = len(ohlcv_df_trend) < current_bars_to_fetch

            try:
                strategy_class = IndicatorFactory.get_strategy_class(strategy_name)
                params = strategy_class.prepare_signal_params(
                    row_data, param_prefix="trend"
                )
                indicator_strategy = IndicatorFactory.get_strategy_instance(
                    strategy_name
                )
                trend_signals = indicator_strategy.calculate_signals(
                    ohlcv_df_trend.copy(), **params
                )

                if not trend_signals.empty:
                    latest_signal = trend_signals.iloc[-1]
                    self.logger.info(
                        f"[{ticker}] Trend signal with {current_bars_to_fetch} bars is: {latest_signal}"
                    )
                    if latest_signal != 0.0:
                        break  # Found non-neutral signal
            except Exception as e:
                self.logger.error(
                    f"Error calculating trend indicator for {ticker}: {e}",
                    exc_info=True,
                )
                break

            if end_of_history:
                self.logger.info(
                    f"[{ticker}] Reached end of history for dominance ({len(ohlcv_df_trend)} bars). Using neutral signal."
                )
                break

            current_bars_to_fetch += self.indicator_history_size

        if ohlcv_df_trend is not None and not ohlcv_df_trend.empty:
            self._export_trend_ohlcv_data(
                ohlcv_df_trend, f"{ticker}_dominance", asset_exchange_id, ccxt_timeframe
            )
            if not trend_signals.empty and len(trend_signals) >= 2:
                portfolio_df.loc[row_idx, "trend_signal_prev_1"] = trend_signals.iloc[
                    -2
                ]
                portfolio_df.loc[row_idx, "trend_signal_prev_2"] = (
                    trend_signals.iloc[-3] if len(trend_signals) >= 3 else 0.0
                )
            else:
                self.logger.warning(
                    f"Not enough TREND indicator data for {ticker}. Setting to 0."
                )
                portfolio_df.loc[row_idx, "trend_signal_prev_1"] = 0.0
                portfolio_df.loc[row_idx, "trend_signal_prev_2"] = 0.0
        else:
            self.logger.warning(
                f"Dominance OHLCV DataFrame for TREND empty for {ticker}. TrendSignals not updated."
            )

    def _export_trend_ohlcv_data(
        self, ohlcv_df_trend, ticker, asset_exchange_id, ccxt_timeframe
    ):
        if self.ohlcv_export_dir:
            try:
                Path(self.ohlcv_export_dir).mkdir(parents=True, exist_ok=True)
                filename = f"{ticker}_{asset_exchange_id}_trend_{ccxt_timeframe}.csv"
                export_path = Path(self.ohlcv_export_dir) / filename
                ohlcv_df_trend.to_csv(export_path)
                self.logger.debug(
                    f"Exported trend OHLCV data for {ticker} to {export_path}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to export trend OHLCV data for {ticker}: {e}"
                )

    def _calculate_trend_signals(
        self, portfolio_df, row_idx, ohlcv_df_trend, strategy_name, row_data
    ):
        try:
            strategy_class = IndicatorFactory.get_strategy_class(strategy_name)
            params = strategy_class.prepare_signal_params(
                row_data, param_prefix="trend"
            )

            indicator_strategy = IndicatorFactory.get_strategy_instance(strategy_name)
            trend_signals = indicator_strategy.calculate_signals(
                ohlcv_df_trend.copy(), **params
            )

            if len(trend_signals) >= 2:
                portfolio_df.loc[row_idx, "trend_signal_prev_1"] = trend_signals.iloc[
                    -2
                ]
                if len(trend_signals) >= 3:
                    portfolio_df.loc[row_idx, "trend_signal_prev_2"] = (
                        trend_signals.iloc[-3]
                    )
                else:
                    portfolio_df.loc[row_idx, "trend_signal_prev_2"] = 0.0
            else:
                self.logger.warning(
                    f"Not enough indicator data for TREND {row_data['ticker']} for TrendSignal_Prev1/2. Setting to 0."
                )
                portfolio_df.loc[row_idx, "trend_signal_prev_1"] = 0.0
                portfolio_df.loc[row_idx, "trend_signal_prev_2"] = 0.0

            try:
                self._validate_trend_position_signals(portfolio_df, row_idx)
            except Exception:
                if hasattr(self, "logger"):
                    self.logger.debug(
                        f"Signal validation failed for row {row_idx} after trend calculation."
                    )

        except Exception as e:
            self.logger.error(
                f"Error calculating trend indicator for {row_data['ticker']} using strategy '{strategy_name}': {e}",
                exc_info=True,
            )
            portfolio_df.loc[row_idx, "trend_signal_prev_1"] = 0.0
            portfolio_df.loc[row_idx, "trend_signal_prev_2"] = 0.0

    def _validate_trend_position_signals(
        self, portfolio_df: pd.DataFrame, row_idx: int
    ):
        """
        Ensure paired consistency between 'position_signal_prev_1' and 'trend_signal_prev_1'.
        If either is neutral (== 0), set both to -1.0 (interpreted as a conservative 'sell'/avoid signal).
        """
        try:
            pos = portfolio_df.loc[row_idx, "position_signal_prev_1"]
            trend = portfolio_df.loc[row_idx, "trend_signal_prev_1"]
        except Exception:
            return

        # Treat NaNs as non-neutral here (no-op). Only act when explicit 0 encountered.
        try:
            if (pos == 0) or (trend == 0):
                portfolio_df.loc[row_idx, "position_signal_prev_1"] = -1.0
                portfolio_df.loc[row_idx, "trend_signal_prev_1"] = -1.0
        except Exception:
            # Defensive: ignore issues when writing back
            if hasattr(self, "logger"):
                self.logger.debug(f"Failed to set validated signals for row {row_idx}.")

    def _check_trend_market_cap(
        self, portfolio_df, row_idx, row_data, ticker, coin_gecko_id
    ):
        if (
            ticker == self.primary_stable_coin_ticker
            or pd.isna(coin_gecko_id)
            or not coin_gecko_id
        ):
            return

        market_cap, fetch_time = self._fetch_market_cap(coin_gecko_id, ticker, row_data)
        if market_cap is not None:
            portfolio_df.loc[row_idx, "last_market_cap"] = market_cap
            if (
                fetch_time is not None
                and fetch_time != row_data["last_market_cap_fetch_time"]
            ):
                portfolio_df.loc[row_idx, "last_market_cap_fetch_time"] = fetch_time
        else:
            self.logger.error(
                f"Failed to get current or stored market cap for {ticker}. 'last_market_cap' will be 0 if not already set."
            )
            if pd.isna(portfolio_df.loc[row_idx, "last_market_cap"]):
                portfolio_df.loc[row_idx, "last_market_cap"] = 0.0

    def _distribute_portfolio(self):
        """
        Orchestrates the entire portfolio distribution process, from initial checks
        to final state updates, by calling a series of helper methods.
        """
        self.logger.info("Starting portfolio distribution cycle...")
        if not self._is_distribution_possible():
            self.logger.error(
                "Portfolio distribution aborted due to failed pre-checks."
            )
            return

        # 1. Initialize balances from exchanges or config
        total_stablecoin_free = self._initialize_balances()
        self._update_stablecoin_total_in_df(total_stablecoin_free)

        # 2. Update prices and calculate total portfolio value
        self._update_asset_prices()
        total_portfolio_value = self._calculate_total_portfolio_value()

        if total_portfolio_value is None:  # Handles zero-value portfolio
            return

        # 3. Calculate allocation targets based on strategy rules
        self._calculate_allocation_targets(total_portfolio_value)

        # 4. Generate and execute trades to meet targets
        trade_proposals = self._generate_optimized_trade_proposals()
        if trade_proposals:
            self._execute_trade_proposals(trade_proposals)
        else:
            self.logger.info("No rebalancing trades are required at this time.")

        # 5. Finalize portfolio state post-distribution
        self._finalize_portfolio_state()
        self.logger.info("Portfolio distribution cycle completed successfully.")

    def _is_distribution_possible(self) -> bool:
        """Performs initial sanity checks to ensure distribution can proceed."""
        if self.portfolio_df.empty:
            self.logger.error("Portfolio is empty. Cannot distribute portfolio.")
            return False
        if not self.exchange_clients:
            self.logger.error(
                "No CCXT clients configured. Cannot distribute portfolio."
            )
            return False
        if not self.primary_stable_coin_ticker:
            self.logger.error(
                "Primary stablecoin not identified. Cannot distribute portfolio."
            )
            return False
        if not self.markets_by_exchange:
            self.logger.error(
                "Market data not available for any exchange. Cannot distribute portfolio."
            )
            return False
        return True

    def _initialize_balances(self) -> float:
        """Initializes asset balances, either from config (dry run) or by fetching from exchanges (live)."""
        self.logger.debug("Determining balances for distribution...")
        self.stablecoin_balance_by_exchange = {}

        if self.dry_run:
            self.logger.warning(
                "[DRY RUN] Initializing balances from config `initial_stablecoin_balance`."
            )
            total_stablecoin_free = self.initial_stablecoin_balance
            if self.exchange_clients:
                first_exchange_id = next(iter(self.exchange_clients))
                self.stablecoin_balance_by_exchange[first_exchange_id] = (
                    total_stablecoin_free
                )
            # Also update the stablecoin's 'free' balance in the portfolio_df for dry run
            stable_coin_indices = self.portfolio_df[
                self.portfolio_df["ticker"] == self.primary_stable_coin_ticker
            ].index
            if not stable_coin_indices.empty:
                self.portfolio_df.loc[stable_coin_indices[0], "free"] = total_stablecoin_free
                self.portfolio_df.loc[stable_coin_indices[0], "price"] = 1.0 # Ensure price is 1.0 for stablecoin
        else:
            self.logger.info("[LIVE MODE] Fetching real balances from exchanges.")
            total_stablecoin_free = 0.0
            for exchange_id, client in self.exchange_clients.items():
                try:
                    balance_data = client.fetch_balance()
                    # Update free/used for all assets on this exchange
                    for idx, row in self.portfolio_df[
                        self.portfolio_df["exchange"] == exchange_id
                    ].iterrows():
                        ticker = row["ticker"]
                        self.portfolio_df.loc[idx, "free"] = balance_data.get(
                            "free", {}
                        ).get(ticker, 0.0)
                        self.portfolio_df.loc[idx, "used"] = balance_data.get(
                            "used", {}
                        ).get(ticker, 0.0)

                    # Track stablecoin balance specifically
                    stable_on_exchange = balance_data.get("free", {}).get(
                        self.primary_stable_coin_ticker, 0.0
                    )
                    self.stablecoin_balance_by_exchange[exchange_id] = (
                        stable_on_exchange
                    )
                    total_stablecoin_free += stable_on_exchange
                    self.logger.debug(
                        f"Balances fetched for {exchange_id}. Stable ({self.primary_stable_coin_ticker}): {stable_on_exchange:.2f}"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Failed to fetch balance for {exchange_id}: {e}", exc_info=True
                    )
                    self.stablecoin_balance_by_exchange[exchange_id] = 0.0

        return total_stablecoin_free

    def _update_stablecoin_total_in_df(self, total_stablecoin_free: float):
        """Updates the primary stablecoin's 'free' and 'price' in the main portfolio DataFrame."""
        stable_coin_indices = self.portfolio_df[
            self.portfolio_df["ticker"] == self.primary_stable_coin_ticker
        ].index
        if not stable_coin_indices.empty:
            idx = stable_coin_indices[0]
            self.portfolio_df.loc[idx, "free"] = total_stablecoin_free
            self.portfolio_df.loc[idx, "price"] = 1.0

    def _update_asset_prices(self):
        """Fetches and updates the current price for all non-stablecoin assets in the portfolio."""
        self.logger.info("Updating live prices for portfolio assets...")
        for idx, row in self.portfolio_df.iterrows():
            if row["level"] == 0 or row["exchange"] == "N/A":
                continue

            # Update price only if it's missing or zero
            if pd.isna(row["price"]) or row["price"] <= 0:
                client = self.get_client(row["exchange"])
                if not client:
                    self.portfolio_df.loc[idx, "price"] = 0
                    continue

                pair_symbol = f"{row['ticker']}/{self.primary_stable_coin_ticker}"
                try:
                    ticker_data = client.fetch_ticker(pair_symbol)
                    live_price = ticker_data.get("last", ticker_data.get("close"))
                    self.portfolio_df.loc[idx, "price"] = (
                        live_price if live_price and live_price > 0 else 0
                    )
                except Exception:
                    self.logger.warning(
                        f"Could not fetch price for {pair_symbol} on {row['exchange']}. Setting price to 0."
                    )
                    self.portfolio_df.loc[idx, "price"] = 0

    def _calculate_total_portfolio_value(self) -> float | None:
        """Calculates the total value of the portfolio in the primary stablecoin."""
        self.portfolio_df["value_in_stable"] = (
            self.portfolio_df["free"] * self.portfolio_df["price"]
        ).fillna(0.0)
        total_value = self.portfolio_df["value_in_stable"].sum()

        if total_value <= 1e-6:
            self.logger.warning(
                "Total portfolio value is effectively zero. Skipping distribution trades."
            )
            # Set default values for a zero-value portfolio
            self.portfolio_df["ratio"] = 0.0
            stable_idx = self.portfolio_df[self.portfolio_df["level"] == 0].index
            if not stable_idx.empty:
                self.portfolio_df.loc[stable_idx, "ratio"] = 1.0
            self.portfolio_df["target_value"] = 0.0
            self.portfolio_df["change"] = 0.0
            return None

        self.logger.info(
            f"Total portfolio value (in {self.primary_stable_coin_ticker}): {total_value:,.2f}"
        )
        self.logger.info(
            f"Stablecoin balances by exchange: {self.stablecoin_balance_by_exchange}"
        )
        return total_value

    def _calculate_allocation_targets(self, total_portfolio_value: float):
        """
        Calculates allocation targets based on a two-tiered signal strategy.
        1. The portfolio is partitioned among assets with a positive TREND signal.
        2. For each of these trending assets, the POSITION signal determines whether
           its allocated share is held in the asset itself (buy signal) or in the
           stablecoin (sell signal).
        3. A cap is applied to the total allocation for altcoins (level >= 2).
        """
        self.logger.info(
            "Calculating allocation targets based on trend/position strategy..."
        )
        df = self.portfolio_df
        df["last_market_cap"] = pd.to_numeric(
            df["last_market_cap"], errors="coerce"
        ).fillna(0)

        # Calculate Dominance relative to all tradable assets
        tradable_assets = df[df["level"] != 0]
        total_mc = tradable_assets["last_market_cap"].sum()
        df["dominance"] = df["last_market_cap"] / total_mc if total_mc > 0 else 0.0
        df.loc[df["level"] == 0, "dominance"] = 0.0

        # 1. Identify the universe of assets with a positive long-term TREND signal.
        #    These are the only assets that get a "strategic allocation slot".
        trending_mask = (
            (df["level"] != 0)
            & (df["exchange"] != "N/A")
            & (df["trend_signal_prev_1"] == 1)
        )
        trending_indices = df[trending_mask].index
        sum_of_trending_dominances = df.loc[trending_indices, "dominance"].sum()

        # Initialize all ratios to zero.
        df["ratio"] = 0.0
        stable_coin_indices = df[df["level"] == 0].index

        if not stable_coin_indices.empty and sum_of_trending_dominances > 0:
            stable_coin_idx = stable_coin_indices[0]

            # Calculate the "slot size" (ratio) for each trending asset based on its dominance.
            # This ratio represents the asset's share of the risk-on portion of the portfolio.
            df.loc[trending_indices, "ratio"] = (
                df.loc[trending_indices, "dominance"] / sum_of_trending_dominances
            )

            # 2. Check the short-term POSITION signal for each trending asset.
            # If the signal is not 'buy' (1), move its allocated ratio to the stablecoin.
            for idx in trending_indices:
                if df.loc[idx, "position_signal_prev_1"] != 1:
                    ratio_to_move = df.loc[idx, "ratio"]
                    df.loc[stable_coin_idx, "ratio"] += ratio_to_move
                    df.loc[idx, "ratio"] = 0.0
                    self.logger.info(
                        f"Asset {df.loc[idx, 'ticker']} is trending but has a sell signal. "
                        f"Moving its ratio ({ratio_to_move:.4f}) to stablecoin."
                    )
        else:
            # If no assets are trending, allocate 100% to the stablecoin.
            self.logger.info(
                "No assets in a positive trend. Allocating 100% to stablecoin."
            )
            if not stable_coin_indices.empty:
                df.loc[stable_coin_indices[0], "ratio"] = 1.0

        # Cap Altcoin (Level >= 2) Allocation ---
        if self.max_alt_coin_ratio < 1.0:  # Only run if a cap is set
            altcoin_mask = df["level"] >= 2
            current_altcoin_ratio_sum = df.loc[altcoin_mask, "ratio"].sum()

            if current_altcoin_ratio_sum > self.max_alt_coin_ratio:
                self.logger.info(
                    f"Current altcoin (level>=2) ratio sum ({current_altcoin_ratio_sum:.4f}) "
                    f"exceeds the configured max_alt_coin_ratio ({self.max_alt_coin_ratio:.4f})."
                )

                # Calculate the scaling factor and the amount to transfer to stablecoin
                scaling_factor = self.max_alt_coin_ratio / current_altcoin_ratio_sum
                ratio_to_transfer = current_altcoin_ratio_sum * (1 - scaling_factor)

                self.logger.info(
                    f"Scaling down altcoin ratios by a factor of {scaling_factor:.4f} "
                    f"and transferring {ratio_to_transfer:.4f} to stablecoin."
                )

                # Apply the scaling factor to all altcoins
                df.loc[altcoin_mask, "ratio"] *= scaling_factor

                # Add the transferred ratio to the stablecoin
                if not stable_coin_indices.empty:
                    stable_coin_idx = stable_coin_indices[0]
                    df.loc[stable_coin_idx, "ratio"] += ratio_to_transfer
                else:
                    self.logger.warning(
                        "Could not transfer excess altcoin ratio: No stablecoin found."
                    )

        # Normalize ratios to correct any floating point inaccuracies and ensure sum is 1.0.
        # This is a safeguard.
        ratio_sum = df["ratio"].sum()
        if abs(ratio_sum - 1.0) > 1e-9:
            self.logger.warning(
                f"Ratios sum to {ratio_sum} after allocation, normalizing to 1.0."
            )
            if ratio_sum > 0:
                df["ratio"] /= ratio_sum
            elif not stable_coin_indices.empty:
                # Fallback if sum is zero (should not happen if logic above is correct)
                df["ratio"] = 0.0
                df.loc[stable_coin_indices[0], "ratio"] = 1.0

        # Calculate the final Target Value and the required Change (amount to buy/sell)
        df["target_value"] = (df["ratio"] * total_portfolio_value).fillna(0.0)
        df["change"] = (df["target_value"] - df["value_in_stable"]).fillna(0.0)

        # Apply rebalance_threshold_percentage
        for idx in df.index:
            current_change = df.loc[idx, "change"]
            target_value = df.loc[idx, "target_value"]
            
            # Only apply threshold if there's a target value to compare against
            if target_value > 0 and abs(current_change) / target_value < self.rebalance_threshold_percentage:
                df.loc[idx, "change"] = 0.0
                self.logger.debug(
                    f"Rebalance for {df.loc[idx, 'ticker']} skipped due to threshold. "
                    f"Change: {current_change:.2f}, Target: {target_value:.2f}, "
                    f"Threshold: {self.rebalance_threshold_percentage:.2%}"
                )

    def _generate_optimized_trade_proposals(self) -> list[dict]:
        """
        Generates an optimized list of trade proposals.
        This version FIXES the logical flaw of misclassifying stablecoin pairs
        as direct asset-to-asset pairs.
        """
        self.logger.info(
            "Generating optimized trade proposals with corrected logic routing..."
        )
        proposals = []

        # Create mutable copies of assets needing to be bought or sold
        buys_df = self.portfolio_df[
            (self.portfolio_df["change"] > self.min_order_value_usd)
            & (self.portfolio_df["position_signal_prev_1"] == 1)
            & (self.portfolio_df["trend_signal_prev_1"] == 1)
        ].copy()
        sells_df = self.portfolio_df[
            (self.portfolio_df["change"] < -self.min_order_value_usd)
            | (
                (self.portfolio_df["position_signal_prev_1"] == -1)
                & (self.portfolio_df["value_in_stable"] > self.min_order_value_usd)
            )
        ].copy()

        # For full sells, the change amount is the entire value
        for idx, row in sells_df.iterrows():
            if row["position_signal_prev_1"] == -1:
                sells_df.loc[idx, "change"] = -row["value_in_stable"]

        self.logger.info(
            "Attempting to match direct asset-to-asset trades (excluding stablecoin pairs)..."
        )
        # --- Direct Pair Matching Logic ---
        for sell_idx, sell_row in sells_df.iterrows():
            if abs(sell_row["change"]) < self.min_order_value_usd:
                continue

            for buy_idx, buy_row in buys_df.iterrows():
                if buy_row["change"] < self.min_order_value_usd:
                    continue

                sell_ticker = sell_row["ticker"]
                buy_ticker = buy_row["ticker"]

                if (
                    sell_ticker == self.primary_stable_coin_ticker
                    or buy_ticker == self.primary_stable_coin_ticker
                ):
                    continue

                if sell_row["exchange"] != buy_row["exchange"]:
                    continue

                exchange_id = sell_row["exchange"]
                markets = self.get_markets_for_exchange(exchange_id)

                market_info, direct_pair_symbol, trade_side = (None, None, None)
                pair1 = f"{buy_ticker}/{sell_ticker}"
                pair2 = f"{sell_ticker}/{buy_ticker}"

                if pair1 in markets:
                    market_info = markets[pair1]
                    direct_pair_symbol = pair1
                    trade_side = "buy"
                elif pair2 in markets:
                    market_info = markets[pair2]
                    direct_pair_symbol = pair2
                    trade_side = "sell"

                if market_info:
                    self.logger.info(
                        f"Found direct market: {direct_pair_symbol} on {exchange_id}"
                    )
                    trade_value = min(abs(sell_row["change"]), buy_row["change"])
                    price = (
                        buy_row["price"] / sell_row["price"]
                        if trade_side == "buy"
                        else sell_row["price"] / buy_row["price"]
                    )

                    base_currency_price_in_stable = (
                        sell_row["price"] if trade_side == "buy" else buy_row["price"]
                    )
                    trade_amount = trade_value / base_currency_price_in_stable

                    proposals.append(
                        {
                            "type": "direct",
                            "symbol": direct_pair_symbol,
                            "side": trade_side,
                            "amount": trade_amount,
                            "value_stable": trade_value,
                            "price": price,
                            "exchange_id": exchange_id,
                            "market_info": market_info,
                            "buy_asset_idx": buy_idx,
                            "sell_asset_idx": sell_idx,
                        }
                    )
                    buys_df.loc[buy_idx, "change"] -= trade_value
                    sells_df.loc[sell_idx, "change"] += trade_value
                    self.logger.info(
                        f"Proposed direct trade: {trade_side} {trade_amount:.6f} {direct_pair_symbol} "
                        f"(value: ~{trade_value:.2f} {self.primary_stable_coin_ticker})."
                    )

        # --- Stablecoin Fallback Logic ---
        self.logger.info("Handling remaining balances via stablecoin trades.")
        for df, side in [(sells_df, "sell"), (buys_df, "buy")]:
            for idx, row in df.iterrows():
                if row["level"] == 0:
                    continue
                value_stable = abs(row["change"])
                if value_stable < self.min_order_value_usd:
                    continue

                ticker = row["ticker"]
                current_price = row["price"]

                if pd.isna(current_price) or current_price <= 0:
                    self.logger.critical(
                        f"CRITICAL SAFETY-NET: Price for {ticker} is invalid (Price: {current_price}). "
                        f"Skipping trade proposal for {value_stable:.2f} USDT."
                    )
                    continue

                exchange_id = row["exchange"]
                markets = self.get_markets_for_exchange(exchange_id)
                stable_pair_symbol = f"{ticker}/{self.primary_stable_coin_ticker}"

                if not markets or stable_pair_symbol not in markets:
                    self.logger.warning(
                        f"Market {stable_pair_symbol} not on {exchange_id}, cannot create stable trade."
                    )
                    continue

                amount_coin = value_stable / current_price

                if side == "sell":
                    available_balance = row.get("free", 0.0)
                    if amount_coin > available_balance * 1.001:
                        self.logger.warning(
                            f"SAFETY-NET: Proposed sell amount for {ticker} ({amount_coin:.8f}) "
                            f"is greater than available balance ({available_balance:.8f}). Adjusting."
                        )
                        amount_coin = available_balance
                        if (amount_coin * current_price) < self.min_order_value_usd:
                            self.logger.warning(
                                f"Skipping sell for {ticker} as adjusted amount is too small."
                            )
                            continue

                proposals.append(
                    {
                        "type": "stable",
                        "symbol": stable_pair_symbol,
                        "side": side,
                        "amount": amount_coin,
                        "value_stable": value_stable,
                        "price": current_price,
                        "exchange_id": exchange_id,
                        "market_info": markets[stable_pair_symbol],
                        "asset_idx": idx,
                    }
                )

        self.logger.debug(f"Generated {len(proposals)} total trade proposals.")
        return proposals

    def _execute_trade_proposals(self, trade_proposals: list[dict]):
        """Executes a list of trade proposals, prioritizing direct trades and sells."""
        self.logger.info(f"Executing {len(trade_proposals)} trade proposals...")

        # Enhanced sorting: direct trades first, then stable sells, then stable buys
        def sort_key(p):
            if p["type"] == "direct":
                return 0  # Highest priority
            if p["side"] == "sell":
                return 1  # Second priority
            return 2  # Lowest priority

        trade_proposals.sort(key=sort_key)

        for proposal in trade_proposals:
            self._execute_single_trade(proposal)
            if not self.dry_run:
                time.sleep(1)  # Small delay between live trades

    def _execute_single_trade(self, proposal: dict):
        """Executes a single trade from a proposal dictionary, handling all checks."""
        # Unpack proposal dictionary
        symbol = proposal["symbol"]
        side = proposal["side"]
        amount_float = proposal["amount"]
        price = proposal["price"]
        ex_id = proposal["exchange_id"]
        market = proposal["market_info"]

        client = self.get_client(ex_id)
        if not client:
            self.logger.error(f"No client for {ex_id} to trade {symbol}. Skipping.")
            return

        min_cost_from_market = market.get("limits", {}).get("cost", {}).get("min")
        min_cost = min_cost_from_market if min_cost_from_market is not None else 0.0

        try:
            price = Decimal(str(price))
            amount = Decimal(str(amount_float))
        except (InvalidOperation, TypeError) as e:
            self.logger.error(
                f"Invalid decimal value for {symbol}: {e}. Skipping trade."
            )
            return

        # Re-check against bot's min_order_value_usd after all adjustments and before execution
        current_trade_value_usd = float(amount * price)
        if current_trade_value_usd < self.min_order_value_usd:
            self.logger.warning(
                f"Trade skipped: {symbol} {side} amount={amount} (value: {current_trade_value_usd:.2f} USD) "
                f"is below bot's configured minimum order value ({self.min_order_value_usd:.2f} USD)."
            )
            return

        # Also check against exchange's minimum cost
        if current_trade_value_usd < (min_cost or 0.0):
            self.logger.warning(
                f"Trade skipped: {symbol} {side} amount={amount} (value: {current_trade_value_usd:.2f} USD) "
                f"is below exchange's minimum cost ({min_cost or 0.0:.2f} USD)."
            )
            return

        # Execute order
        try:
            if self.dry_run:
                order_result = self._simulate_market_order(
                    symbol, side, amount, price, ex_id, market
                )
            else:
                # Standardize amount just before live execution
                min_prec = market.get("precision", {}).get("amount")
                min_limit = market.get("limits", {}).get("amount", {}).get("min")
                final_amount = standardized_amount(float(amount), min_prec, min_limit)

                self.logger.info(
                    f"Attempting LIVE MARKET {side.upper()} on {ex_id}: {final_amount} of {market.get('base', '')} ({symbol})"
                )
                order_result = client.create_market_order(
                    symbol=symbol, side=side, amount=float(final_amount)
                )

            if not order_result or not isinstance(order_result, dict):
                self.logger.error(
                    f"Order for {symbol} on {ex_id} returned invalid result. Skipping portfolio update."
                )
                return

            # Pass the original proposal to the update function to know how to handle it
            self._update_portfolio_post_trade(order_result, proposal)

        except (
            ccxt.InsufficientFunds,
            ccxt.InvalidOrder,
            ccxt.NetworkError,
            ccxt.ExchangeError,
        ) as e:
            self.logger.error(
                f"ORDER FAILED for {symbol} {side} on {ex_id}: {e}", exc_info=False
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error during order execution for {symbol} on {ex_id}: {e}",
                exc_info=True,
            )

    def _adjust_amount_for_balance(
        self,
        client,
        ex_id: str,
        side: str,
        symbol: str,
        amount: float,
        price: float,
        df_idx,
    ) -> tuple:
        """
        Checks available balance and adjusts trade amount if necessary.
        Returns the adjusted amount and potentially updated price.
        """
        # In dry run, we trust our calculated balances.
        if self.dry_run:
            return amount, price

        # For live trades, fetch fresh balances and ticker
        try:
            balance = client.fetch_balance()
            live_ticker = client.fetch_ticker(symbol)
            live_price = (
                live_ticker.get("ask") if side == "buy" else live_ticker.get("bid")
            )
            if not live_price:
                live_price = live_ticker.get("last", price)
        except Exception as e:
            self.logger.error(
                f"Failed to refresh balance/ticker for {symbol} on {ex_id}: {e}. Skipping trade."
            )
            return None, None

        if side == "buy":
            quote_currency = self.primary_stable_coin_ticker
            available_quote = balance.get("free", {}).get(quote_currency, 0.0)
            required_quote = amount * live_price
            if required_quote > available_quote:
                self.logger.warning(
                    f"Insufficient {quote_currency} on {ex_id} for {symbol}. Have: {available_quote:.4f}, Need: {required_quote:.4f}. Adjusting."
                )
                amount = (
                    available_quote / live_price
                ) * self.balance_adjustment_factor_buy
        elif side == "sell":
            base_currency = symbol.split("/")[0]
            available_base = balance.get("free", {}).get(base_currency, 0.0)
            if amount > available_base:
                self.logger.warning(
                    f"Attempting to sell {amount:.8f} {base_currency}, but only "
                    f"{available_base:.8f} available. Adjusting."
                )
                amount = available_base * self.balance_adjustment_factor_sell

        # Re-standardize amount after adjustment
        min_prec = (
            self.get_markets_for_exchange(ex_id)[symbol]
            .get("precision", {})
            .get("amount")
        )
        return standardized_amount(amount, min_prec, 0.0), live_price

    def _simulate_market_order(
        self, symbol, side, amount, price, exchange_id, market
    ) -> dict:
        """Creates a mock order result for dry runs."""
        value = amount * price
        self.logger.info(
            f"[DRY RUN] Simulating MARKET {side.upper()} on {exchange_id}: {amount:.8f} "
            f"of {market['base']} at ~{price:.4f} (Value: {value:.2f} {market['quote']})"
        )
        return {
            "symbol": symbol,
            "side": side,
            "type": "market",
            "amount": float(amount),
            "filled": float(amount),
            "average": float(price),
            "status": "closed",
            "cost": float(value),
            "fee": None,
            "id": f"dry_run_dist_{int(time.time())}",
        }

    def _update_portfolio_post_trade(self, order_result: dict, proposal: dict):
        """
        Updates the portfolio DataFrame and local balance estimates after a trade.
        Handles incomplete order data gracefully in LIVE mode.
        """
        side = order_result.get("side")
        symbol = order_result.get("symbol")

        filled = order_result.get("filled")
        avg_price = order_result.get("average")

        if not self.dry_run and (filled is None or avg_price is None):
            self.logger.info(
                f"LIVE order for {symbol} {side} was submitted, but result data is incomplete. "
                "Skipping local portfolio update and relying on final balance fetch for accuracy."
            )
            return

        if filled is None:
            filled = 0.0
        if avg_price is None:
            avg_price = 0.0

        trade_type = proposal["type"]
        ex_id = proposal["exchange_id"]

        if self.dry_run:
            if trade_type == "stable":
                df_idx = proposal["asset_idx"]
                cost = filled * avg_price
                if side == "buy":
                    self.portfolio_df.loc[df_idx, "free"] += filled
                    self.stablecoin_balance_by_exchange[ex_id] -= cost
                elif side == "sell":
                    self.portfolio_df.loc[df_idx, "free"] -= filled
                    self.stablecoin_balance_by_exchange[ex_id] += cost

            elif trade_type == "direct":
                buy_idx = proposal["buy_asset_idx"]
                sell_idx = proposal["sell_asset_idx"]
                if side == "buy":
                    amount_bought = filled
                    amount_sold = filled * avg_price
                else:
                    amount_sold = filled
                    amount_bought = filled * avg_price
                self.portfolio_df.loc[buy_idx, "free"] += amount_bought
                self.portfolio_df.loc[sell_idx, "free"] -= amount_sold
                self.logger.info(
                    f"[DRY RUN] Direct trade update: +{amount_bought:.6f} {self.portfolio_df.loc[buy_idx, 'ticker']}, -{amount_sold:.6f} {self.portfolio_df.loc[sell_idx, 'ticker']}"
                )

        log_prefix = "[DRY RUN] " if self.dry_run else ""
        self.logger.info(
            f"{log_prefix}Trade Executed Successfully: {side} "
            f"{filled:.6f} {symbol} @ ~{avg_price:.6f}"
        )

        if self.dry_run:
            stable_ticker = self.primary_stable_coin_ticker
            stable_idx = self.portfolio_df[
                self.portfolio_df["ticker"] == stable_ticker
            ].index
            if not stable_idx.empty:
                new_total_stable = sum(self.stablecoin_balance_by_exchange.values())
                self.portfolio_df.loc[stable_idx[0], "free"] = new_total_stable

    def _finalize_portfolio_state(self):
        """Refreshes balances in live mode and performs final cleanup checks."""
        self.logger.debug("Finalizing portfolio state after distribution cycle.")

        if not self.dry_run:
            # Re-fetch all balances to get the true state after trades.
            self.logger.info(
                "[LIVE MODE] Fetching final real balances from all exchanges."
            )
            final_total_stable = 0.0

            for exchange_id, client in self.exchange_clients.items():
                try:
                    balance_data = client.fetch_balance()
                    # Update free and used balances for all assets on this exchange
                    for idx, row in self.portfolio_df[
                        self.portfolio_df["exchange"] == exchange_id
                    ].iterrows():
                        ticker = row["ticker"]
                        self.portfolio_df.loc[idx, "free"] = balance_data.get(
                            "free", {}
                        ).get(ticker, 0.0)
                        self.portfolio_df.loc[idx, "used"] = balance_data.get(
                            "used", {}
                        ).get(ticker, 0.0)

                    # Track stablecoin balance specifically
                    stable_on_exchange = balance_data.get("free", {}).get(
                        self.primary_stable_coin_ticker, 0.0
                    )
                    self.stablecoin_balance_by_exchange[exchange_id] = (
                        stable_on_exchange
                    )
                    final_total_stable += stable_on_exchange
                    self.logger.debug(
                        f"Balances fetched for {exchange_id}. Stable ({self.primary_stable_coin_ticker}): {stable_on_exchange:.2f}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to fetch balance for {exchange_id}: {e}", exc_info=True
                    )
                    self.stablecoin_balance_by_exchange[exchange_id] = 0.0

            # Update total stablecoin balance in the portfolio DataFrame
            self._update_stablecoin_total_in_df(final_total_stable)

        # Recalculate value based on final balances.
            self.portfolio_df["value_in_stable"] = (
                self.portfolio_df["free"] * self.portfolio_df["price"]
            ).fillna(0.0)

    @retry_on_network_error
    def _fetch_current_total_stablecoin_balance(self) -> float:
        """
        Fetches the current total free stablecoin balance across all configured exchanges.
        This method does NOT update the portfolio_df, it only returns the sum.
        """
        if not self.primary_stable_coin_ticker:
            self.logger.warning("Primary stablecoin not set. Cannot fetch total balance.")
            return 0.0

        total_stablecoin_free = 0.0
        if self.dry_run:
            # In dry run, we use the initial_stablecoin_balance as the current balance
            total_stablecoin_free = self.initial_stablecoin_balance
            self.logger.debug(f"[DRY RUN] Current total stablecoin balance: {total_stablecoin_free:.2f}")
        else:
            for exchange_id, client in self.exchange_clients.items():
                try:
                    balance_data = client.fetch_balance()
                    stable_on_exchange = balance_data.get("free", {}).get(
                        self.primary_stable_coin_ticker, 0.0
                    )
                    total_stablecoin_free += stable_on_exchange
                    self.logger.debug(
                        f"Fetched stablecoin balance for {exchange_id}: {stable_on_exchange:.2f}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to fetch balance for {exchange_id} during total balance check: {e}",
                        exc_info=True,
                    )
        return total_stablecoin_free

    def run_cycle(self):
        self.logger.info(
            f"--- Starting new cycle: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} ---"
        )
        self.logger.info("Clearing cycle-level OHLCV cache.")
        self.cycle_ohlcv_cache = {}

        try:
            self._load_portfolio_data()
        except Exception as e:
            self.logger.error(
                f"Critical error loading portfolio data in run_cycle: {e}. Cycle skipped.",
                exc_info=True,
            )
            return

        if self.portfolio_df.empty:
            self.logger.warning("Portfolio is empty. Skipping operational cycle logic.")
            self.logger.info(
                f"--- Cycle finished (portfolio empty): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} ---"
            )
            return

        self._initialize_cycle()

        # --- Logic to define missing variables ---
        action_needed_overall = False
        action_reasons = []

        # 1. Check for significant balance changes
        current_total_stablecoin_balance = self._fetch_current_total_stablecoin_balance()
        if self.last_total_stablecoin_balance is None:
            # First run, set the balance and assume an action is needed to initialize.
            self.last_total_stablecoin_balance = current_total_stablecoin_balance
            balance_changed = True
            action_reasons.append("First cycle initialization")
        else:
            balance_diff = abs(current_total_stablecoin_balance - self.last_total_stablecoin_balance)
            # Check if difference is greater than the minimum order value to avoid minor fluctuations
            if balance_diff > self.min_order_value_usd:
                balance_changed = True
                action_reasons.append(f"Stablecoin balance changed by {balance_diff:.2f}")
            else:
                balance_changed = False
        
        # 2. Check for new signals
        self.logger.info("Checking for new signal intervals to update portfolio state...")
        signals_changed, signal_reasons = self._update_signals_on_interval()
        if signals_changed:
            action_needed_overall = True
            action_reasons.extend(signal_reasons)
        self.logger.info("Signal interval check complete.")

        # --- End of new logic ---

        if action_needed_overall or balance_changed:
            self.logger.info(
                f"Overall action condition met. Reasons: {'; '.join(list(set(action_reasons)))}. "
                f"Proceeding with portfolio distribution and rebalancing."
            )
            if (
                self.active_clients_for_cycle
                and self.primary_stable_coin_ticker
                and self.markets_by_exchange
            ):
                self._distribute_portfolio()
                self.portfolio_df.loc[
                    (self.portfolio_df["level"] != 0)
                    & (self.portfolio_df["exchange"] != "N/A"),
                    "done",
                ] = True
                self._save_portfolio_status()
                # Update last_total_stablecoin_balance after distribution
                self.last_total_stablecoin_balance = self._fetch_current_total_stablecoin_balance()
            else:
                self.logger.warning(
                    "Action needed, but no active CCXT clients, stablecoin, or markets ready. Skipping portfolio distribution."
                )
        else:
            self.logger.info("No signals changed and no significant balance change. Skipping portfolio distribution.")

        self._export_portfolio_data()

        if self.telegram_bot_token and self.telegram_chat_id:
            self.logger.info("Sending cycle summary to Telegram...")
            asyncio.run(
                send_info(
                    self.log_file_path,
                    self.telegram_bot_token,
                    self.telegram_chat_id,
                    self.logger,
                )
            )
            if Path(self.global_status_file_path).exists():
                asyncio.run(
                    send_info(
                        self.global_status_file_path,
                        self.telegram_bot_token,
                        self.telegram_chat_id,
                        self.logger,
                    )
                )

        self.logger.info(
            f"--- Cycle finished: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} ---"
        )

    def _initialize_cycle(self):
        stable_coin_series = self.portfolio_df.loc[
            self.portfolio_df["level"] == 0, "ticker"
        ]
        if not stable_coin_series.empty:
            self.primary_stable_coin_ticker = stable_coin_series.iloc[0]
        else:
            self.logger.error(
                "No stablecoin (Level 0) defined. Critical operations will fail."
            )
            self.primary_stable_coin_ticker = None

        self.markets_by_exchange = {}
        self.active_clients_for_cycle = {}
        for exchange_id, client_instance in self.exchange_clients.items():
            try:
                markets_data_list = client_instance.fetch_markets()
                self.markets_by_exchange[exchange_id] = {
                    m["symbol"]: m for m in markets_data_list
                }
                self.active_clients_for_cycle[exchange_id] = client_instance
                self.logger.debug(
                    f"Fetched {len(self.markets_by_exchange[exchange_id])} markets for {exchange_id}."
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to fetch markets for {exchange_id}: {e}. This exchange might be unavailable for trading this cycle."
                )

        if not self.active_clients_for_cycle:
            self.logger.error(
                "No exchange clients successfully loaded markets. Most operations will be skipped."
            )

    def _update_signals_on_interval(self):
        """
        Loops through each asset and checks if its position or trend signal
        needs to be re-calculated based on their respective time intervals.
        Returns a boolean indicating if any signals changed, and a list of reasons.
        """
        utcnow_timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
        signals_changed_this_cycle = False
        action_reasons = []

        for idx in self.portfolio_df.index:
            ticker = self.portfolio_df.loc[idx, "ticker"]
            asset_exchange_id = self.portfolio_df.loc[idx, "exchange"]

            if self.portfolio_df.loc[idx, "level"] == 0 or asset_exchange_id == "N/A":
                continue

            if asset_exchange_id not in self.active_clients_for_cycle:
                self.logger.warning(
                    f"Client or markets for exchange '{asset_exchange_id}' (asset: {ticker}) not available this cycle. Skipping signal checks."
                )
                continue
            
            # This helper now contains the core logic and returns if an update happened
            if self._check_and_update_asset_signals(idx, utcnow_timestamp):
                signals_changed_this_cycle = True
                action_reasons.append(f"Signal update for {ticker}")

        # Save any potential updates to the portfolio status file
        self._save_portfolio_status()
        self.logger.info("Portfolio status saved after signal interval checks.")
        return signals_changed_this_cycle, action_reasons

    def _check_and_update_asset_signals(self, idx, utcnow_timestamp):
        """
        For a single asset, determines if a new time interval ("bucket") has been
        reached for position or trend checks. If so, triggers the respective
        signal calculation (_check_position or _check_trend).
        Returns True if an update was performed, False otherwise.
        """
        ticker = self.portfolio_df.loc[idx, "ticker"]
        an_update_occurred = False

        # --- Position Signal Check ---
        p_interval = self.portfolio_df.loc[idx, "position_interval"]
        if p_interval > 0:
            last_pos_check_bucket = self.portfolio_df.loc[
                idx, "position_last_check_bucket"
            ]
            current_pos_check_bucket = utcnow_timestamp // p_interval

            if last_pos_check_bucket != current_pos_check_bucket:
                self.logger.info(
                    f"New position interval bucket ({int(current_pos_check_bucket)}) detected for {ticker}. "
                    f"Previous was ({int(last_pos_check_bucket)}). Updating position signal."
                )
                self.portfolio_df = self._check_position(
                    user_dataframe=self.portfolio_df, row_idx=idx
                )
                an_update_occurred = True
            else:
                self.logger.debug(
                    f"Position signal for {ticker} is up-to-date. Skipping check."
                )

        # --- Trend Signal Check ---
        t_interval = self.portfolio_df.loc[idx, "trend_interval"]
        if t_interval > 0:
            last_trend_check_bucket = self.portfolio_df.loc[
                idx, "trend_last_check_bucket"
            ]
            current_trend_check_bucket = utcnow_timestamp // t_interval

            if last_trend_check_bucket != current_trend_check_bucket:
                self.logger.info(
                    f"New trend interval bucket ({int(current_trend_check_bucket)}) detected for {ticker}. "
                    f"Previous was ({int(last_trend_check_bucket)}). Updating trend signal."
                )
                self.portfolio_df = self._check_trend(
                    user_dataframe=self.portfolio_df, row_idx=idx
                )
                an_update_occurred = True
            else:
                self.logger.debug(
                    f"Trend signal for {ticker} is up-to-date. Skipping check."
                )

        return an_update_occurred

    def _export_portfolio_data(self):
        if self.portfolio_export_path and not self.portfolio_df.empty:
            try:
                export_dir = Path(self.portfolio_export_path).parent
                if export_dir:
                    export_dir.mkdir(parents=True, exist_ok=True)
                self.portfolio_df.to_csv(self.portfolio_export_path, index=False)
                self.logger.info(
                    f"Portfolio data successfully exported to {self.portfolio_export_path}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to export portfolio data to CSV: {e}", exc_info=True
                )


# --- Main Execution Block ---
if __name__ == "__main__":
    paths_cfg = APP_CONFIG.get("paths", {})
    log_dir = paths_cfg.get("log_dir", "./portfolio/logfile/")
    portfolio_strategies_dir = paths_cfg.get("portfolio_strategies_dir")
    global_status_file_path = paths_cfg.get("global_status_file_path")
    log_filename = paths_cfg.get("log_filename", "trading_bot.log")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    current_log_file_path = Path(log_dir) / log_filename

    logging_cfg = APP_CONFIG.get("logging", {})
    log_level_str = logging_cfg.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = logging_cfg.get(
        "format",
        "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)-22s [%(filename)s:%(lineno)d] %(message)s",
    )
    log_date_format = logging_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_date_format,
        handlers=[
            logging.FileHandler(current_log_file_path, mode="a"),
            logging.StreamHandler(),
        ],
    )
    main_logger = logging.getLogger("MainBot")
    bot_logger = logging.getLogger("TradingBotCore")

    ext_lib_levels = logging_cfg.get("external_lib_levels", {})
    logging.getLogger("pycoingecko").setLevel(
        ext_lib_levels.get("pycoingecko", "WARNING").upper()
    )
    logging.getLogger("urllib3").setLevel(ext_lib_levels.get("urllib3", "INFO").upper())
    logging.getLogger("ccxt").setLevel(ext_lib_levels.get("ccxt", "INFO").upper())

    main_logger.info(f"Configuration loaded from {DEFAULT_CONFIG_FILE_PATH}")
    main_logger.info(f"Using log file: {current_log_file_path}")
    main_logger.info(f"Portfolio strategies directory: {portfolio_strategies_dir}")
    main_logger.info(f"Global status file: {global_status_file_path}")

    # --- CCXT Client Initialization (Multi-Exchange) ---
    initialized_clients = {}
    portfolio_assets_cfg = APP_CONFIG.get("portfolio_assets", [])
    exchange_configs_cfg = APP_CONFIG.get("exchange_configurations", {})
    credentials_cfg = APP_CONFIG.get("exchanges_credentials", {})

    # Determine unique exchange IDs needed from portfolio_assets
    unique_exchange_ids_in_portfolio = set()
    for asset in portfolio_assets_cfg:
        asset_exchange_id = asset.get("exchange")
        if asset_exchange_id and asset_exchange_id != "N/A":
            unique_exchange_ids_in_portfolio.add(asset_exchange_id)

    main_logger.info(
        f"Attempting to initialize clients for exchanges: {unique_exchange_ids_in_portfolio}"
    )

    for exchange_id in unique_exchange_ids_in_portfolio:
        if exchange_id not in exchange_configs_cfg:
            main_logger.error(
                f"Configuration for exchange '{exchange_id}' not found in 'exchange_configurations'. Client not initialized."
            )
            continue

        ex_config = exchange_configs_cfg[exchange_id]
        creds_key = ex_config.get("credentials_key")

        if not creds_key or creds_key not in credentials_cfg:
            main_logger.error(
                f"Credentials key '{creds_key}' for exchange '{exchange_id}' not found or invalid. Client not initialized."
            )
            continue

        creds = credentials_cfg[creds_key]

        if not creds.get("apiKey") or not creds.get("secret"):
            main_logger.error(
                f"apiKey or secret missing for credentials key '{creds_key}' (exchange: {exchange_id}). Client not initialized."
            )
            continue

        try:
            ccxt_params = {
                "apiKey": creds["apiKey"],
                "secret": creds["secret"],
                "enableRateLimit": ex_config.get("enable_rate_limit", True),
                "options": ex_config.get(
                    "options", {}
                ),  # Base options from exchange_configurations
            }
            # Merge options from credentials block if they exist (more specific)
            if "options" in creds and isinstance(creds["options"], dict):
                ccxt_params["options"].update(creds["options"])

            # Add password if present in credentials (e.g., for OKX)
            if creds.get("password"):
                ccxt_params["password"] = creds["password"]

            # Add defaultType if specified in ex_config, otherwise ccxt default applies
            if "default_type" in ex_config:
                ccxt_params["options"]["defaultType"] = ex_config["default_type"]

            client_class = getattr(ccxt, exchange_id)
            client = client_class(ccxt_params)

            # Test connection / load markets early (optional, TradingBot.run_cycle also does this)
            # client.load_markets() # Can be time-consuming here, moved to run_cycle

            initialized_clients[exchange_id] = client
            main_logger.info(
                f"CCXT client initialized for {exchange_id} using credentials key '{creds_key}'. Default type: {ccxt_params.get('options', {}).get('defaultType', 'N/A')}."
            )
            if (
                ccxt_params.get("options", {}).get("test", False)
                or "test" in client.urls
            ):  # Check if testnet
                main_logger.info(
                    f"CCXT client for {exchange_id} appears to be configured for TESTNET."
                )

        except AttributeError:
            main_logger.error(
                f"CCXT exchange ID '{exchange_id}' not found in ccxt library."
            )
        except ccxt.AuthenticationError as e:
            main_logger.error(
                f"CCXT Authentication Error for {exchange_id} using key '{creds_key}': {e}. Check API keys."
            )
        except Exception as e:
            main_logger.error(
                f"Failed to initialize CCXT client for {exchange_id}: {e}",
                exc_info=True,
            )

    if not initialized_clients:
        main_logger.warning(
            "No CCXT clients were successfully initialized. The bot might not be able to trade."
        )
    # --- End CCXT Client Initialization ---

    trading_bot_instance = TradingBot(
        app_config=APP_CONFIG,
        log_file_path=current_log_file_path,
        logger=bot_logger,
        exchange_clients=initialized_clients,  # MODIFIED
    )

    bot_settings_main_cfg = APP_CONFIG.get("bot_settings", {})
    cycle_sleep_seconds = bot_settings_main_cfg.get("cycle_sleep_duration_seconds", 60)

    main_logger.info("Trading bot starting main loop...")
    while True:
        try:
            trading_bot_instance.run_cycle()
            main_logger.info(
                f"Main loop cycle complete. Sleeping for {cycle_sleep_seconds} seconds."
            )
            time.sleep(cycle_sleep_seconds)
        except KeyboardInterrupt:
            main_logger.info("Bot manually interrupted. Exiting.")
            if (
                trading_bot_instance.telegram_bot_token
                and trading_bot_instance.telegram_chat_id
            ):
                asyncio.run(
                    send_info(
                        trading_bot_instance.log_file_path,
                        trading_bot_instance.telegram_bot_token,
                        trading_bot_instance.telegram_chat_id,
                        main_logger,
                    )
                )
                if Path(trading_bot_instance.global_status_file_path).exists():
                    asyncio.run(
                        send_info(
                            trading_bot_instance.global_status_file_path,
                            trading_bot_instance.telegram_bot_token,
                            trading_bot_instance.telegram_chat_id,
                            main_logger,
                        )
                    )
            break
        except Exception as main_loop_exception:
            main_logger.critical(
                f"Unhandled CRITICAL exception in main loop: {main_loop_exception}",
                exc_info=True,
            )
            if (
                trading_bot_instance.telegram_bot_token
                and trading_bot_instance.telegram_chat_id
            ):
                asyncio.run(
                    send_info(
                        trading_bot_instance.log_file_path,
                        trading_bot_instance.telegram_bot_token,
                        trading_bot_instance.telegram_chat_id,
                        main_logger,
                    )
                )
                if Path(trading_bot_instance.global_status_file_path).exists():
                    asyncio.run(
                        send_info(
                            trading_bot_instance.global_status_file_path,
                            trading_bot_instance.telegram_bot_token,
                            trading_bot_instance.telegram_chat_id,
                            main_logger,
                        )
                    )
            main_logger.info(
                f"Attempting to recover by sleeping for {cycle_sleep_seconds} seconds before retrying loop."
            )
            time.sleep(cycle_sleep_seconds)
