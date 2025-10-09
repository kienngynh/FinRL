# Prompt

I understand you want to maintain the SAC-based portfolio allocation in `test.py` but integrate the Keltner Channel indicator logic from `src/trading_bot.py` and potentially use hyperoptimized parameters from the `run_hyperopt.py` process.

Here is my proposed plan to achieve this:

**1. Copy Keltner Channel Classes:**
I will copy the `IndicatorMixin`, `AverageTrueRange`, and `KeltnerChannel` classes from `src/trading_bot.py` directly into `test.py`. This will make the Keltner Channel calculation logic available for use. I will also add necessary imports like `ABC` and `abstractmethod` from the `abc` module.

**2. Create a Custom `FeatureEngineer` in `test.py`:**
I will define a new `FeatureEngineer` class within `test.py` that inherits from `finrl.meta.preprocessor.preprocessors.FeatureEngineer`. This custom class will allow us to extend its functionality without modifying the FinRL library files.

**3. Override `add_technical_indicator` Method:**
Inside the custom `FeatureEngineer`, I will override the `add_technical_indicator` method. This method will:
    * First, calculate the existing technical indicators using `stockstats` as it currently does.
    * Then, for each unique ticker in the DataFrame, it will calculate the Keltner Channel bands (`kc_mband`, `kc_hband`, `kc_lband`), Channel Width (`kc_wband`), and Percentage Bandwidth (`kc_pband`) using the copied `KeltnerChannel` class.
    * These new Keltner Channel features will be added as columns to the DataFrame.

**4. Integrate Hyperoptimized Parameters (Optional):**
If you have run `run_hyperopt.py` and generated strategy JSON files (e.g., `portfolio/strategies/BTC_strategy.json`), I can add logic to `test.py` to:
    *Load these JSON files for each ticker.
    *   Extract the optimized Keltner Channel parameters (e.g., `position_kc_window`, `position_kc_atr`, `position_kc_mult`).
    *   Pass these optimized parameters to the `KeltnerChannel` constructor when calculating the indicators for the respective tickers. If no optimized parameters are found, default values will be used.

**5. Update `tech_indicator_list`:**
The `tech_indicator_list` attribute of the custom `FeatureEngineer` will be updated to include the new Keltner Channel indicators, ensuring they are recognized and used as features for the DRL agent.

This plan ensures that the FinRL DRL agent framework remains, while enhancing its feature set with the Keltner Channel indicator and its optimized parameters.

Does this plan align with your expectations? If so, please let me know, and I will ask you to switch to ACT MODE to begin the implementation.

# Trading Bot Logic Documentation

This document explains the core logic of the `trading_bot.py` script, covering configuration, technical indicators, portfolio management, rebalancing strategies, and trade execution.

## 1. Configuration Loading and Validation

The bot starts by loading its configuration from a JSON file, specified by `DEFAULT_CONFIG_FILE_PATH` (defaulting to `./portfolio/configs/portfolio_config.json`).

### `load_app_config(config_path)`

This function is responsible for:

- Reading the JSON configuration file.
- Validating the structure and content of the configuration using several helper functions.
- Exiting the application if any critical configuration errors are found.

### Validation Functions

- `_validate_required_sections(config_data, config_path)`: Ensures all essential top-level sections (e.g., `bot_settings`, `paths`, `exchange_configurations`, `portfolio_assets`) are present.
- `_validate_exchange_configurations(config_data)`: Checks that each defined exchange configuration has a `credentials_key` and that this key exists in the `exchanges_credentials` section.
- `_validate_paths(config_data)`: Verifies that `portfolio_strategies_dir` and `global_status_file_path` are defined within the `paths` section.
- `_validate_portfolio_assets(config_data)`: Ensures the `portfolio_assets` list is not empty and each asset entry contains `ticker`, `level`, `strategy_file`, and `strategy_name`. It also checks that tradable assets (level != 0) have an `exchange` key.

## 2. Retry Decorator

The `retry_on_network_error` decorator provides a robust mechanism to handle transient network-related errors when interacting with external APIs (exchanges, CoinGecko).

### `retry_on_network_error`

- **Purpose**: Automatically retries a function call if specific network-related exceptions occur.
- **Configuration**: Uses `MAX_RETRIES_CFG` (default: 3) and `RETRY_DELAY_SECONDS_CFG` (default: 5) from `bot_settings` in the configuration.
- **Mechanism**:
  - It detects if the decorated function is `async` or `sync` and applies the appropriate retry logic (`_retry_logic_async` or `_retry_logic_sync`).
  - Catches `ccxt.NetworkError`, `ccxt.RequestTimeout`, `ccxt.ExchangeNotAvailable`, `ccxt.OnMaintenance`, and `ConnectionError`.
  - Logs warnings for each retry attempt.
  - If all retries fail, `_handle_retry_failure` is called.
- **`_handle_retry_failure`**:
  - Logs an error message.
  - Provides specific fallback behavior for `fetch_ohlcv` (returns an empty DataFrame) and `fetch_market_cap` (returns last known market cap and fetch time).
  - Re-raises the last exception if no specific fallback is defined or if fallback data is unavailable.

## 3. IndicatorMixin and Keltner Channel

This section defines the base for technical indicators and implements the Keltner Channel.

### `IndicatorMixin`

An abstract base class providing common utility methods for technical indicators:

- `_check_fillna(serie, value=0)`: Fills NaN values in a pandas Series if `_fillna` is enabled.
- `_true_range(high, low, close)`: Calculates the True Range (TR) for OHLCV data.
  - `TR = max(High - Low, abs(High - PreviousClose), abs(Low - PreviousClose))`
- `_crossed_above(series1, series2)`: Returns a boolean Series indicating when `series1` crosses above `series2`.
- `_crossed_below(series1, series2)`: Returns a boolean Series indicating when `series1` crosses below `series2`.

### `AverageTrueRange`

Calculates the Average True Range (ATR), a measure of market volatility.

- **Initialization**: Takes `high`, `low`, `close` Series and a `window` (default: 14).
- **Calculation (`_run`)**:
    1. Calculates `true_range` using `_true_range`.
    2. Computes the Exponentially Weighted Moving Average (EWMA) of the `true_range`.
        - `ATR = EMA(True Range, window)`
        - `alpha = 1 / window` (used in `ewm` for EMA calculation)

### `KeltnerChannel`

Calculates the Keltner Channel, a volatility-based envelope around a moving average.

- **Initialization**: Takes `open`, `high`, `low`, `close` Series, `window` (for EMA, default: 20), `window_atr` (for ATR, default: 10), and `window_mult` (multiplier for ATR, default: 2).
- **Calculation (`_run`)**:
    1. **Typical Price (TP)**: `TP = (Open + High + Low + Close) / 4.0`
    2. **Middle Band (KC_MBAND)**: `KC_MBAND = EMA(Typical Price, window)`
    3. **Average True Range (ATR)**: Calculated using `AverageTrueRange` with `window_atr`.
    4. **Upper Band (KC_HBAND)**: `KC_HBAND = KC_MBAND + (window_mult * ATR)`
    5. **Lower Band (KC_LBAND)**: `KC_LBAND = KC_MBAND - (window_mult * ATR)`
- **Additional Indicators**:
  - `keltner_channel_wband()`: Channel Width. `KC_WBAND = ((KC_HBAND - KC_LBAND) / KC_MBAND) * 100`
  - `keltner_channel_pband()`: Percentage Bandwidth. `KC_PBAND = (Close - KC_LBAND) / (KC_HBAND - KC_LBAND)`
  - `keltner_channel_close_hband_indicator()`: `True` if `Close` crosses above `KC_HBAND`.
  - `keltner_channel_high_hband_indicator()`: `True` if `High` crosses above `KC_HBAND`.
  - `keltner_channel_close_lband_indicator()`: `True` if `Close` crosses below `KC_LBAND`.
  - `keltner_channel_low_lband_indicator()`: `True` if `Low` crosses below `KC_LBAND`.

## 4. Modular Indicator Strategy Framework

This framework allows for easy integration of new trading strategies.

### `IndicatorStrategy` (Abstract Base Class)

Defines the interface for any trading strategy:

- `get_strategy_params()`: Returns a list of parameter names used by the strategy.
- `get_default_config(is_stablecoin)`: Returns default configuration parameters, potentially varying for stablecoins.
- `get_param_data_types()`: Maps parameter names to their data types.
- `prepare_signal_params(row_data, param_prefix)`: Extracts and prepares parameters from a portfolio row for signal calculation.
- `calculate_signals(df, **params)`: Abstract method to calculate trading signals based on OHLCV data and parameters.

### `KeltnerChannelStrategy`

An implementation of `IndicatorStrategy` for the Keltner Channel.

- **`get_strategy_params()`**: Returns parameters like `position_kc_window`, `trend_kc_mult`, etc.
- **`get_default_config(is_stablecoin)`**: Provides default values for Keltner Channel parameters, with specific defaults for stablecoins (all zeros, effectively disabling the indicator).
- **`get_param_data_types()`**: Defines types for its parameters (e.g., `int`, `float`).
- **`prepare_signal_params(row_data, param_prefix)`**: Extracts `window`, `window_atr`, `window_mult` from the `row_data` using the provided `param_prefix` (e.g., "position" or "trend").
- **`calculate_signals(self, df, **params)`**:
    1. Initializes a `KeltnerChannel` object with the provided OHLCV `df` and parameters.
    2. Calculates `buy_crossupper` (close crosses above upper band) and `sell_crosslower` (close crosses below lower band) indicators.
    3. Combines these into a `position` signal:
        - `1.0` for buy signals (`buy_crossupper`).
        - `-1.0` for sell signals (`sell_crosslower`).
        - `0.0` for neutral (no signal, or filled forward from previous non-neutral signal).
        - `df["position"] = df["position"].replace(0, np.nan).ffill().fillna(0)`: This line propagates the last non-neutral signal forward, treating `0` as a placeholder for "no new signal, maintain previous".

### `IndicatorFactory`

A factory class to manage and retrieve `IndicatorStrategy` instances.

- `_strategies`: A dictionary mapping strategy names (e.g., "keltner_channel") to their respective strategy classes.
- `get_strategy_instance(name)`: Returns an instance of the specified strategy.
- `get_strategy_class(name)`: Returns the class type of the specified strategy.
- **Extensibility**: New strategies can be added by creating a new class inheriting from `IndicatorStrategy` and registering it in the `_strategies` dictionary.

## 5. Helper Functions

### `standardized_amount(amount_float, precision_step, min_amount_limit)`

- **Purpose**: Adjusts a float amount to match exchange precision requirements and minimum limits.
- **Logic**:
    1. Converts `amount_float` to `Decimal` for precise calculations.
    2. If `precision_step` is provided and non-zero, it quantizes the amount to the nearest multiple of `precision_step`.
        - `standardized_val = (amount_decimal // quantizer) * quantizer`
    3. Ensures the `standardized_val` is not less than `min_amount_limit`. If it is, returns `Decimal("0")`.

### `send_info(path, bot_token, chat_id, logger)`

- **Purpose**: Sends a file as a document to a Telegram chat.
- **Mechanism**: Uses the `python-telegram-bot` library. Checks if the file exists before sending. Logs success or failure.

## 6. TradingBot Class Initialization

The `TradingBot` class encapsulates the entire bot's logic.

### `__init__(self, app_config, log_file_path, logger, exchange_clients)`

The constructor orchestrates the bot's setup:

- Calls several `_initialize_*` methods to set up attributes, paths, external clients (Telegram, CoinGecko), and log the operating mode (dry run vs. live).
- Initiates the `_run_initialization_sequence()`.

### Initialization Methods

- `_initialize_basic_attributes()`: Sets up core bot attributes from `app_config`, including `indicator_history_size`, `min_order_value_usd`, `rebalance_threshold_percentage`, `signal_confirmation_cycles`, and various balance adjustment factors. The `dry_run` setting is assumed to be `False` for live mode.
- `_initialize_paths()`: Configures paths for strategies, global status, portfolio export, and OHLCV export, creating directories if they don't exist.
- `_initialize_telegram()`: Sets up Telegram bot token and chat ID if enabled in config.
- `_initialize_coingecko()`: Initializes the `CoinGeckoAPI` client.
- `_log_initialization_mode()`: Logs that the bot is running in "LIVE TRADING" mode.
- `_run_initialization_sequence()`:
    1. Loads portfolio data (`_load_portfolio_data`).
    2. Identifies the primary stablecoin (`_identify_primary_stablecoin`).
    3. Performs initial signal checks for all active assets (`_perform_initial_signal_checks`).
    4. Loads exchange market data (`_load_exchange_markets`).
    5. Performs initial portfolio distribution (`_perform_initial_portfolio_distribution`).
    6. Saves the initial portfolio status (`_save_portfolio_status`).

## 7. Portfolio Data Management

The bot manages its portfolio using a pandas DataFrame (`self.portfolio_df`) and JSON files for persistence.

### `_load_portfolio_data()`

- Loads asset configurations from `app_config["portfolio_assets"]`.
- Loads global status data from `self.global_status_file_path`.
- For each asset:
  - Loads static data from `asset_config`.
  - Loads strategy-specific parameters from a JSON file in `portfolio_strategies_dir`. If not found, it creates a default strategy config and saves it.
  - Loads asset-specific status data from `global_status_data`. If not found, it creates default status data using `_get_default_status_data`.
- Combines all data into `self.portfolio_df`.
- Ensures all `CORE_STATIC_COLUMNS`, `CORE_STATUS_COLUMNS`, and strategy-specific columns are present and applies correct data types using `_apply_column_dtype`.

### `_get_default_status_data(ticker, level)`

- Provides a dictionary of default status values for a new asset.
- For stablecoins (level 0), it sets `price` to 1.0, `free` and `value_in_stable` to `initial_stablecoin_balance`, and `ratio` and `target_value` to 1.0 and `initial_stablecoin_balance` respectively.

### `_save_portfolio_status()`

- Iterates through `self.portfolio_df` and extracts data for `STATUS_COLUMNS` for each asset.
- Saves this aggregated status data to `self.global_status_file_path` as a JSON file.
- Includes a `convert_numpy_types` helper to ensure pandas/numpy data types are correctly serialized to JSON.

### `_load_json_file(file_path, is_global_status=False)` and `_save_json_file(file_path, data)`

- Utility functions for reading and writing JSON files, with error handling and directory creation.
- `_save_json_file` includes a check to prevent saving to a directory instead of a file.

## 8. OHLCV and Market Cap Fetching

The bot retrieves market data from exchanges and CoinGecko.

### `_get_ccxt_timeframe(interval_seconds)`

- Maps a given interval in seconds to a CCXT-compatible timeframe string (e.g., 3600s -> "1h").

### `_fetch_ohlcv_ccxt(client, symbol_pair, timeframe, n_bars)`

- **Purpose**: Fetches historical Open, High, Low, Close, Volume (OHLCV) data for a given symbol pair and timeframe from an exchange.
- **Caching**: Uses `self.cycle_ohlcv_cache` to store OHLCV data for the current cycle, avoiding redundant API calls.
- **Pagination**: Handles fetching more bars than an exchange's `fetchOHLCVLimit` by making multiple requests, adjusting the `since` parameter.
- **Data Processing**: Converts raw OHLCV data into a pandas DataFrame, sets `datetime` as index, converts columns to numeric types, and handles duplicates/NaT values.
- **Retry**: Decorated with `@retry_on_network_error`.

### `_fetch_market_cap(coin_gecko_id, ticker, current_row_data)`

- **Purpose**: Fetches the current market capitalization for an asset using CoinGecko.
- **Fallback**: If CoinGecko fails or returns no data, it attempts to use the `last_market_cap` and `last_market_cap_fetch_time` from `current_row_data` if it's not stale (within `market_cap_stale_threshold_seconds`).
- **Retry**: Decorated with `@retry_on_network_error`.

## 9. Signal Calculation Logic

The bot calculates "position" and "trend" signals for each asset based on configured indicators.

### `_check_position(user_dataframe, row_idx)`

- **Purpose**: Calculates the short-term "position" signal for a specific asset.
- **Process**:
    1. Retrieves asset details (ticker, exchange, strategy name, position interval).
    2. Fetches OHLCV data using `_fetch_ohlcv_ccxt` for the asset's `position_interval`. It iteratively fetches more bars (up to `max_indicator_history_size`) until a non-neutral signal is found or history limit is reached.
    3. Uses `IndicatorFactory` to get the appropriate strategy instance and calculates signals using `calculate_signals`.
    4. Updates the asset's `price`, `position_signal_prev_1`, `position_signal_prev_2` in `portfolio_df`.
    5. Exports OHLCV data to CSV if `ohlcv_export_dir` is configured.
    6. Updates `position_last_check_bucket` to track when the last check occurred.
    7. Calls `_validate_trend_position_signals` for consistency.

### `_check_trend(user_dataframe, row_idx)`

- **Purpose**: Calculates the long-term "trend" signal for a specific asset, primarily based on market cap dominance.
- **Process**:
    1. Calls `_check_trend_price_component` to calculate trend signals based on historical dominance.
    2. Calls `_check_trend_market_cap` to update the asset's market cap from CoinGecko.
    3. Updates `trend_last_check_bucket`.
    4. Calls `_validate_trend_position_signals` for consistency.

### `_calculate_historical_dominance_ohlcv(target_ticker, timeframe, n_bars)`

- **Purpose**: Generates synthetic OHLCV data representing an asset's market cap dominanc>e over the portfolio.
- **Formulas**:
    1. **Fetch all OHLCV**: Fetches historical price OHLCV for all tradable assets in the portfolio.
    2. **Estimate Historical Market Cap**: For each asset, estimates historical market cap based on its current market cap and historical price changes.
        - `MC_hist = (last_market_cap / current_price) * price_hist`
    3. **Calculate Total Historical Market Cap**: Sums the estimated historical market caps of all assets for each OHLCV candle.
    4. **Calculate Dominance OHLCV**:
        - `Dominance_Open = (Target_Asset_MC_Open / Total_MC_Open) * 100`
        - Similar formulas for `High`, `Low`, `Close`.
    5. Adds volume from the target asset's original price chart.

### `_check_trend_price_component(...)`

- **Purpose**: Uses the dominance OHLCV data to calculate the trend signal.
- **Process**: Similar to `_check_position`, but uses `_calculate_historical_dominance_ohlcv` to get the OHLCV data and then applies the configured strategy (e.g., Keltner Channel) to this dominance data.
- Updates `trend_signal_prev_1` and `trend_signal_prev_2`.

### `_validate_trend_position_signals(portfolio_df, row_idx)`

- **Purpose**: Ensures consistency between position and trend signals.
- **Logic**: If either `position_signal_prev_1` or `trend_signal_prev_1` is `0` (neutral), both are conservatively set to `-1.0` (interpreted as a sell/avoid signal). This prevents the bot from taking a position if one of the signals is ambiguous or missing.

## 10. Portfolio Rebalancing Logic and Formulas

The core rebalancing logic is encapsulated in `_distribute_portfolio`.

### `_distribute_portfolio()`

- Orchestrates the entire rebalancing process:
    1. `_is_distribution_possible()`: Performs sanity checks (non-empty portfolio, active clients, stablecoin identified, markets loaded).
    2. `_initialize_balances()`: Fetches current balances from exchanges (live mode) or uses `initial_stablecoin_balance` (dry run).
    3. `_update_stablecoin_total_in_df()`: Updates the stablecoin's balance in `portfolio_df`.
    4. `_update_asset_prices()`: Fetches live prices for all assets.
    5. `_calculate_total_portfolio_value()`: Calculates the total portfolio value in stablecoin.
    6. `_calculate_allocation_targets()`: Determines the target allocation for each asset.
    7. `_generate_optimized_trade_proposals()`: Creates a list of trades needed to reach targets.
    8. `_execute_trade_proposals()`: Executes the generated trades.
    9. `_finalize_portfolio_state()`: Refreshes balances and recalculates values after trades.

### `_initialize_balances()`

- Iterates through `exchange_clients`, fetches `fetch_balance()`, and updates `free` and `used` balances for all assets on that exchange in `portfolio_df`. It also aggregates the total free stablecoin balance across all exchanges.

### `_update_asset_prices()`

- Iterates through non-stablecoin assets.
- If an asset's `price` is missing or zero, it fetches the latest ticker price from the respective exchange using `client.fetch_ticker()` and updates `portfolio_df`.

### `_calculate_total_portfolio_value()`

- **Formula**: `value_in_stable = free_balance * current_price`
- Calculates `value_in_stable` for each asset and sums them to get `total_portfolio_value`.
- If `total_portfolio_value` is effectively zero, it sets all `ratio`, `target_value`, `change` to zero (except stablecoin ratio to 1.0) and returns `None`.

### `_calculate_allocation_targets(total_portfolio_value)`

This is the core rebalancing logic, determining how the portfolio should be distributed.

- **Dominance Calculation**:
  - `tradable_assets = all assets with level != 0`
  - `total_mc = sum of last_market_cap for tradable_assets`
  - `dominance = last_market_cap / total_mc` (for tradable assets)
  - Stablecoin dominance is set to 0.0.
- **Two-Tiered Signal Strategy**:
    1. **Trend Filter**: Only assets with a positive long-term `trend_signal_prev_1 == 1` are considered for allocation.
    2. **Ratio Allocation**: For trending assets, their `ratio` (strategic allocation slot) is calculated based on their dominance relative to the sum of all trending assets' dominances:
        - `ratio_trending_asset = trending_asset_dominance / sum_of_trending_dominances`
    3. **Position Filter**: For each trending asset, if its short-term `position_signal_prev_1 != 1` (i.e., not a buy signal), its calculated `ratio` is moved to the primary stablecoin.
    4. **No Trending Assets Fallback**: If no assets are trending, 100% of the portfolio `ratio` is allocated to the primary stablecoin.
- **Altcoin Cap**: If `max_alt_coin_ratio` is set (less than 1.0):
  - It checks if the sum of `ratio` for altcoins (level >= 2) exceeds `max_alt_coin_ratio`.
  - If it does, altcoin ratios are scaled down proportionally, and the excess `ratio` is transferred to the stablecoin.
- **Ratio Normalization**: All `ratio` values are normalized to ensure their sum is exactly 1.0, correcting for floating-point inaccuracies.
- **Target Value Calculation**:
  - `target_value = ratio * total_portfolio_value`
- **Change Calculation**:
  - `change = target_value - value_in_stable` (positive for buy, negative for sell)
- **Rebalance Threshold**: Trades are skipped if the absolute `change` relative to `target_value` is below `rebalance_threshold_percentage`.
  - `abs(change) / target_value < rebalance_threshold_percentage`

## 11. Trade Execution

The bot generates and executes trade proposals to adjust the portfolio according to the calculated targets.

### `_generate_optimized_trade_proposals()`

- **Purpose**: Creates a list of trade proposals (buy/sell orders) to bring the portfolio closer to its target allocation.
- **Filtering**:
  - `buys_df`: Assets with `change > min_order_value_usd` AND `position_signal_prev_1 == 1` AND `trend_signal_prev_1 == 1`.
  - `sells_df`: Assets with `change < -min_order_value_usd` OR (`position_signal_prev_1 == -1` AND `value_in_stable > min_order_value_usd`). For full sells (position signal -1), `change` is set to `-value_in_stable`.
- **Direct Pair Matching Logic**:
  - Attempts to find direct asset-to-asset trades (e.g., BTC/ETH) on the same exchange, excluding stablecoin pairs.
  - If `buy_ticker/sell_ticker` or `sell_ticker/buy_ticker` market exists, a direct trade proposal is created.
  - `trade_value = min(abs(sell_row["change"]), buy_row["change"])`
  - `price = buy_row["price"] / sell_row["price"]` (for buy `buy_ticker/sell_ticker`) or `sell_row["price"] / buy_row["price"]` (for sell `sell_ticker/buy_ticker`)
  - `trade_amount = trade_value / base_currency_price_in_stable`
  - Updates `change` in `buys_df` and `sells_df` to reflect the matched trade.
- **Stablecoin Fallback Logic**:
  - For any remaining `buys_df` or `sells_df` entries, it generates trade proposals against the `primary_stable_coin_ticker` (e.g., BTC/USDT).
  - `amount_coin = value_stable / current_price`
  - Includes a safety check for sell orders to ensure `amount_coin` does not exceed available `free` balance, adjusting if necessary.

### `_execute_trade_proposals(trade_proposals)`

- Sorts trade proposals: direct trades first, then stablecoin sells, then stablecoin buys. This prioritizes reducing exposure and direct conversions.
- Iterates through sorted proposals and calls `_execute_single_trade` for each.
- Adds a small `time.sleep(1)` between live trades to avoid rate limits.

### `_execute_single_trade(proposal)`

- **Pre-checks**:
  - Ensures a CCXT client is available for the exchange.
  - Validates `price` and `amount` as `Decimal` values.
  - Checks if the `current_trade_value_usd` is above `min_order_value_usd` (bot's minimum) and `min_cost` (exchange's minimum).
- **Execution**:
  - Calls `standardized_amount` to ensure the amount adheres to exchange precision and limits.
  - Executes a market order using `client.create_market_order()`.
- **Post-trade Update**: Calls `_update_portfolio_post_trade` with the order result and original proposal.
- **Error Handling**: Catches `ccxt.InsufficientFunds`, `ccxt.InvalidOrder`, `ccxt.NetworkError`, `ccxt.ExchangeError`, and other exceptions during order execution.

### `_adjust_amount_for_balance(...)` (Currently not directly used in `_execute_single_trade` but available)

- **Purpose**: Fetches fresh balances and ticker prices to dynamically adjust trade amounts if available funds are insufficient.
- **Logic**:
  - For buys: If `required_quote > available_quote`, adjusts `amount = (available_quote / live_price) * balance_adjustment_factor_buy`.
  - For sells: If `amount > available_base`, adjusts `amount = available_base * balance_adjustment_factor_sell`.
  - Re-standardizes the adjusted amount.

### `_update_portfolio_post_trade(order_result, proposal)`

- **Purpose**: Updates `portfolio_df` and internal balance estimates after a trade.
- If `order_result` is incomplete, it skips local updates, relying on a full balance fetch later.

### `_finalize_portfolio_state()`

- Re-fetches all balances from exchanges to get the true, post-trade state. Updates `portfolio_df` and `self.stablecoin_balance_by_exchange`.
- Recalculates `value_in_stable` for all assets based on the final balances.

## 12. Main Execution Loop

The `if __name__ == "__main__":` block sets up logging, initializes exchange clients, and runs the main bot loop.

### Initialization Block

- **Logging Setup**: Configures logging to both a file (`trading_bot.log`) and console, with configurable log levels and formats. Sets specific log levels for external libraries (pycoingecko, urllib3, ccxt).
- **CCXT Client Initialization**:
  - Iterates through `portfolio_assets` to identify unique exchange IDs.
  - For each unique exchange, it retrieves configuration and credentials from `app_config`.
  - Initializes the appropriate `ccxt` client (e.g., `ccxt.binance`, `ccxt.bybit`) with API keys, secret, and other options (rate limiting, default type, password).
  - Logs initialization status, including warnings for testnet configurations or failed client setups.
- **TradingBot Instance Creation**: Creates an instance of `TradingBot` with the loaded configuration, log paths, and initialized CCXT clients.

### `run_cycle()`

- This method represents a single operational cycle of the bot.
- **Cycle Start**: Logs the start time and clears the `cycle_ohlcv_cache`.
- **Portfolio Data Load**: Reloads portfolio data (`_load_portfolio_data`) to ensure it's up-to-date. Handles critical errors by skipping the cycle.
- **Cycle Initialization**: Calls `_initialize_cycle()` to identify the primary stablecoin and load exchange markets for the current cycle.
- **Action Trigger Logic**:
    1. **Balance Change Check**: Fetches `current_total_stablecoin_balance`. If `self.last_total_stablecoin_balance` is `None` (first run) or if the absolute difference between current and last balance exceeds `min_order_value_usd`, `balance_changed` is set to `True`.
    2. **Signal Update Check**: Calls `_update_signals_on_interval()` to check if any asset's position or trend signals need recalculation based on their configured intervals.
  - If `action_needed_overall` (from signal changes) or `balance_changed` is `True`, the bot proceeds with portfolio distribution.
- **Portfolio Distribution**: If conditions are met and clients/markets are ready, it calls `_distribute_portfolio()`, sets the `done` flag for tradable assets, and saves the portfolio status. It then updates `self.last_total_stablecoin_balance`.
- **Export Data**: Exports the current `portfolio_df` to a CSV file if `portfolio_export_path` is configured.
- **Telegram Notifications**: If Telegram is configured, it sends the log file and global status file to the specified chat.
- **Cycle End**: Logs the cycle completion time.

### `_initialize_cycle()`

- Identifies the `primary_stable_coin_ticker`.
- Reloads `markets_by_exchange` and `active_clients_for_cycle` by fetching markets for all configured exchanges. This ensures fresh market data and active client instances for the current cycle.

### `_update_signals_on_interval()`

- Iterates through all tradable assets in `portfolio_df`.
- For each asset, it calls `_check_and_update_asset_signals()` to determine if its position or trend signal needs updating based on time intervals.
- Returns `True` if any signals were updated, along with a list of reasons.

### `_check_and_update_asset_signals(idx, utcnow_timestamp)`

- Checks if the current timestamp has crossed a new "bucket" for either the `position_interval` or `trend_interval` compared to `position_last_check_bucket` or `trend_last_check_bucket`.
- If a new bucket is detected, it calls `_check_position()` or `_check_trend()` respectively to recalculate the signals.
- Returns `True` if any signal was updated.

### Main Loop (`while True`)

- Continuously calls `trading_bot_instance.run_cycle()`.
- Sleeps for `cycle_sleep_seconds` (default: 60) between cycles.
- Includes robust error handling for `KeyboardInterrupt` (graceful exit) and other `Exception` types (logs critical error, attempts to recover by sleeping and retrying).
- Sends log and status files to Telegram on exit or critical error.

# Hyperoptimization Process for Strategy Parameters

This document explains the methodology used to find the optimal parameters for the trading strategies employed by `trading_bot.py`. The process uses the powerful hyperoptimization features of the **Freqtrade** framework, automated by the `run_hyperopt.py` script.

The goal is to find a set of parameters for the Keltner Channel indicator that is specifically tuned for two different purposes:

1. **Position Signal**: A short-term signal based on the asset's price action.
2. **Trend Signal**: A long-term signal based on the asset's market cap dominance relative to other assets in the portfolio.

## 1. Overview of the Automated Hyperoptimization Process

The `run_hyperopt.py` script serves as the master controller for the entire optimization process. For each tradable asset defined in `portfolio_config.json`, it performs the following steps:

1. **Isolate Optimization**: It runs two completely separate hyperoptimization tasks for each asset:
    - One for the `KeltnerChannelStrategyPosition` strategy on its specified `position_interval` timeframe.
    - One for the `KeltnerChannelStrategyTrend` strategy on its specified `trend_interval` timeframe.
2. **Dynamic Configuration**: It creates temporary Freqtrade configuration files for each task to ensure that the correct strategy, pair, and timeframe are used, preventing conflicts between concurrent processes.
3. **Data Download**: It ensures the historical market data (OHLCV) required for the backtest is downloaded and validated.
4. **Execute Freqtrade**: It calls the `freqtrade hyperopt` command-line interface to run the optimization. Freqtrade then uses the specified strategy and loss function to test thousands of parameter combinations over the historical data.
5. **Result Extraction**: After Freqtrade completes, the script extracts the best-performing parameters from the strategy's JSON output file.
6. **Combine and Save**: The optimal parameters from both the "position" and "trend" runs are combined into a single strategy file (e.g., `BTC_strategy.json`) in the `portfolio/strategies/` directory. This file is then read by the main `trading_bot.py` during live operations.

## 2. The Trading Strategies Explained

Two distinct Freqtrade strategies are used to find the parameters for the two signals.

### 2.1 `KeltnerChannelStrategyPosition.py` (Position Signal)

This strategy is straightforward and focuses on pure price action to generate short-term entry and exit signals.

- **Purpose**: To identify potential entry/exit points when the price breaks out of a volatility-based channel.
- **Signal Logic**:
  - **Buy Signal (`enter_long`)**: A buy signal is generated when the asset's closing price crosses *above* the upper band of the Keltner Channel.
  - **Sell Signal (`exit_long`)**: A sell signal is generated when the asset's closing price crosses *below* the lower band of the Keltner Channel.
- **Optimizable Parameters**: Freqtrade will test different combinations of the following parameters to find the most profitable setup:
  - `kc_window`: The lookback period for the Exponential Moving Average (EMA) that forms the middle line of the channel.
  - `kc_mult`: The multiplier for the Average True Range (ATR) that determines how wide the channel is.
  - `kc_atrs`: The lookback period for the ATR calculation, which measures volatility.

### 2.2 `KeltnerChannelStrategyTrend.py` (Trend Signal)

This strategy is more complex as its goal is to replicate the "market cap dominance" logic from `trading_bot.py` within the Freqtrade backtesting environment.

- **Purpose**: To determine the long-term trend of an asset not by its price, but by its strength relative to the entire defined portfolio. A rising dominance suggests the asset is outperforming its peers.
- **Dominance Calculation Logic**:
    1. **Load Portfolio Context**: The strategy first loads `portfolio_config.json` and the global status file to get the list of all tradable assets and their last known market caps.
    2. **Fetch All Price Data**: Using Freqtrade's `DataProvider`, it fetches the historical OHLCV data for *every tradable asset* in the portfolio for the specified timeframe.
    3. **Estimate Historical Market Cap**: For each asset, it calculates an estimated historical market cap time series. The formula used is:
        `Historical MC = (Last Known MC / Last Known Price) * Historical Price Series`
    4. **Calculate Total Portfolio MC**: It sums the estimated historical market caps of all assets at each candle to create a time series of the total portfolio market cap.
    5. **Generate Dominance OHLCV**: It creates a new, synthetic OHLCV dataframe where the "price" (open, high, low, close) is the target asset's dominance percentage. The formula is:
        `Dominance "Close" = (Target Asset's Historical MC "Close" / Total Portfolio MC "Close") * 100`
- **Signal Logic**:
  - The strategy applies the exact same Keltner Channel logic as the Position strategy, but it uses the **synthetic dominance OHLCV data** instead of the actual price data.
  - **Buy Signal (`enter_long`)**: Generated when the asset's *dominance* crosses above the upper Keltner Channel band. This indicates the asset is becoming significantly stronger relative to its peers.
  - **Sell Signal (`exit_long`)**: Generated when the asset's *dominance* crosses below the lower Keltner Channel band. This indicates the asset is becoming significantly weaker.
- **Optimizable Parameters**: The same parameters (`kc_window`, `kc_mult`, `kc_atrs`) are optimized, but their optimal values will be different because they are being applied to the dominance data, which has a different scale and volatility profile than price data.

## 3. The Hyperopt Loss Function (`SortinoHyperOptLossDaily.py`)

The loss function is the most critical piece of the hyperoptimization puzzle. It evaluates the backtest result of each parameter combination and returns a single numerical "score" (the "loss"). Freqtrade's goal is to **minimize this score**. Our custom loss function, `SortinoHyperOptLossDaily`, is designed to find a strategy that is not just profitable, but also frequent and safe.

It balances three competing objectives:

### Objective 1: Maximize Performance (Sortino Ratio)

- **What it is**: The Sortino Ratio is a modification of the popular Sharpe Ratio. While the Sharpe Ratio penalizes all volatility (both up and down), the Sortino Ratio only penalizes **downside volatility**. This is superior for evaluating trading strategies, because upward price swings (upside volatility) are desirable.
- **How it works**: It measures the risk-adjusted return of the strategy, giving a higher score to strategies that produce high returns with low downside risk.

### Objective 2: Ensure Sufficient Frequency (Trade Count Penalty)

- **The Problem**: A backtest might show incredible profits from just one or two lucky trades over many years. This is not a statistically reliable or practical strategy. We need a strategy that trades frequently enough to be dependable.
- **How it works**:
    1. A baseline is set: `BASELINE_TRADES_PER_YEAR` (e.g., 365 trades) for a `BASELINE_TIMEFRAME` (e.g., '15m').
    2. This baseline is automatically scaled to the timeframe being tested. For example, a '4h' strategy is expected to trade far less frequently than a '15m' one. The scaling factor corrects for this.
    3. A `trade_penalty` is calculated: `penalty = actual_trades / target_trades`. If the strategy produces fewer trades than the scaled target, this penalty will be a fraction less than 1.0.

### Objective 3: Minimize Risk (Maximum Drawdown Penalty)

- **What it is**: The Maximum Drawdown is the largest percentage drop from a portfolio's peak value to its subsequent lowest point. It answers the question: "What is the most money I could have lost if I invested at the worst possible time?"
- **How it works**:
    1. The maximum relative drawdown is calculated from the backtest results (e.g., 0.25 for a 25% drawdown).
    2. A `drawdown_penalty` is calculated using the formula: `penalty = (1 - relative_drawdown) ** DRAWDOWN_PENALTY_WEIGHT`.
    3. This formula heavily penalizes high drawdowns. A 25% drawdown results in a penalty factor of 0.75, while a 50% drawdown results in a factor of 0.5, significantly reducing the final score.

### Final Loss Calculation

The three components are combined into a single score that Freqtrade will minimize.

**`Loss Score = -Sortino Ratio * Trade Penalty * Drawdown Penalty`**

- The Sortino Ratio is multiplied by the two penalty factors. If either the trade count is too low or the drawdown is too high, the penalties will be less than 1.0, reducing the overall score.
- The entire result is negated (`-`). This is because hyperopt works to **minimize** the loss value. By making our desired outcome a large positive number and then negating it, we align our goal of *maximization* with Freqtrade's goal of *minimization*. The best-performing parameter set will be the one that produces the lowest (most negative) loss score.
