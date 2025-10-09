import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nest_asyncio

# --- Global Configuration Variables ---
FREQTRADE_LOGFILE_PATH = "/freqtrade/user_data/logs/freqtrade.log"
PORTFOLIO_CONFIG_PATH = Path("./portfolio/configs/portfolio_config.json")
FREQTRADE_BASE_CONFIG_PATH = Path("./user_data/configs/hyperopt_config.json")
HYPEROPT_SPACES = ["buy"]  # This can be extended if other spaces are needed later
HYPEROPT_JOB_WORKERS = "-1"
HYPEROPT_EPOCHS = "500"
HYPEROPT_TIMERANGE = "20120101-"

# --- Global Configuration & Setup ---

# Apply nest_asyncio to allow running asyncio within an existing event loop (e.g., in Jupyter)
nest_asyncio.apply()

# Configure logging
LOG_FILE = Path("./portfolio/logfiles/run_hyperopt.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Set freqtrade logger to WARNING level to suppress INFO logs
logging.getLogger("freqtrade").setLevel(logging.WARNING)

# --- Helper Functions ---

INTERVAL_TO_TIMEFRAME = {
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


def get_timeframe_from_interval(interval_seconds: int) -> str:
    """Get timeframe string from interval in seconds."""
    return INTERVAL_TO_TIMEFRAME.get(interval_seconds, "1d")


# --- Main Logic Class ---


class PairOptimizer:
    """
    Encapsulates the entire hyper-optimization process for a single trading pair.
    """

    def __init__(
        self,
        pair: str,
        exchange: str,
        position_interval: int,
        trend_interval: int,
        base_config: Dict[str, Any],
    ):
        self.pair = pair
        self.ticker = pair.split("/")[0]
        self.exchange = exchange
        self.position_timeframe = get_timeframe_from_interval(position_interval)
        self.trend_timeframe = get_timeframe_from_interval(trend_interval)
        self.base_config = base_config
        self.user_data_dir = base_config["user_data_dir"]
        self.data_format = base_config.get("dataformat_ohlcv", "feather")
        self.logger = logging.getLogger(f"{__name__}.{self.pair.replace('/', '_')}")

    async def run(self) -> None:
        """
        Executes the full optimization pipeline for the pair.
        1. Runs position strategy optimization.
        2. Runs trend strategy optimization.
        3. Combines the results into a final strategy file.
        """
        self.logger.info("Starting optimization process.")
        try:
            # Create a temporary config file to avoid race conditions
            with tempfile.NamedTemporaryFile(
                mode="w+", dir="./user_data/configs", suffix=".json", delete=False
            ) as tmp_config_file:
                self.config_path = Path(tmp_config_file.name)
                json.dump(self.base_config, tmp_config_file, indent=4)
            trend_params = await self._run_stage("trend")
            position_params = await self._run_stage("position")

            if position_params and trend_params:
                self._save_combined_strategy(position_params, trend_params)
                self.logger.info("Successfully completed optimization pipeline.")
            else:
                self.logger.error("Failed to get parameters from one or both stages.")

        except Exception as e:
            self.logger.critical(f"A critical error occurred: {e}", exc_info=True)
        finally:
            # Ensure temporary config file is cleaned up
            if hasattr(self, "config_path") and self.config_path.exists():
                self.config_path.unlink()

    async def _run_stage(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Runs a single optimization stage (e.g., 'position' or 'trend').
        This method encapsulates the duplicated logic from the original script.
        """
        self.logger.info(f"--- Starting '{stage_name}' stage ---")
        is_position_stage = stage_name == "position"

        strategy_name = (
            "KeltnerChannelStrategyPosition"
            if is_position_stage
            else "KeltnerChannelStrategyTrend"
        )
        timeframe = (
            self.position_timeframe if is_position_stage else self.trend_timeframe
        )
        spaces = HYPEROPT_SPACES

        # 1. Update and save configuration for this stage
        self._update_freqtrade_config(strategy_name, timeframe, spaces)

        # 2. Download data
        if not await self._download_data_with_retry(timeframe):
            self.logger.error(f"Failed to download data for stage '{stage_name}'.")
            return None

        # 3. Run Freqtrade commands via OS commands
        config_path_str = str(self.config_path)
        self.logger.info(f"Running hyperopt for {strategy_name} using CLI...")
        try:
            hyperopt_command = [
                "freqtrade",
                "hyperopt",
                "--config",
                config_path_str,
                "--logfile",
                FREQTRADE_LOGFILE_PATH,
                "--job-workers",
                HYPEROPT_JOB_WORKERS,
                "--epochs",
                HYPEROPT_EPOCHS,
                "--timerange",
                HYPEROPT_TIMERANGE,
            ]
            hyperopt_command.extend(["--spaces"] + spaces)
            result = subprocess.run(
                hyperopt_command, capture_output=True, text=True, check=True
            )
            self.logger.info(f"Hyperopt stdout:\n{result.stdout}")
            if result.stderr:
                self.logger.warning(f"Hyperopt stderr:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Hyperopt command failed with error: {e}")
            self.logger.error(f"Hyperopt stdout:\n{e.stdout}")
            self.logger.error(f"Hyperopt stderr:\n{e.stderr}")
            return None
        except FileNotFoundError:
            self.logger.error(
                "Freqtrade command not found. Is Freqtrade installed and in PATH?"
            )
            return None

        self.logger.info(f"Running backtesting for {strategy_name} using CLI...")
        try:
            backtesting_command = [
                "freqtrade",
                "backtesting",
                "--config",
                config_path_str,
                "--logfile",
                FREQTRADE_LOGFILE_PATH,
            ]
            result = subprocess.run(
                backtesting_command, capture_output=True, text=True, check=True
            )
            self.logger.info(f"Backtesting stdout:\n{result.stdout}")
            if result.stderr:
                self.logger.warning(f"Backtesting stderr:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Backtesting command failed with error: {e}")
            self.logger.error(f"Backtesting stdout:\n{e.stdout}")
            self.logger.error(f"Backtesting stderr:\n{e.stderr}")
            return None
        except FileNotFoundError:
            self.logger.error(
                "Freqtrade command not found. Is Freqtrade installed and in PATH?"
            )
            return None

        # 4. Extract results
        results_path = Path(f"{self.user_data_dir}/strategies/{strategy_name}.json")
        if not results_path.exists():
            self.logger.error(f"Results file not found at {results_path}")
            return None

        with results_path.open("r") as f:
            params = json.load(f).get("params")
            if not params:
                self.logger.error(f"No 'params' key in results file {results_path}")
                return None

        self.logger.info(f"--- Finished '{stage_name}' stage ---")
        return params

    def _update_freqtrade_config(
        self, strategy: str, timeframe: str, spaces: List[str]
    ):
        """Updates the config file for a specific optimization run."""
        config_data = self.base_config.copy()
        config_data.update(
            {
                "strategy": strategy,
                "timeframe": timeframe,
                "spaces": spaces,
                "exchange": {
                    "name": self.exchange,
                    "pair_whitelist": [self.pair],
                },
            }
        )
        with self.config_path.open("w") as f:
            json.dump(config_data, f, indent=4)
        self.logger.info(
            f"Updated config for strategy '{strategy}' and timeframe '{timeframe}'."
        )

    async def _download_data_with_retry(
        self, timeframe: str, max_retries: int = 3
    ) -> bool:
        """Downloads data for the pair/timeframe, with retries."""
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Attempt {attempt + 1}/{max_retries} to download data for "
                    f"timeframe {timeframe}."
                )
                # The download command uses the pair_whitelist from the config file
                # freqtrade download-data --config user_data/configs/hyperopt_config.json --timeframes '15m'
                download_command = [
                    "freqtrade",
                    "download-data",
                    "--config",
                    str(self.config_path),
                    "--pairs",
                    self.pair,
                    "--timeframes",
                    timeframe,
                    "--logfile",
                    FREQTRADE_LOGFILE_PATH,
                    "--prepend",
                ]
                result = subprocess.run(
                    download_command, capture_output=True, text=True, check=True
                )
                self.logger.info(f"Download data stdout:\n{result.stdout}")
                if result.stderr:
                    self.logger.warning(f"Download data stderr:\n{result.stderr}")

                if self._validate_data(timeframe):
                    self.logger.info("Data download and validation successful.")
                    return True
                self.logger.warning("Data downloaded but failed validation.")

            except Exception as e:
                self.logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time

                    time.sleep(5 * (attempt + 1))
        return False

    def _validate_data(self, timeframe: str) -> bool:
        """Checks if the data file exists and is not empty."""
        data_path = (
            Path(f"{self.user_data_dir}/data/{self.exchange}")
            / f"{self.ticker}_{self.pair.split('/')[1]}-{timeframe}.{self.data_format}"
        )
        self.logger.info(f"Validating data at: {data_path}")
        return data_path.exists() and data_path.stat().st_size > 0

    def _save_combined_strategy(self, pos_params: Dict, trend_params: Dict):
        """Combines parameters and saves the final strategy JSON."""
        strategy_config = {
            "position_kc_window": pos_params["buy"]["kc_window"],
            "position_kc_atr": pos_params["buy"]["kc_atrs"],
            "position_kc_mult": pos_params["buy"]["kc_mult"],
            "trend_kc_window": trend_params["buy"]["kc_window"],
            "trend_kc_atr": trend_params["buy"]["kc_atrs"],
            "trend_kc_mult": trend_params["buy"]["kc_mult"],
        }
        strategy_file = Path(f"portfolio/strategies/{self.ticker}_strategy.json")
        strategy_file.parent.mkdir(parents=True, exist_ok=True)
        with strategy_file.open("w") as f:
            json.dump(strategy_config, f, indent=4)
        self.logger.info(f"Saved combined strategy to {strategy_file}")


def load_app_config(config_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Loads the main portfolio configuration."""
    with config_path.open() as f:
        config_data = json.load(f)

    portfolio_assets = config_data.get("portfolio_assets", [])
    stablecoin = next(
        (asset["ticker"] for asset in portfolio_assets if asset.get("level") == 0),
        None,
    )
    if not stablecoin:
        raise ValueError(
            "No stablecoin (asset with level 0) found in portfolio_config.json"
        )

    assets_to_process = [asset for asset in portfolio_assets if asset.get("level") != 0]
    return stablecoin, assets_to_process


async def main():
    """Main execution function."""
    try:
        stablecoin, assets = load_app_config(PORTFOLIO_CONFIG_PATH)

        with FREQTRADE_BASE_CONFIG_PATH.open() as f:
            base_ft_config = json.load(f)
            # Set a common runmode
            base_ft_config["runmode"] = "hyperopt"

        tasks = []
        for asset in assets:
            optimizer = PairOptimizer(
                pair=f"{asset['ticker']}/{stablecoin}",
                exchange=asset["exchange"],
                position_interval=asset["position_interval"],
                trend_interval=asset["trend_interval"],
                base_config=base_ft_config,
            )
            tasks.append(optimizer.run())

        logger.info(f"Starting concurrent optimization for {len(tasks)} pairs.")
        await asyncio.gather(*tasks)
        logger.info("All optimization tasks completed.")

    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.critical(f"Configuration error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"An unexpected fatal error occurred in main: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    # Use the modern, simpler way to run an asyncio event loop
    asyncio.run(main())
