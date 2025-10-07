# --- START OF FILE test.py ---

import os
import sys
import warnings
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ccxt
import pyfolio
import plotly.graph_objs as go

from finrl import config as finrl_config
from finrl import config_tickers
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
from pypfopt.efficient_frontier import EfficientFrontier
from gymnasium import spaces
import gymnasium


# ==============================================================================
# Suppress Warnings
# ==============================================================================
warnings.filterwarnings('ignore', category=UserWarning, module='pyfolio')
warnings.filterwarnings('ignore', category=FutureWarning, module='pyfolio')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.dates')
matplotlib.use("Agg")

# Append FinRL library path if not already in sys.path
if "../FinRL-Library" not in sys.path:
    sys.path.append("../FinRL-Library")

# ==============================================================================
# Configuration Parameters
# ==============================================================================
# --- Directory Config ---
DATA_SAVE_DIR = finrl_config.DATA_SAVE_DIR
TRAINED_MODEL_DIR = finrl_config.TRAINED_MODEL_DIR
TENSORBOARD_LOG_DIR = finrl_config.TENSORBOARD_LOG_DIR
RESULTS_DIR = finrl_config.RESULTS_DIR
DIRECTORIES = [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]

# --- Data Config ---
TICKER_LIST = config_tickers.CRYPTO_10_TICKER
TRAIN_START_DATE = "2018-04-30"
TRAIN_END_DATE = "2021-03-11"
TRADE_START_DATE = "2021-03-12"
TRADE_END_DATE = "2025-03-12"
BASELINE_TICKER = "BTC"

# --- Model Config ---
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
TRAINED_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, 'trained_sac.zip')

# ==============================================================================
# Helper Functions & Classes
# ==============================================================================

def setup_directories(dirs):
    """Create directories if they do not exist."""
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

class CCXTDownloader:
    """A class to download crypto data using the CCXT library."""
    def __init__(self, start_date, end_date, exchange_name="binance", timeframe="1d"):
        self.exchange = getattr(ccxt, exchange_name)()
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe

    def get_data(self, ticker_list):
        all_data = []
        for ticker in ticker_list:
            print(f"Downloading {ticker} from {self.exchange.id}...")
            symbol = f"{ticker}/USDT"
            since = self.exchange.parse8601(f"{self.start_date}T00:00:00Z")
            end_ts = self.exchange.parse8601(f"{self.end_date}T00:00:00Z")

            ohlcv = []
            while since < end_ts:
                try:
                    fetched_data = self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
                    if not fetched_data:
                        break
                    ohlcv.extend(fetched_data)
                    since = ohlcv[-1][0] + self.exchange.parse_timeframe(self.timeframe) * 1000
                except Exception as e:
                    print(f"Could not download {ticker}: {e}")
                    break # Stop trying for this ticker

            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df[df['timestamp'] < pd.to_datetime(self.end_date)]
                df["tic"] = ticker
                all_data.append(df)
            else:
                print(f"No data downloaded for {ticker}")

        if not all_data:
            return pd.DataFrame()

        full_df = pd.concat(all_data, axis=0).sort_values(by=["timestamp", "tic"]).reset_index(drop=True)
        full_df.rename(columns={"timestamp": "date"}, inplace=True)
        full_df["adjcp"] = full_df["close"]
        full_df["day"] = full_df["date"].dt.dayofweek
        return full_df[["date", "open", "high", "low", "close", "adjcp", "volume", "tic", "day"]]


def densify_data(df):
    """Fills missing date/ticker combinations using forward-fill."""
    if df.empty:
        return df
    print("Densifying data...")
    all_dates = pd.to_datetime(df["date"].unique())
    all_tickers = df["tic"].unique()
    multi_index = pd.MultiIndex.from_product([all_dates, all_tickers], names=["date", "tic"])
    
    df = df.set_index(["date", "tic"]).reindex(multi_index)
    df = df.groupby(level="tic").ffill().bfill()
    df.reset_index(inplace=True)
    print("Data densification complete.")
    return df

def add_covariance_matrix(data, lookback=252):
    """Calculates and adds covariance matrices to the dataframe."""
    if data.empty:
        return data
    data = data.sort_values(["date", "tic"], ignore_index=True)
    unique_dates = data.date.unique()
    
    if len(unique_dates) < lookback:
        print("Not enough data for covariance matrix calculation. Returning original data.")
        data['cov_list'] = [np.array([])] * len(data)
        data['return_list'] = [pd.DataFrame()] * len(data)
        return data.dropna().reset_index(drop=True)

    data.index = data.date.factorize()[0]
    cov_list, return_list = [], []

    for i in range(lookback, len(unique_dates)):
        data_lookback = data[data['date'].isin(unique_dates[i-lookback:i])]
        price_lookback = data_lookback.pivot_table(index="date", columns="tic", values="close")
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        
        covs = return_lookback.cov().values if not return_lookback.empty and len(return_lookback) > 1 else np.zeros((len(data["tic"].unique()), len(data["tic"].unique())))
        cov_list.append(covs)

    df_cov = pd.DataFrame({"date": unique_dates[lookback:], "cov_list": cov_list, "return_list": return_list})
    data = data.merge(df_cov, on="date").sort_values(["date", "tic"]).reset_index(drop=True)
    return data

def data_processing_pipeline(df, train_start, train_end, trade_start, trade_end):
    """Full data processing pipeline."""
    fe = FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)
    df = fe.preprocess_data(df)

    train = data_split(df, train_start, train_end)
    trade = data_split(df, trade_start, trade_end)
    
    train = densify_data(train)
    
    train_tickers = train['tic'].unique()
    trade = trade[trade['tic'].isin(train_tickers)].reset_index(drop=True)
    
    trade = densify_data(trade)

    train = add_covariance_matrix(train)
    trade = add_covariance_matrix(trade)
    
    if not train.empty:
        train.index = train.date.factorize()[0]
    if not trade.empty:
        trade.index = trade.date.factorize()[0]
    
    return train, trade

def train_agent(train_df, env_kwargs, model_params, model_save_path):
    """Trains the DRL agent and saves the model."""
    if train_df.empty:
        print("Training data is empty. Skipping training.")
        return None

    e_train_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    agent = DRLAgent(env=env_train)
    model = agent.get_model("sac", model_kwargs=model_params)
    
    log_path = os.path.join(TENSORBOARD_LOG_DIR, "sac")
    new_logger = configure(log_path, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    
    trained_model = model.learn(total_timesteps=5000) # Increased timesteps for better learning
    trained_model.save(model_save_path)
    return trained_model


def run_backtest(trade_df, env_kwargs, model_path):
    """Runs a backtest prediction using the trained model."""
    if trade_df.empty:
        print("Trade data is empty. Skipping backtest.")
        return pd.DataFrame(), pd.DataFrame()

    e_trade_gym = StockPortfolioEnv(df=trade_df, **env_kwargs)
    
    try:
        # CORRECTED LINE: Load the model using the SAC class directly
        trained_model = SAC.load(model_path)
        print("Trained model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return pd.DataFrame(), pd.DataFrame()
        
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=e_trade_gym)
    return df_daily_return, df_actions

def calculate_min_variance_portfolio(trade_df, initial_capital=1000000):
    """Calculates the performance of a minimum variance portfolio."""
    if trade_df.empty or 'cov_list' not in trade_df.columns:
        return pd.DataFrame()

    unique_trade_date = pd.to_datetime(trade_df.date.unique())
    portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date, dtype=float)
    portfolio.loc[0, unique_trade_date[0]] = initial_capital
    
    for i in range(len(unique_trade_date) - 1):
        current_date, next_date = unique_trade_date[i], unique_trade_date[i+1]
        
        df_temp = trade_df[trade_df.date == current_date].reset_index(drop=True)
        df_temp_next = trade_df[trade_df.date == next_date].reset_index(drop=True)
        
        if df_temp.empty or df_temp_next.empty: continue

        Sigma = df_temp['cov_list'].iloc[0]
        ef = EfficientFrontier(None, Sigma, weight_bounds=(0, 1))
        
        cleaned_weights = ef.min_volatility()
        cap = portfolio.iloc[0, i]
        current_cash = [w * cap for w in cleaned_weights.values()]
        current_shares = np.array(current_cash) / np.array(df_temp.close)
        
        portfolio.iloc[0, i+1] = np.dot(current_shares, df_temp_next.close)
        
    portfolio = portfolio.T.reset_index()
    portfolio.columns = ['date', 'account_value']
    portfolio['date'] = pd.to_datetime(portfolio['date'])
    return portfolio.set_index('date')

def plot_cumulative_returns(drl_returns, min_var_portfolio, baseline_returns):
    """Generates and shows a Plotly chart of cumulative returns."""
    fig = go.Figure()

    if not drl_returns.empty:
        drl_cumpod = (1 + drl_returns.daily_return).cumprod() - 1
        time_ind = pd.to_datetime(drl_returns.date)
        fig.add_trace(go.Scatter(x=time_ind, y=drl_cumpod, mode='lines', name='SAC Portfolio'))

    if not baseline_returns.empty:
        baseline_cumpod = (1 + baseline_returns).cumprod() - 1
        fig.add_trace(go.Scatter(x=baseline_cumpod.index, y=baseline_cumpod, mode='lines', name=f'{BASELINE_TICKER}'))
    
    if not min_var_portfolio.empty:
        min_var_cumpod = (1 + min_var_portfolio.account_value.pct_change().fillna(0)).cumprod() - 1
        fig.add_trace(go.Scatter(x=min_var_cumpod.index, y=min_var_cumpod, mode='lines', name='Min-Variance Portfolio'))

    fig.update_layout(
        title={'text': "Cumulative Return Comparison", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title="Cumulative Return",
        legend=dict(x=0.01, y=0.99, traceorder="normal", bgcolor="White", bordercolor="black", borderwidth=1),
        paper_bgcolor='white', plot_bgcolor='white'
    )
    fig.update_xaxes(showline=True, linecolor='black', showgrid=True, gridcolor='LightGray')
    fig.update_yaxes(showline=True, linecolor='black', showgrid=True, gridcolor='LightGray', zerolinecolor='Gray')
    fig.show()

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    setup_directories(DIRECTORIES)
    
    downloader = CCXTDownloader(start_date="2018-01-01", end_date=TRADE_END_DATE)
    raw_df = downloader.get_data(TICKER_LIST)
    
    train_df, trade_df = data_processing_pipeline(raw_df, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE)
    
    if train_df.empty or trade_df.empty:
        print("Not enough data to proceed with training or trading. Exiting.")
        return

    # --- FIX: DYNAMICALLY CONFIGURE ENVIRONMENT BASED ON PROCESSED DATA ---
    stock_dimension = len(train_df.tic.unique())
    state_space = stock_dimension
    print(f"Data Processed. Actual Stock Dimension for Training: {stock_dimension}")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": finrl_config.INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }
    # --- END OF FIX ---

    print("--- Starting Agent Training ---")
    train_agent(train_df, env_kwargs, SAC_PARAMS, TRAINED_MODEL_PATH)
    print("--- Agent Training Finished ---")

    print("--- Starting Backtesting ---")
    drl_daily_return, _ = run_backtest(trade_df, env_kwargs, TRAINED_MODEL_PATH)
    if not drl_daily_return.empty:
        drl_daily_return.to_csv(os.path.join(RESULTS_DIR, 'drl_daily_return.csv'))
    print("--- Backtesting Finished ---")

    print("--- Analyzing Performance ---")
    drl_strat_returns = convert_daily_return_to_pyfolio_ts(drl_daily_return)

    baseline_df = raw_df[(raw_df["tic"] == BASELINE_TICKER) & (raw_df["date"] >= pd.to_datetime(TRADE_START_DATE)) & (raw_df["date"] < pd.to_datetime(TRADE_END_DATE))].copy()
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    if baseline_returns.index.tz is None:
        baseline_returns = baseline_returns.tz_localize('UTC')
    
    min_var_portfolio = calculate_min_variance_portfolio(trade_df, initial_capital=env_kwargs['initial_amount'])

    aligned_drl, aligned_baseline = drl_strat_returns.align(baseline_returns, join='inner')
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(returns=aligned_drl, benchmark_rets=aligned_baseline, set_context=False)

    print("--- Generating Final Plot ---")
    plot_cumulative_returns(drl_daily_return, min_var_portfolio, baseline_returns)

if __name__ == "__main__":
    main()