# --- START OF FILE test.py ---

import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Suppress specific warnings from third-party libraries that are not easily fixed
warnings.filterwarnings('ignore', category=UserWarning, module='pyfolio.pos')
warnings.filterwarnings('ignore', category=FutureWarning, module='pyfolio.plotting')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.dates')
warnings.filterwarnings('ignore', category=UserWarning, module='pyfolio.plotting')


matplotlib.use("Agg")
# %matplotlib inline
import datetime
import ccxt

from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import (
    backtest_stats,
    backtest_plot,
    get_daily_return,
    get_baseline,
    convert_daily_return_to_pyfolio_ts,
)
from finrl.meta.data_processor import DataProcessor
import sys

sys.path.append("../FinRL-Library")
import os

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


class CCXTDownloader:
    def __init__(
        self,
        exchange_name="binance",
        start_date="2020-01-01",
        end_date="2023-01-01",
        timeframe="1d",
    ):
        self.exchange = getattr(ccxt, exchange_name)()
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe

    def get_data(self, ticker_list):
        all_data = []
        for ticker in ticker_list:
            print(f"Downloading {ticker} from {self.exchange.id}...")
            symbol = ticker + "/USDT"  # Assuming USDT pairs for crypto

            # Convert start_date to milliseconds timestamp
            since = self.exchange.parse8601(self.start_date + "T00:00:00Z")

            ohlcv = []
            while since < self.exchange.parse8601(self.end_date + "T00:00:00Z"):
                fetched_data = self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
                if not fetched_data:
                    break
                ohlcv.extend(fetched_data)
                since = (
                    ohlcv[-1][0] + self.exchange.parse_timeframe(self.timeframe) * 1000
                )  # Move to the next timeframe

            if ohlcv:
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["tic"] = ticker
                all_data.append(df)
            else:
                print(f"No data downloaded for {ticker}")

        if all_data:
            full_df = pd.concat(all_data, axis=0)
            full_df = full_df.sort_values(by=["timestamp", "tic"]).reset_index(
                drop=True
            )

            # Format to match the required output
            full_df.rename(columns={"timestamp": "date"}, inplace=True)
            full_df["adjcp"] = full_df[
                "close"
            ]  # For crypto, adjcp can be the same as close
            full_df["day"] = full_df["date"].dt.dayofweek

            return full_df[
                [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjcp",
                    "volume",
                    "tic",
                    "day",
                ]
            ]
        else:
            return pd.DataFrame()


downloader = CCXTDownloader(
    start_date=config.TRAIN_START_DATE, end_date=config.TRAIN_END_DATE
)
df = downloader.get_data(config_tickers.CRYPTO_10_TICKER)
print(f"DataFrame shape after data download: {df.shape}")

fe = FeatureEngineer(
    use_technical_indicator=True, use_turbulence=False, user_defined_feature=False
)

df = fe.preprocess_data(df)
print(f"DataFrame shape after preprocessing: {df.shape}")

# --- CORRECTED DATA PROCESSING PIPELINE ---

# 1. Split data FIRST
train = data_split(df, "2018-04-30", "2020-07-01")

# --- NEW STEP TO DENSIFY THE TRAINING DATA ---
# This ensures that every date has a row for every ticker, preventing shape mismatches.
print("Densifying training data...")
# Get the full range of dates and all unique tickers from the training set
all_dates = pd.to_datetime(train["date"].unique())
all_tickers = train["tic"].unique()
multi_index = pd.MultiIndex.from_product(
    [all_dates, all_tickers], names=["date", "tic"]
)

# Set the index and reindex to create a complete, dense frame.
# This will introduce NaNs for dates/tickers that were missing.
train = train.set_index(["date", "tic"])
train = train.reindex(multi_index)

# Group by ticker and forward-fill the missing data.
# bfill is used to fill any remaining NaNs at the beginning of the data.
train = train.groupby(level="tic").ffill().bfill()

# Reset the index to bring 'date' and 'tic' back as columns
train.reset_index(inplace=True)
print("Data densification complete.")


# 2. Function to calculate and add covariance matrix to a dataframe
def add_covariance_matrix(data):
    # Sort and factorize index
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]

    cov_list = []
    return_list = []
    # look back is one year
    lookback = 252

    for i in range(lookback, len(data.index.unique())):
        data_lookback = data.loc[i - lookback : i, :]
        price_lookback = data_lookback.pivot_table(
            index="date", columns="tic", values="close"
        )
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        if not return_lookback.empty and len(return_lookback) > 1:
            covs = return_lookback.cov().values
        else:
            # Handle case with insufficient data
            num_tickers = len(data["tic"].unique())
            covs = np.zeros((num_tickers, num_tickers))
        cov_list.append(covs)

    # Merge covariance data
    # Make sure to only use dates that have a calculated covariance matrix
    df_cov = pd.DataFrame(
        {
            "date": data.date.unique()[lookback:],
            "cov_list": cov_list,
            "return_list": return_list,
        }
    )
    data = data.merge(df_cov, on="date")
    data = data.sort_values(["date", "tic"]).reset_index(drop=True)
    return data


# 3. Apply the function to the train set
train = add_covariance_matrix(train)

# --- FIX: SET A DAY-BASED INDEX FOR THE ENVIRONMENT ---
train.index = train.date.factorize()[0]

# --- END OF CORRECTION ---

import numpy as np
import pandas as pd
from gymnasium.utils import seeding
from gymnasium import spaces
import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class StockPortfolioEnv(gymnasium.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """

    metadata = {"render.modes": ["human"]}

    # --- CORRECTED __init__ METHOD for StockPortfolioEnv ---

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
    ):
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        # --- FIX: ADD THE MISSING ATTRIBUTE ---
        self.model_name = self.__class__.__name__

        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
        )

        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].iloc[0]

        tech_indicator_data = self.data[self.tech_indicator_list].values.T
        self.state = np.vstack((self.covs, tech_indicator_data))

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount

        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, False, {}

        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].iloc[0]

            tech_indicator_data = self.data[self.tech_indicator_list].values.T
            self.state = np.vstack((self.covs, tech_indicator_data))

            # calcualte portfolio return
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            self.reward = new_portfolio_value

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]

        self.covs = self.data["cov_list"].iloc[0]

        tech_indicator_data = self.data[self.tech_indicator_list].values.T
        self.state = np.vstack((self.covs, tech_indicator_data))

        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state, {}

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        # The .reset() method of a VecEnv returns the initial observation
        obs = e.reset()
        # Return the environment and the initial observation
        return e, obs


stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockPortfolioEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))
# initialize
agent = DRLAgent(env = env_train)
# --- START OF FIX ---

from stable_baselines3.common.logger import configure

### Model 4: **SAC**

agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

# 1. Get the model from the agent
model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

# 2. Create and set a new, clean logger for the SAC model.
#    This bypasses the faulty default logger configuration from the FinRL wrapper.
#    We are telling it to output to the console ("stdout") and to TensorBoard.
tmp_path = config.TENSORBOARD_LOG_DIR + "/sac/"
new_logger = configure(tmp_path, ["stdout", "tensorboard"])
model_sac.set_logger(new_logger)

# 3. Call model.learn() directly instead of using agent.train_model().
#    This ensures our new logger is used and prevents the "rollout_buffer" error.
trained_sac = model_sac.learn(total_timesteps=50) # 50000

# 4. Manually save the trained model.
trained_sac.save(config.TRAINED_MODEL_DIR + '/trained_sac.zip')

# --- END OF FIX ---
trade = data_split(df, '2020-07-01', '2021-10-31')

# --- NEW STEP: Ensure consistent tickers across train and trade sets ---
# Filter the trade data to only include tickers present in the training data.
# This is crucial for ensuring the observation and action spaces match the trained model.
train_tickers = train.tic.unique()
trade = trade[trade['tic'].isin(train_tickers)].reset_index(drop=True)
print(f"Tickers in train: {len(train_tickers)}. Tickers in trade after filtering: {len(trade.tic.unique())}")


# 1. Densify the trading data
print("Densifying trading data...")
all_dates_trade = pd.to_datetime(trade['date'].unique())
all_tickers_trade = trade['tic'].unique()
multi_index_trade = pd.MultiIndex.from_product([all_dates_trade, all_tickers_trade], names=['date', 'tic'])

trade = trade.set_index(['date', 'tic'])
trade = trade.reindex(multi_index_trade)
trade = trade.groupby(level='tic').ffill().bfill()
trade.reset_index(inplace=True)
print("Trading data densification complete.")

# 2. Add covariance matrix to the trade set
trade = add_covariance_matrix(trade)

# 3. Set the day-based index for the environment
trade.index = trade.date.factorize()[0]

# --- END OF FIX ---

# Now, the trade environment will have the same dimensions as the train environment
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)

trade.shape
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_sac,
                        environment = e_trade_gym)
df_daily_return.head()
df_daily_return.to_csv('df_daily_return.csv')
df_actions.head()
df_actions.to_csv('df_actions.csv')
# <a id='6'></a>
# Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.
# <a id='6.1'></a>
## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class

from pyfolio import timeseries
DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
print("==============DRL Strategy Stats===========")
perf_stats_all
#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI",
        start = df_daily_return.loc[0,'date'],
        end = df_daily_return.loc[len(df_daily_return)-1,'date'])

stats = backtest_stats(baseline_df, value_col_name = 'close')
# <a id='6.2'></a>
## 7.2 BackTestPlot
import pyfolio
from pyfolio import timeseries
# %matplotlib inline

# --- START OF FIX ---

# 1. Fetch the baseline returns.
baseline_df = get_baseline(
    ticker='^DJI', start=df_daily_return.loc[0,'date'], end='2021-11-01'
)
baseline_returns = get_daily_return(baseline_df, value_col_name="close")

# 2. Robustly ensure the baseline is in UTC.
# First, check if the index is already timezone-aware.
if baseline_returns.index.tz is None:
    # If it's "naive" (no timezone), localize it to UTC.
    baseline_returns = baseline_returns.tz_localize('UTC')
else:
    # If it's already "aware," convert it to UTC.
    baseline_returns = baseline_returns.tz_convert('UTC')

# 3. Align the strategy and benchmark returns (this part remains the same).
# This ensures we only compare dates that exist in both datasets.
aligned_returns, aligned_baseline_returns = DRL_strat.align(baseline_returns, join='inner')

# --- END OF FIX ---

# 4. Pass the aligned, UTC-standardized data to pyfolio.
with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=aligned_returns,
                                   benchmark_rets=aligned_baseline_returns,
                                   set_context=False)
## Min-Variance Portfolio Allocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
unique_tic = trade.tic.unique()
unique_trade_date = trade.date.unique()
df.head()
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models

unique_tic = trade.tic.unique()
unique_trade_date = trade.date.unique()

# --- START OF FIX ---

# Diagnostic: Let's confirm how many tickers we are working with.
# This will likely print a number less than 10.
print(f"Number of unique tickers in the final trade set: {len(unique_tic)}")

# Calculate portfolio minimum variance
portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date)
initial_capital = 1000000
portfolio.loc[0, unique_trade_date[0]] = initial_capital

for i in range(len(unique_trade_date) - 1):
    df_temp = trade[trade.date == unique_trade_date[i]].reset_index(drop=True)
    df_temp_next = trade[trade.date == unique_trade_date[i+1]].reset_index(drop=True)

    Sigma = df_temp['cov_list'].iloc[0]

    # Portfolio allocation
    # The min_volatility method does not need expected returns, so we pass None.
    # We relax the weight bound to (0, 1) to ensure a solution is always feasible.
    ef_min_var = EfficientFrontier(None, Sigma, weight_bounds=(0, 1))

    # Minimum variance
    raw_weights_min_var = ef_min_var.min_volatility()
    # Get weights
    cleaned_weights_min_var = ef_min_var.clean_weights()

    # Current capital
    cap = portfolio.iloc[0, i]
    # Current cash invested for each stock
    current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]
    # Current held shares
    current_shares = list(np.array(current_cash) / np.array(df_temp.close))
    # Next time period price
    next_price = np.array(df_temp_next.close)
    # next_price * current share to calculate next total account value
    portfolio.iloc[0, i+1] = np.dot(current_shares, next_price)

# --- END OF FIX ---

portfolio = portfolio.T
portfolio.columns = ['account_value']

# --- FIX for FutureWarning ---
# Explicitly convert the 'account_value' column to a numeric type before calculations.
portfolio['account_value'] = pd.to_numeric(portfolio['account_value'], errors='coerce')


print("Successfully calculated minimum variance portfolio:")
print(portfolio.head())

# This line will no longer produce a warning.
sac_cumpod =(df_daily_return.daily_return+1).cumprod()-1
min_var_cumpod =(portfolio.account_value.pct_change().fillna(0)+1).cumprod()-1
dji_cumpod =(baseline_returns+1).cumprod()-1
## Plotly: DRL, Min-Variance, DJIA
# %pip install plotly
from datetime import datetime as dt

import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import nbformat
from nbformat import v4 as nbf
time_ind = pd.Series(df_daily_return.date)
trace0_portfolio = go.Scatter(x = time_ind, y = sac_cumpod, mode = 'lines', name = 'SAC (Portfolio Allocation)')

trace1_portfolio = go.Scatter(x = time_ind, y = dji_cumpod, mode = 'lines', name = 'DJIA')
trace2_portfolio = go.Scatter(x = time_ind, y = min_var_cumpod, mode = 'lines', name = 'Min-Variance')
#trace3_portfolio = go.Scatter(x = time_ind, y = ddpg_cumpod, mode = 'lines', name = 'DDPG')
#trace4_portfolio = go.Scatter(x = time_ind, y = addpg_cumpod, mode = 'lines', name = 'Adaptive-DDPG')
#trace5_portfolio = go.Scatter(x = time_ind, y = min_cumpod, mode = 'lines', name = 'Min-Variance')

#trace4 = go.Scatter(x = time_ind, y = addpg_cumpod, mode = 'lines', name = 'Adaptive-DDPG')

#trace2 = go.Scatter(x = time_ind, y = portfolio_cost_minv, mode = 'lines', name = 'Min-Variance')
#trace3 = go.Scatter(x = time_ind, y = spx_value, mode = 'lines', name = 'SPX')
fig = go.Figure()
fig.add_trace(trace0_portfolio)

fig.add_trace(trace1_portfolio)

fig.add_trace(trace2_portfolio)



fig.update_layout(
    legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=15,
            color="black"
        ),
        bgcolor="White",
        bordercolor="white",
        borderwidth=2

    ),
)
#fig.update_layout(legend_orientation="h")
fig.update_layout(title={
        #'text': "Cumulative Return using FinRL",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
#with Transaction cost
#fig.update_layout(title =  'Quarterly Trade Date')
fig.update_layout(
#    margin=dict(l=20, r=20, t=20, b=20),

    paper_bgcolor='rgba(1,1,0,0)',
    plot_bgcolor='rgba(1, 1, 0, 0)',
    #xaxis_title="Date",
    yaxis_title="Cumulative Return",
xaxis={'type': 'date',
       'tick0': time_ind[0],
        'tickmode': 'linear',
       'dtick': 86400000.0 *80}

)
fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

fig.show()