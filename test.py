# --- START OF FILE test.py ---

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# %matplotlib inline
import datetime
import ccxt

from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
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
    def __init__(self, exchange_name='binance', start_date='2020-01-01', end_date='2023-01-01', timeframe='1d'):
        self.exchange = getattr(ccxt, exchange_name)()
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe

    def get_data(self, ticker_list):
        all_data = []
        for ticker in ticker_list:
            print(f"Downloading {ticker} from {self.exchange.id}...")
            symbol = ticker + '/USDT' # Assuming USDT pairs for crypto
            
            # Convert start_date to milliseconds timestamp
            since = self.exchange.parse8601(self.start_date + 'T00:00:00Z')
            
            ohlcv = []
            while since < self.exchange.parse8601(self.end_date + 'T00:00:00Z'):
                fetched_data = self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
                if not fetched_data:
                    break
                ohlcv.extend(fetched_data)
                since = ohlcv[-1][0] + self.exchange.parse_timeframe(self.timeframe) * 1000 # Move to the next timeframe
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['tic'] = ticker
                all_data.append(df)
            else:
                print(f"No data downloaded for {ticker}")

        if all_data:
            full_df = pd.concat(all_data, axis=0)
            full_df = full_df.sort_values(by=['timestamp', 'tic']).reset_index(drop=True)
            
            # Format to match the required output
            full_df.rename(columns={'timestamp': 'date'}, inplace=True)
            full_df['adjcp'] = full_df['close'] # For crypto, adjcp can be the same as close
            full_df['day'] = full_df['date'].dt.dayofweek
            
            return full_df[['date', 'open', 'high', 'low', 'close', 'adjcp', 'volume', 'tic', 'day']]
        else:
            return pd.DataFrame()

downloader = CCXTDownloader(start_date=config.TRAIN_START_DATE, end_date=config.TRAIN_END_DATE)
df = downloader.get_data(config_tickers.CRYPTO_10_TICKER)
print(f"DataFrame shape after data download: {df.shape}")

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df = fe.preprocess_data(df)
print(f"DataFrame shape after preprocessing: {df.shape}")

# --- CORRECTED DATA PROCESSING PIPELINE ---

# 1. Split data FIRST
train = data_split(df, '2018-04-30','2020-07-01')

# --- NEW STEP TO DENSIFY THE TRAINING DATA ---
# This ensures that every date has a row for every ticker, preventing shape mismatches.
print("Densifying training data...")
# Get the full range of dates and all unique tickers from the training set
all_dates = pd.to_datetime(train['date'].unique())
all_tickers = train['tic'].unique()
multi_index = pd.MultiIndex.from_product([all_dates, all_tickers], names=['date', 'tic'])

# Set the index and reindex to create a complete, dense frame.
# This will introduce NaNs for dates/tickers that were missing.
train = train.set_index(['date', 'tic'])
train = train.reindex(multi_index)

# Group by ticker and forward-fill the missing data.
# bfill is used to fill any remaining NaNs at the beginning of the data.
train = train.groupby(level='tic').ffill().bfill()

# Reset the index to bring 'date' and 'tic' back as columns
train.reset_index(inplace=True)
print("Data densification complete.")

# 2. Function to calculate and add covariance matrix to a dataframe
def add_covariance_matrix(data):
    # Sort and factorize index
    data = data.sort_values(['date', 'tic'], ignore_index=True)
    data.index = data.date.factorize()[0]
    
    cov_list = []
    return_list = []
    # look back is one year
    lookback = 252
    
    for i in range(lookback, len(data.index.unique())):
        data_lookback = data.loc[i - lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        if not return_lookback.empty and len(return_lookback) > 1:
            covs = return_lookback.cov().values
        else:
            # Handle case with insufficient data
            num_tickers = len(data['tic'].unique())
            covs = np.zeros((num_tickers, num_tickers))
        cov_list.append(covs)

    # Merge covariance data
    # Make sure to only use dates that have a calculated covariance matrix
    df_cov = pd.DataFrame({
        'date': data.date.unique()[lookback:],
        'cov_list': cov_list,
        'return_list': return_list
    })
    data = data.merge(df_cov, on='date')
    data = data.sort_values(['date', 'tic']).reset_index(drop=True)
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
matplotlib.use('Agg')
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
    metadata = {'render.modes': ['human']}

# --- CORRECTED __init__ METHOD for StockPortfolioEnv ---

    def __init__(self,
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
                day = 0):
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        # --- FIX: ADD THE MISSING ATTRIBUTE ---
        self.model_name = self.__class__.__name__

        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list),self.state_space))
        
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].iloc[0]
        
        tech_indicator_data = self.data[self.tech_indicator_list].values.T
        self.state = np.vstack((self.covs, tech_indicator_data))

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount

        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(),'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()

            plt.plot(self.portfolio_return_memory,'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, False, {}

        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].iloc[0]
            
            tech_indicator_data = self.data[self.tech_indicator_list].values.T
            self.state = np.vstack((self.covs, tech_indicator_data))

            # calcualte portfolio return
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
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
        self.data = self.df.loc[self.day,:]
        
        self.covs = self.data['cov_list'].iloc[0]
        
        tech_indicator_data = self.data[self.tech_indicator_list].values.T
        self.state = np.vstack((self.covs, tech_indicator_data))

        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]
        return self.state, {}

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output


    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

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
        e.reset()
        return e, e.get_attr("model_name")[0]
        
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
    "reward_scaling": 1e-4

}

e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))
# ... (your existing code up to the creation of env_train)

# --- START OF FIX ---

# Apply the same processing to the trade dataframe
trade = data_split(df, '2020-07-01', '2021-10-31')

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

# Now, you can safely create the trading environment
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
# initialize
agent = DRLAgent(env = env_train)
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
trained_a2c = agent.train_model(model=model_a2c,
                                tb_log_name='a2c',
                                total_timesteps=50000)
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c,
                        environment = e_trade_gym)