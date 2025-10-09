# --- START OF FILE test.py ---

import os
import sys
import warnings
import datetime
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ccxt
import pyfolio
import plotly.graph_objs as go
import json
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from pypfopt.efficient_frontier import EfficientFrontier
from stockstats import StockDataFrame as Sdf # Needed for FeatureEngineer
from sklearn.base import BaseEstimator, TransformerMixin # Needed for FeatureEngineer
from sklearn.preprocessing import MaxAbsScaler # Needed for FeatureEngineer
import statistics # Needed for DRLAgent
import time # Needed for DRLAgent
import copy # Needed for plot functions
from copy import deepcopy # Needed for plot functions
from pyfolio import timeseries # Needed for plot functions
import matplotlib.dates as mdates # Needed for plot functions
from typing import Type
import yfinance as yf

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
# FinRL Config (Copied from finrl/config.py - necessary parts)
# ==============================================================================
DATA_SAVE_DIR_FINRL = "datasets"
TRAINED_MODEL_DIR_FINRL = "trained_models"
TENSORBOARD_LOG_DIR_FINRL = "tensorboard_log"
RESULTS_DIR_FINRL = "results"

INDICATORS_FINRL = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# ==============================================================================
# FinRL Config Tickers (Copied from finrl/config_tickers.py - necessary parts)
# ==============================================================================
CRYPTO_10_TICKER_FINRL = [
    "BTC", "ETH", "ADA", "SOL", "LINK", "DOT", "APT", "SUI", "OP", "NEAR"
]

# ==============================================================================
# Configuration Parameters (Updated to use local FinRL config)
# ==============================================================================
# --- Directory Config ---
DATA_SAVE_DIR = DATA_SAVE_DIR_FINRL
TRAINED_MODEL_DIR = TRAINED_MODEL_DIR_FINRL
TENSORBOARD_LOG_DIR = TENSORBOARD_LOG_DIR_FINRL
RESULTS_DIR = RESULTS_DIR_FINRL
DIRECTORIES = [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]

# --- Data Config ---
TICKER_LIST = CRYPTO_10_TICKER_FINRL
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

# A2C_PARAMS and other model parameters from finrl/config.py
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}


# ==============================================================================
# Helper Functions & Classes
# ==============================================================================

def setup_directories(dirs):
    """Create directories if they do not exist."""
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None, auto_adjust=False) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                proxy=proxy,
                auto_adjust=auto_adjust,
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )

            if not auto_adjust:
                data_df = self._adjust_prices(data_df)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def _adjust_prices(self, data_df: pd.DataFrame) -> pd.DataFrame:
        # use adjusted close price instead of close price
        data_df["adj"] = data_df["adjcp"] / data_df["close"]
        for col in ["open", "high", "low", "close"]:
            data_df[col] *= data_df["adj"]

        # drop the adjusted close price column
        return data_df.drop(["adjcp", "adj"], axis=1)

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df

def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

class FinRLFeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            use user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS_FINRL,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.ffill().bfill()
        return df

    def clean_data(self, data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    print(f"Error calculating indicator {indicator} for ticker {unique_ticker[i]}: {e}")
            
            if indicator in indicator_df.columns:
                df = df.merge(
                    indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
                )
            else:
                print(f"Indicator {indicator} not found in indicator_df, skipping merge for this indicator.")
        df = df.sort_values(by=["date", "tic"])
        return df

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        return df

    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        print("YahooDownloader is not available in this context.")
        return data

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index

class FinRLStockPortfolioEnv(gym.Env):
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
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
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

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

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
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

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
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) /
            #                   (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            # print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
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
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        # This method uses gymnasium.utils.seeding, which is not strictly needed if not using _seed
        # However, to keep the copied class complete, I will include it.
        # If gymnasium.utils.seeding is not available, this might cause an issue.
        # For now, I will assume it's available or that _seed is not called.
        # self.np_random, seed = seeding.np_random(seed) # seeding needs to be imported
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: globals()[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
                # Print the original error and the inner error for debugging
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True

    def _on_rollout_end(self) -> bool:
        try:
            rollout_buffer_rewards = self.locals["rollout_buffer"].rewards.flatten()
            self.logger.record(
                key="train/reward_min", value=min(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_mean", value=statistics.mean(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_max", value=max(rollout_buffer_rewards)
            )
        except BaseException as error:
            # Handle the case where "rewards" is not found
            self.logger.record(key="train/reward_min", value=None)
            self.logger.record(key="train/reward_mean", value=None)
            self.logger.record(key="train/reward_max", value=None)
            print("Logging Error:", error)
        return True


class FinRLDRLAgent: # Renamed from DRLAgent
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        tb_log_name,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """make a prediction and get results"""
        test_env, test_obs = environment.get_sb_env()
        account_memory = None
        actions_memory = None

        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)

            if (
                i == max_steps - 1
            ):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )
        try:
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        state = environment.reset()
        episode_returns = []
        episode_total_assets = [environment.initial_total_asset]
        done = False
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)

            total_asset = (
                environment.amount
                + (environment.price_ary[environment.day] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets

def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def FinRLGetBaseline(ticker, start, end):
    return YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()


def load_keltner_params(ticker: str, strategies_dir: str = "./portfolio/strategies/") -> dict:
    """
    Loads hyperoptimized Keltner Channel parameters for a given ticker from a JSON file.
    Returns default parameters if the file is not found or parameters are missing.
    """
    strategy_file_path = os.path.join(strategies_dir, f"{ticker}_strategy.json")
    default_params = {
        "window": 20,
        "window_atr": 10,
        "window_mult": 2.0,
    }
    
    try:
        with open(strategy_file_path, 'r') as f:
            params = json.load(f)
            # We are interested in 'position' parameters for the FinRL feature engineering
            # The hyperopt script saves them as 'position_kc_window', etc.
            kc_params = {
                "window": params.get("position_kc_window", default_params["window"]),
                "window_atr": params.get("position_kc_atr", default_params["window_atr"]),
                "window_mult": params.get("position_kc_mult", default_params["window_mult"]),
            }
            print(f"Loaded optimized Keltner Channel parameters for {ticker}: {kc_params}")
            return kc_params
    except FileNotFoundError:
        print(f"Strategy file not found for {ticker} at {strategy_file_path}. Using default Keltner Channel parameters.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {strategy_file_path}: {e}. Using default Keltner Channel parameters.")
    except KeyError as e:
        print(f"Missing expected parameter in {strategy_file_path}: {e}. Using default Keltner Channel parameters.")
    except Exception as e:
        print(f"Unexpected error loading Keltner Channel parameters for {ticker}: {e}. Using default parameters.")
    
    return default_params


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
        typical_price = (self._open + self._high + self._low + self._close) / 4.0
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


class CustomFeatureEngineer(FinRLFeatureEngineer): # Changed base class
    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS_FINRL, # Changed from finrl_config.INDICATORS
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        super().__init__(
            use_technical_indicator,
            tech_indicator_list,
            use_vix,
            use_turbulence,
            user_defined_feature,
        )
        # Add Keltner Channel indicators to the list
        self.tech_indicator_list.extend([
            "kc_mband", "kc_hband", "kc_lband", "kc_wband", "kc_pband"
        ])
        # Ensure no duplicates
        self.tech_indicator_list = list(set(self.tech_indicator_list))

    def add_technical_indicator(self, data):
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])

        # Temporarily set tech_indicator_list to only original FinRL indicators for super() call
        original_finrl_indicators = [ind for ind in INDICATORS_FINRL if ind not in ["kc_mband", "kc_hband", "kc_lband", "kc_wband", "kc_pband"]]
        
        # Create a temporary FinRLFeatureEngineer instance to call the parent's add_technical_indicator
        # This avoids modifying self.tech_indicator_list which is needed for the environment
        temp_fe = FinRLFeatureEngineer( # Changed class name
            use_technical_indicator=True,
            tech_indicator_list=original_finrl_indicators,
            use_vix=self.use_vix,
            use_turbulence=self.use_turbulence,
            user_defined_feature=self.user_defined_feature
        )
        df = temp_fe.add_technical_indicator(df)

        # Now, calculate Keltner Channel indicators separately
        unique_ticker = df.tic.unique()
        keltner_data = []

        for ticker in unique_ticker:
            ticker_df = df[df.tic == ticker].copy()
            
            # Load hyperoptimized parameters for the current ticker
            kc_params = load_keltner_params(ticker)

            if len(ticker_df) > max(kc_params["window"], kc_params["window_atr"]): 
                kc = KeltnerChannel(
                    open=ticker_df["open"],
                    high=ticker_df["high"],
                    low=ticker_df["low"],
                    close=ticker_df["close"],
                    window=kc_params["window"],
                    window_atr=kc_params["window_atr"],
                    window_mult=kc_params["window_mult"],
                    fillna=True,
                )
                ticker_df["kc_mband"] = kc.keltner_channel_mband()
                ticker_df["kc_hband"] = kc.keltner_channel_hband()
                ticker_df["kc_lband"] = kc.keltner_channel_lband()
                ticker_df["kc_wband"] = kc.keltner_channel_wband()
                ticker_df["kc_pband"] = kc.keltner_channel_pband()
            else:
                print(f"Not enough data for {ticker} to calculate Keltner Channel with parameters: {kc_params}. Filling with NaN.")
                ticker_df["kc_mband"] = np.nan
                ticker_df["kc_hband"] = np.nan
                ticker_df["kc_lband"] = np.nan
                ticker_df["kc_wband"] = np.nan
                ticker_df["kc_pband"] = np.nan
            
            keltner_data.append(ticker_df[["date", "tic", "kc_mband", "kc_hband", "kc_lband", "kc_wband", "kc_pband"]])
        
        if keltner_data:
            keltner_df = pd.concat(keltner_data, ignore_index=True)
            df = df.merge(keltner_df, on=["date", "tic"], how="left")
        
        df = df.sort_values(by=["date", "tic"])
        return df


class CustomStockPortfolioEnv(FinRLStockPortfolioEnv): # Changed base class
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
        minimum_acceptable_return: float = 0.0, # Added for Sortino-like reward
    ):
        super().__init__( # Call super().__init__
            df,
            stock_dim,
            hmax,
            initial_amount,
            transaction_cost_pct,
            reward_scaling,
            state_space,
            action_space,
            tech_indicator_list,
            turbulence_threshold,
            lookback,
            day,
        )
        self.minimum_acceptable_return = minimum_acceptable_return # Store MAR

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

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
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            
            # Calculate Sortino Ratio at terminal state
            if not df_daily_return.empty and df_daily_return["daily_return"].std() != 0:
                # Using the Sortino logic from src/hyperopt_loss_sortino_daily.py
                resample_freq = "1D" # Assuming daily returns
                days_in_year = 365
                
                # Reindex to ensure all days are present for accurate calculation
                min_date = self.date_memory[0]
                max_date = self.date_memory[-1]
                t_index = pd.date_range(start=min_date, end=max_date, freq=resample_freq, normalize=True)
                
                # Create a temporary DataFrame for daily returns with proper date index
                temp_returns_df = pd.DataFrame({
                    "date": self.date_memory,
                    "daily_return": self.portfolio_return_memory
                }).set_index("date")
                
                sum_daily = temp_returns_df.resample(resample_freq).agg({"daily_return": "sum"}).reindex(t_index).fillna(0)
                
                total_profit = sum_daily["daily_return"] - self.minimum_acceptable_return
                expected_returns_mean = total_profit.mean()
                
                sum_daily["downside_returns"] = 0.0
                sum_daily.loc[total_profit < 0, "downside_returns"] = total_profit
                total_downside = sum_daily["downside_returns"]
                
                down_stdev = 0.0
                if len(total_downside) > 1: # Ensure enough data for std dev
                    down_stdev = math.sqrt((total_downside**2).sum() / len(total_downside))

                if down_stdev != 0:
                    sortino_ratio = expected_returns_mean / down_stdev * math.sqrt(days_in_year)
                    print("Sortino Ratio: ", sortino_ratio)
                else:
                    print("Sortino Ratio: N/A (downside standard deviation is zero)")
            else:
                print("Sortino Ratio: N/A (not enough data or zero standard deviation)")
            print("=================================")

            return self.state, self.reward, self.terminal, False, {}

        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) /
            #                   (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            # print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # The reward is now Sortino-like
            excess_return = portfolio_return - self.minimum_acceptable_return
            # A simple penalty for negative excess returns.
            # The magnitude of this penalty can be tuned.
            # For now, let's make negative excess returns twice as bad.
            if excess_return < 0:
                self.reward = excess_return * 2.0
            else:
                self.reward = excess_return
            
            self.reward *= self.reward_scaling # Apply overall reward scaling

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
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
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed) # seeding needs to be imported
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


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
    fe = CustomFeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)
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

def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)

def train_agent(train_df, env_kwargs, model_params, model_save_path):
    """Trains the DRL agent and saves the model."""
    if train_df.empty:
        print("Training data is empty. Skipping training.")
        return None

    e_train_gym = CustomStockPortfolioEnv(df=train_df, **env_kwargs) # Changed to CustomStockPortfolioEnv
    env_train, _ = e_train_gym.get_sb_env()
    
    agent = FinRLDRLAgent(env=env_train) # Changed to FinRLDRLAgent
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

    # Use the CustomStockPortfolioEnv for backtesting as well
    e_trade_gym = CustomStockPortfolioEnv(df=trade_df, **env_kwargs)
    
    try:
        # CORRECTED LINE: Load the model using the SAC class directly
        trained_model = SAC.load(model_path)
        print("Trained model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return pd.DataFrame(), pd.DataFrame()
        
    df_daily_return, df_actions = FinRLDRLAgent.DRL_prediction(model=trained_model, environment=e_trade_gym) # Changed to FinRLDRLAgent
    return df_daily_return, df_actions

def calculate_min_variance_portfolio(trade_df, initial_capital=1000):
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

    # Get the full list of technical indicators including the new KC ones
    full_tech_indicator_list = CustomFeatureEngineer().tech_indicator_list

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": full_tech_indicator_list, # Use the extended list
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "minimum_acceptable_return": 0.0, # Set MAR for Sortino-like reward
    }
    # --- END OF FIX ---

    print("--- Starting Agent Training ---")
    # Use the CustomStockPortfolioEnv
    e_train_gym = CustomStockPortfolioEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    agent = FinRLDRLAgent(env=env_train) # Changed to FinRLDRLAgent
    model = agent.get_model("sac", model_kwargs=SAC_PARAMS)
    
    log_path = os.path.join(TENSORBOARD_LOG_DIR, "sac")
    new_logger = configure(log_path, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    
    trained_model = model.learn(total_timesteps=5000) # Increased timesteps for better learning
    trained_model.save(TRAINED_MODEL_PATH)
    # train_agent(train_df, env_kwargs, SAC_PARAMS, TRAINED_MODEL_PATH) # Original call
    print("--- Agent Training Finished ---")

    print("--- Starting Backtesting ---")
    drl_daily_return, _ = run_backtest(trade_df, env_kwargs, TRAINED_MODEL_PATH)
    if not drl_daily_return.empty:
        drl_daily_return.to_csv(os.path.join(RESULTS_DIR, 'drl_daily_return.csv'))
    print("--- Backtesting Finished ---")

    print("--- Analyzing Performance ---")
    drl_strat_returns = convert_daily_return_to_pyfolio_ts(drl_daily_return)

    baseline_df = FinRLGetBaseline(ticker=BASELINE_TICKER, start=TRADE_START_DATE, end=TRADE_END_DATE) # Changed to FinRLGetBaseline
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
