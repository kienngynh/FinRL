"""
SortinoHyperOptLossDaily

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.

FINAL VERSION v2: This version is fully automatic and balances three objectives:
1. High Sortino Ratio (performance).
2. Sufficient trade count (frequency), scaled by timeframe.
3. Low maximum drawdown (safety).
"""

import math
import re
from datetime import datetime
from pandas import DataFrame, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss
from freqtrade.data.metrics import calculate_max_drawdown


# --- CẤU HÌNH CƠ SỞ ---
# 1. Timeframe "chuẩn" và số trade tối thiểu tương ứng.
BASELINE_TIMEFRAME = '15m'
BASELINE_TRADES_PER_YEAR = 365

# --- CẤU HÌNH PHẠT DRAWDOWN ---
# Trọng số cho khoản phạt drawdown.
# > 1.0: Phạt nặng hơn.
# < 1.0: Phạt nhẹ hơn.
# 1.0: Phạt tuyến tính.
DRAWDOWN_PENALTY_WEIGHT = 1.0
# --------------------


class SortinoHyperOptLossDaily(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.
    This implementation uses a fully automatic, scaling penalty for both
    low trade counts and high maximum drawdown.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        starting_balance: float,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for more optimal results.
        """
        # --- Trade Count Penalty Logic ---
        def timeframe_to_minutes(timeframe: str) -> int:
            try:
                value = int(re.sub(r'\D', '', timeframe))
                unit = re.sub(r'\d', '', timeframe)
                if unit == 'm': return value
                if unit == 'h': return value * 60
                if unit == 'd': return value * 1440
                if unit == 'w': return value * 10080
            except:
                return 15

        current_timeframe = kwargs.get('timeframe', BASELINE_TIMEFRAME)
        baseline_minutes = timeframe_to_minutes(BASELINE_TIMEFRAME)
        current_minutes = timeframe_to_minutes(current_timeframe)

        if baseline_minutes > 0:
            scaling_factor = current_minutes / baseline_minutes
            min_trades_per_year = BASELINE_TRADES_PER_YEAR / scaling_factor
        else:
            min_trades_per_year = BASELINE_TRADES_PER_YEAR

        min_trades_per_year = max(1, min_trades_per_year)
        duration_years = (max_date - min_date).days / 365.25
        target_total_trades = min_trades_per_year * duration_years
        trade_penalty = 1.0
        if trade_count < target_total_trades and target_total_trades > 0:
            trade_penalty = trade_count / target_total_trades

        # --- Tính toán và áp dụng phạt Drawdown ---
        drawdown_penalty = 1.0
        try:
            drawdown_result = calculate_max_drawdown(
                results, value_col='profit_abs', starting_balance=starting_balance
            )
            # Lấy giá trị sụt giảm vốn tương đối (ví dụ: 0.2 cho 20%)
            relative_drawdown = drawdown_result.relative_account_drawdown
            # Công thức phạt: (1 - drawdown)^weight.
            # Drawdown càng lớn, hệ số phạt càng nhỏ (tiến về 0).
            drawdown_penalty = (1 - relative_drawdown) ** DRAWDOWN_PENALTY_WEIGHT
        except (ValueError, KeyError):
            # Xảy ra lỗi (ví dụ: không có giao dịch thua lỗ) -> không phạt
            drawdown_penalty = 1.0

        # --- Original Sortino Ratio Calculation ---
        resample_freq = "1D"
        slippage_per_trade_ratio = 0.0005
        days_in_year = 365
        minimum_acceptable_return = 0.0

        results.loc[:, "profit_ratio_after_slippage"] = (
            results["profit_ratio"] - slippage_per_trade_ratio
        )
        t_index = date_range(start=min_date, end=max_date, freq=resample_freq, normalize=True)
        sum_daily = (
            results.resample(resample_freq, on="close_date")
            .agg({"profit_ratio_after_slippage": "sum"})
            .reindex(t_index)
            .fillna(0)
        )
        total_profit = sum_daily["profit_ratio_after_slippage"] - minimum_acceptable_return
        expected_returns_mean = total_profit.mean()
        sum_daily["downside_returns"] = 0.0
        sum_daily.loc[total_profit < 0, "downside_returns"] = total_profit
        total_downside = sum_daily["downside_returns"]
        down_stdev = math.sqrt((total_downside**2).sum() / len(total_downside))

        if down_stdev != 0:
            sortino_ratio = expected_returns_mean / down_stdev * math.sqrt(days_in_year)
        else:
            sortino_ratio = -20.0

        # --- Apply All Penalties to Final Loss ---
        # Nhân tất cả các yếu tố lại với nhau
        return -sortino_ratio * trade_penalty * drawdown_penalty
