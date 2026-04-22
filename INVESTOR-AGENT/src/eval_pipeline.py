# import json
# import os
# from datetime import datetime
# from typing import Dict, List, Tuple

# import numpy as np
# import pandas as pd
# from rich import print

# from .agent import FinMemAgent
# from .portfolio import PortfolioMultiAsset


# # ========================= SAFE FUNCTIONS ========================= #

# def safe_std(x):
#     x = np.nan_to_num(x)
#     std = np.std(x)
#     return std if std != 0 else 1e-8


# def safe_returns(price_list):
#     price_list = np.array(price_list)
#     returns = np.diff(price_list) / price_list[:-1]
#     return np.nan_to_num(returns)


# # ========================= METRIC FUNCTIONS ========================= #

# def calculate_sharpe_ratio(Rp, Rf, sigma_p, price_list, trading_days=252):
#     if sigma_p == 0:
#         return 0.0
#     Rp = Rp / (len(price_list) / trading_days)
#     return (Rp - Rf) / sigma_p


# def calculate_max_drawdown(daily_returns):
#     cumulative = np.cumprod([1 + r for r in daily_returns])
#     peak = np.maximum.accumulate(cumulative)
#     drawdown = (peak - cumulative) / peak
#     return np.max(drawdown)


# def calculate_metrics(price_list):
#     returns = safe_returns(price_list)

#     cum_ret = (price_list[-1] - price_list[0]) / price_list[0]
#     std = safe_std(returns)
#     ann_vol = std * np.sqrt(252)
#     sharpe = calculate_sharpe_ratio(cum_ret, 0, ann_vol, price_list)
#     mdd = calculate_max_drawdown(returns)

#     return cum_ret, sharpe, mdd, ann_vol


# # ========================= MAIN FUNCTION ========================= #

# def output_metric_summary_multi(
#     trading_symbols: List[str],
#     data_root_path: Dict[str, str],
#     output_path: str,
#     result_path: str,
#     model_name
# ) -> None:

#     print("\n==================== 📊 INDIVIDUAL ASSET METRICS ====================\n")

#     # ✅ INDIVIDUAL METRICS (BTC, TSLA)
#     for symbol in trading_symbols:
#         with open(data_root_path[symbol], "r", encoding = "utf-8") as f:
#             data = json.load(f)

#         prices = []
#         for d, content in data.items():
#             if content and "prices" in content:
#                 prices.append(float(content["prices"]))

#         if len(prices) < 2:
#             print(f"{symbol}: Not enough data")
#             continue

#         prices = np.array(prices)

#         cum_ret, sharpe, mdd, vol = calculate_metrics(prices)

#         print(f"🔹 {symbol}")
#         print(f"   Cumulative Return: {cum_ret:.4f}")
#         print(f"   Sharpe Ratio: {sharpe:.4f}")
#         print(f"   Max Drawdown: {mdd:.4f}")
#         print(f"   Volatility: {vol:.4f}\n")

#     # ========================= PORTFOLIO ========================= #

#     print("\n==================== 📊 PORTFOLIO METRICS ====================\n")

#     portfolio = PortfolioMultiAsset.load_checkpoint(
#         os.path.join(result_path, "agent")
#     )

#     records = portfolio.get_action_record()
#     portfolio_value = np.array(records["price"])

#     portfolio_value = np.nan_to_num(portfolio_value)

#     cum_ret, sharpe, mdd, vol = calculate_metrics(portfolio_value)

#     # ========================= EQUAL WEIGHT ========================= #

#     price_dict = portfolio.trading_price

#     min_len = min(len(v) for v in price_dict.values())
#     for s in price_dict:
#         price_dict[s] = price_dict[s][-min_len:]

#     equal_weight = np.mean(np.array(list(price_dict.values())), axis=0)

#     eq_cum_ret, eq_sharpe, eq_mdd, eq_vol = calculate_metrics(equal_weight)

#     # ========================= FINAL TABLE ========================= #

#     df = pd.DataFrame(
#         {
#             "Metric": [
#                 "Cumulative Return",
#                 "Sharpe Ratio",
#                 "Max Drawdown",
#                 "Annualized Volatility",
#             ],
#             "Equal Weight Portfolio": [
#                 eq_cum_ret,
#                 eq_sharpe,
#                 eq_mdd,
#                 eq_vol,
#             ],
#             "Agent Portfolio": [
#                 cum_ret,
#                 sharpe,
#                 mdd,
#                 vol,
#             ],
#         }
#     )

#     print(df.to_markdown(index=False))

#     # ================= SAVE RESULTS ================= #

#     os.makedirs(output_path, exist_ok=True)
    
#     save_file = os.path.join(output_path, "evaluation_results.txt")
    
#     with open(save_file, "w", encoding="utf-8") as f:

#         # 🔹 Model info (if available)
#         if model_name:
#             f.write("MODEL INFO\n")
#             f.write("----------------------\n")
#             f.write(f"Model Used: {model_name}\n\n")
#         f.write("INDIVIDUAL ASSET METRICS\n\n")
    
#         for symbol in trading_symbols:
#             f.write(f"{symbol}\n")
#             f.write("----------------------\n")
    
#             with open(data_root_path[symbol], "r", encoding="utf-8") as data_file:
#                 data = json.load(data_file)
    
#             prices = [
#                 float(v["prices"])
#                 for v in data.values()
#                 if v and "prices" in v
#             ]
    
#             if len(prices) < 2:
#                 continue
            
#             prices = np.array(prices)
    
#             cum_ret, sharpe, mdd, vol = calculate_metrics(prices)
    
#             f.write(f"Cumulative Return: {cum_ret:.4f}\n")
#             f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
#             f.write(f"Max Drawdown: {mdd:.4f}\n")
#             f.write(f"Volatility: {vol:.4f}\n\n")
    
#         f.write("\nPORTFOLIO METRICS\n\n")
#         f.write(df.to_string(index=False))
    
#     print(f"\n✅ Results saved at: {save_file}")

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from rich import print

from .portfolio import PortfolioMultiAsset


# ========================= SAFE FUNCTIONS ========================= #

def safe_std(x):
    x = np.nan_to_num(x)
    std = np.std(x)
    return std if std != 0 else 1e-8


def safe_returns(price_list):
    price_list = np.array(price_list)
    returns = np.diff(price_list) / price_list[:-1]
    return np.nan_to_num(returns)


# ========================= METRIC FUNCTIONS ========================= #

def calculate_sharpe_ratio(Rp, Rf, sigma_p, price_list, trading_days=252):
    if sigma_p == 0:
        return 0.0
    Rp = Rp / (len(price_list) / trading_days)
    return (Rp - Rf) / sigma_p


def calculate_max_drawdown(daily_returns):
    cumulative = np.cumprod([1 + r for r in daily_returns])
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown)


def calculate_metrics(price_list):

    price_list = np.array(price_list)

    if len(price_list) < 2:
        return 0, 0, 0, 0

    returns = np.diff(price_list) / price_list[:-1]

    returns = np.nan_to_num(returns)

    if np.std(returns) == 0:
        return 0, 0, 0, 0

    cum_ret = (price_list[-1] - price_list[0]) / price_list[0]
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    mdd = calculate_max_drawdown(returns)

    return cum_ret, sharpe, mdd, ann_vol

# ========================= MAIN FUNCTION ========================= #

def output_metric_summary_multi(
    trading_symbols: List[str],
    data_root_path: Dict[str, str],
    output_path: str,
    result_path: str,
    model_name=None
) -> None:

    print("\n==================== 🤖 MODEL INFO ====================")
    print(f"Model Used: {model_name}\n")

    print("==================== 📊 INDIVIDUAL ASSET METRICS ====================\n")

    # ✅ LOAD PORTFOLIO (IMPORTANT FIX)
    portfolio = PortfolioMultiAsset.load_checkpoint(
        os.path.join(result_path, "agent")
    )

    # ================= INDIVIDUAL METRICS ================= #

    asset_metrics = {}

    for symbol in trading_symbols:

        prices = portfolio.asset_value.get(symbol, [])

        if len(prices) < 2:
            print(f"{symbol}: Not enough data")
            continue

        prices = np.array(prices)

        cum_ret, sharpe, mdd, vol = calculate_metrics(prices)

        asset_metrics[symbol] = (cum_ret, sharpe, mdd, vol)

        print(f"🔹 {symbol}")
        print(f"   Cumulative Return: {cum_ret:.4f}")
        print(f"   Sharpe Ratio: {sharpe:.4f}")
        print(f"   Max Drawdown: {mdd:.4f}")
        print(f"   Volatility: {vol:.4f}\n")

    # ================= PORTFOLIO METRICS ================= #

    print("\n==================== 📊 PORTFOLIO METRICS ====================\n")

    records = portfolio.get_action_record()
    portfolio_value = np.array(records["price"])
    portfolio_value = np.nan_to_num(portfolio_value)

    cum_ret, sharpe, mdd, vol = calculate_metrics(portfolio_value)

    # ================= EQUAL WEIGHT ================= #

    price_dict = portfolio.trading_price

    min_len = min(len(v) for v in price_dict.values())

    for s in price_dict:
        price_dict[s] = price_dict[s][-min_len:]

    equal_weight = np.mean(np.array(list(price_dict.values())), axis=0)

    eq_cum_ret, eq_sharpe, eq_mdd, eq_vol = calculate_metrics(equal_weight)

    # ================= FINAL TABLE ================= #

    df = pd.DataFrame(
        {
            "Metric": [
                "Cumulative Return",
                "Sharpe Ratio",
                "Max Drawdown",
                "Annualized Volatility",
            ],
            "Equal Weight Portfolio": [
                eq_cum_ret,
                eq_sharpe,
                eq_mdd,
                eq_vol,
            ],
            "Agent Portfolio": [
                cum_ret,
                sharpe,
                mdd,
                vol,
            ],
        }
    )

    print(df.to_markdown(index=False))

    # ================= SAVE RESULTS ================= #

    os.makedirs(output_path, exist_ok=True)
    save_file = os.path.join(output_path, "evaluation_results.txt")

    with open(save_file, "w", encoding="utf-8") as f:

        # 🔹 MODEL INFO
        f.write("MODEL INFO\n")
        f.write("----------------------\n")
        f.write(f"Model Used: {model_name}\n\n")

        # 🔹 INDIVIDUAL METRICS
        f.write("INDIVIDUAL ASSET METRICS\n\n")

        for symbol, metrics in asset_metrics.items():
            cum_ret, sharpe, mdd, vol = metrics

            f.write(f"{symbol}\n")
            f.write("----------------------\n")
            f.write(f"Cumulative Return: {cum_ret:.4f}\n")
            f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
            f.write(f"Max Drawdown: {mdd:.4f}\n")
            f.write(f"Volatility: {vol:.4f}\n\n")

        # 🔹 PORTFOLIO METRICS
        f.write("\nPORTFOLIO METRICS\n\n")
        f.write(df.to_string(index=False))

    print(f"\n✅ Results saved at: {save_file}")

def output_metrics_summary_single(
    start_date: str,
    end_date: str,
    ticker: str,
    output_path: str,
    data_path: str,
    result_path: str,
) -> None:

    print(f"\n📊 SINGLE ASSET METRICS: {ticker}\n")

    with open(data_path, "r", encoding = "utf-8") as f:
        data = json.load(f)

    prices = []
    for d, content in data.items():
        if content and "prices" in content:
            prices.append(float(content["prices"]))

    if len(prices) < 2:
        print("Not enough data")
        return

    prices = np.array(prices)

    cum_ret, sharpe, mdd, vol = calculate_metrics(prices)

    print(f"Cumulative Return: {cum_ret:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {mdd:.4f}")
    print(f"Volatility: {vol:.4f}")

    