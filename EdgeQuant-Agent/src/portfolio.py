import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List

import orjson
from pydantic import BaseModel


# ========================= ENUM ========================= #

class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


# ========================= DATA STRUCT ========================= #

@dataclass
class PortfolioState:
    date: date
    cash: float
    positions: Dict[str, int]
    portfolio_value: float


class MultiPortfolioDump(BaseModel):
    symbols: List[str]
    buying_power: float
    trading_dates: List[date]
    trading_price: Dict[str, List[float]]
    portfolio_value: List[float]
    cur_portfolio_shares: Dict[str, float]
    asset_value: Dict[str, List[float]]  # ✅ NEW
    portfolio_config: Dict


# ========================= BASE ========================= #

class PortfolioBase(ABC):
    @abstractmethod
    def record_action(self, *args, **kwargs):
        pass


# ========================= MAIN CLASS ========================= #

class PortfolioMultiAsset(PortfolioBase):
    def __init__(self, portfolio_config=None):
        self.trading_symbols = portfolio_config["trading_symbols"]
        self.buying_power = portfolio_config["cash"]
        self.portfolio_config = portfolio_config

        self.trading_dates = []
        self.trading_price = {s: [] for s in self.trading_symbols}
        self.portfolio_value = []

        # weights for next day move (1=long, -1=short, 0=flat)
        self.current_weights = {s: 0 for s in self.trading_symbols}

        # per-asset tracking
        self.asset_value = {s: [] for s in self.trading_symbols}

        self.asset_cash = {
            s: self.buying_power / len(self.trading_symbols)
            for s in self.trading_symbols
        }

        self.asset_shares = {s: 0.0 for s in self.trading_symbols}

    # ========================= CORE LOGIC ========================= #

    def record_action(self, action_date, action, price_info, evidence):
        self.trading_dates.append(list(action_date.values())[0])
    
        # Record daily prices
        for s in price_info:
            self.trading_price[s].append(price_info[s])
    
        # ================= TASK 3 TRADING LOGIC ================= #
        # Rule: BUY -> LONG, HOLD -> FLAT, SELL -> SHORT
        # Rule: Each new action fully replaces the previous day's position.
    
        total_portfolio_value = 0
    
        for s in self.trading_symbols:
            current_price = price_info[s]
            
            # Initial state or first step handling
            if not self.asset_value[s]:
                # Initialize with allocated cash
                prev_value = self.asset_cash[s]
                prev_price = current_price
                prev_action = "hold"
            else:
                prev_value = self.asset_value[s][-1]
                prev_price = self.trading_price[s][-2]
                # We need to know what the previous action was to calculate profit/loss
            # A cleaner way using weights:
            # V_t = V_{t-1} * (1 + return * weight_{t-1})
            
            # 1. Update value based on PREVIOUS day's weight and price move
            if len(self.trading_price[s]) > 1:
                p_prev = self.trading_price[s][-2]
                p_curr = self.trading_price[s][-1]
                daily_return = (p_curr - p_prev) / p_prev
                
                # Weight from yesterday applied to today's move
                weight = self.current_weights[s]
                multiplier = 1 + (weight * daily_return)
                new_asset_value = prev_value * multiplier
            else:
                new_asset_value = self.asset_cash[s] # Initial allocation

            # 2. Record this day's ending value
            self.asset_value[s].append(new_asset_value)
            
            # 3. SET NEW WEIGHT for the NEXT day's move based on current action
            decision = action[s].value.lower()
            if decision == "buy":
                self.current_weights[s] = 1   # LONG
            elif decision == "sell":
                self.current_weights[s] = -1  # SHORT
            else:
                self.current_weights[s] = 0   # FLAT
                
            total_portfolio_value += new_asset_value

        self.portfolio_value.append(total_portfolio_value)
        self.buying_power = total_portfolio_value

    # ========================= OUTPUT ========================= #

    def get_action_record(self):
        return {
            "date": self.trading_dates,
            "price": self.portfolio_value,
            "symbol": self.trading_symbols,
            "position": [1] * len(self.trading_dates),
        }

    # ========================= SAVE ========================= #

    def save_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "portfolio.pkl"), "wb") as f:
            pickle.dump(self, f)

    # ========================= LOAD ========================= #

    @classmethod
    def load_checkpoint(cls, path: str):
        file_path = os.path.join(path, "portfolio.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Portfolio checkpoint not found at {file_path}")

        with open(file_path, "rb") as f:
            portfolio = pickle.load(f)

        return portfolio


# ========================= FACTORY ========================= #

def construct_portfolio(portfolio_config):
    if portfolio_config["type"] == "multi-assets":
        return PortfolioMultiAsset(portfolio_config=portfolio_config)
    else:
        raise NotImplementedError(
            f"Portfolio type {portfolio_config['type']} not supported"
        )