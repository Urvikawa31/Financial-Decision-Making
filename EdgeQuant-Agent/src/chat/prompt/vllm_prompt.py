from datetime import date, timedelta
from typing import Dict, List, Union

from ...utils import RunMode
from .base import MultiAssetBasePromptConstructor, SingleAssetBasePromptConstructor


# prompt construction
def _add_momentum_info(momentum: int, investment_info: str) -> str:
    mapping = {-1: "bearish", 0: "neutral", 1: "bullish"}
    term = mapping.get(momentum, "unknown")
    investment_info += f"The current market momentum for this asset is {term}. "
    if momentum == -1:
        investment_info += "This reflects a negative trend over the past window."
    elif momentum == 1:
        investment_info += "This reflects a positive trend over the past window."
    return investment_info


# stock
# memory layer + id
stock_short_memory_id_desc = "The id of the short-term information."
stock_mid_memory_id_desc = "The id of the mid-term information."
stock_long_memory_id_desc = "The id of the long-term information."
stock_reflection_memory_id_desc = "The id of the reflection-term information."

# prefix
stock_warmup_investment_info_prefix = """Today's price of {symbol} is ${cur_price:.2f} (Today: {day_of_week}, Tomorrow: {tomorrow_day}).
The historical prices of {symbol} are {historical_prices}.
The GROUND TRUTH MANDATE for tomorrow is: {ground_truth_mandate}.
Your final investment decision MUST align with this mandate.
However, in Step 1 of your reasoning, you must evaluate today's trend objectively using only the provided trend signal, without confusing it with tomorrow's mandate."""
stock_test_investment_info_prefix = """The current date is {cur_date} (Today: {day_of_week}, Tomorrow: {tomorrow_day}). Here are the observed financial market facts:
For {symbol}, the current price is ${cur_price:.2f}.
Historical Prices (last 10 days): {historical_prices}

"""

# action
stock_momentum_explanation = """#### 📊 MOMENTUM DYNAMICS:
In the high-volatility equity market, we treat momentum as a confirmation signal. High Magnitude Catalysts (Tier 1) can override momentum filters, but Tactical Catalysts (Tier 3) require momentum alignment for execution."""

# Institutional Risk/Reward Framework
stock_sentiment_explanation = """#### 🔍 RESEARCH GUIDELINES:
1. **CATALYST FRESHNESS**: Your primary 'Catalyst' MUST be derived from the 'Short-term (TODAY'S CRITICAL NEWS)' block. Do not recycle catalysts from the 'PREVIOUS THESIS' block unless today's news specifically updates them.
2. **CATALYST MAGNITUDE (M)**: Rate 1-5. (5 = Needle-mover like Q4 Deliveries; 1 = Secondary sentiment).
3. **EXPECTATION VARIANCE (V)**: Is this news a **Positive/Negative Surprise** relative to the 10-day price trend?
4. **BULL/BEAR TENSION**: If news is mixed, pivot to the catalyst with the higher 'Cumulative Cash Flow' impact.

#### 🏗️ TIER CLASSIFICATION:
- **TIER 1 (Structural)**: Unit-delivery misses/beats, Margin pivots, Structural regulatory shifts.
- **TIER 2 (Competitive)**: Market share shifts (BYD/NIO check), Pricing wars.
- **TIER 3 (Tactical)**: SEC noise, secondary macro signals."""

stock_test_action_choice = "Given the information, please make an investment decision: buy, or sell, or hold the each of the stocks in {trading_symbols}."

# final prompt
stock_warmup_final_prompt = """### STRATEGY EXTRACTION (WARMUP MODE):
Analyze the provided catalyst and its corresponding price action to define a **High-Conviction Trading Rule**.

### 🛠️ LOGICAL CONSTRAINTS:
1.  **CATALYST TYPE**: Classify the news as **Structural** (Delivery variance, Net Income, M&A) or **Tactical** (Price cuts, Sentiment).
2.  **IMPACT GRADIENT**: Explain why the catalyst was strong enough to drive the move.
3.  **PATTERN RULE**: Formulate a rule (e.g., "Unit delivery misses > 5% override positive macro sentiment").

### OUTPUT FORMAT (JSON):
{{
  "investment_decision": "BUY/SELL/HOLD",
  "reasoning": "Rule: [If X then Y] | Evidence: [News catalyst vs Price response]."
}}"""

stock_test_final_prompt = """### 🏛️ INSTITUTIONAL RESEARCH COMMITTEE (TEST MODE):
State your ALPHA-driven directional stance for {symbol}.

### 🕵️ INVESTMENT CASE:
1. **PRIMARY DRIVER**: Isolate the news catalyst with the highest TIER rating.
2. **MAGNITUDE CHECK**: Rate the catalyst impact (1-5).
3. **VARIANCE CHECK**: Does the catalyst represent a trend-reversal surprise?
4. **EXECUTION**: Select a decisive action. **Avoid neutral 'waiting' unless catalysts are perfectly flat (Magnitude < 2).**

### 📋 COMPLIANCE:
- **LIQUIDITY**: Assume full fill at session close.
- **CONVICTION**: You are managing a $100M book. **Capture alpha aggressively.**

### OUTPUT FORMAT (JSON):
{{
  "investment_decision": "{allowed_decisions}",
  "reasoning": "[TIER X] Magnitude: [X/5] | Variance: [Surprise vs Trend] | Catalyst: [One-sentence impact rule] | Conviction: [High/Medium]."
}}"""



# crypto
# memory layer + id
crypto_short_memory_id_desc = "The id of the short-term information."
crypto_mid_memory_id_desc = "The id of the mid-term information."
crypto_long_memory_id_desc = "The id of the long-term information."
crypto_reflection_memory_id_desc = "The id of the reflection-term information."

# prefix
crypto_warmup_investment_info_prefix = """Today's price of {symbol} is ${cur_price:.2f} (Today: {day_of_week}, Tomorrow: {tomorrow_day}).
The historical prices of {symbol} are {historical_prices}.
The GROUND TRUTH MANDATE for tomorrow is: {ground_truth_mandate}.
Your final investment decision MUST align with this mandate.
However, in Step 1 of your reasoning, you must evaluate today's trend objectively using only the provided trend signal, without confusing it with tomorrow's mandate."""
crypto_test_investment_info_prefix = """The current date is {cur_date} (Today: {day_of_week}, Tomorrow: {tomorrow_day}). Here are the observed financial market facts:
For {symbol}, the current price is ${cur_price:.2f}.
Historical Prices (last 10 days): {historical_prices}

"""

# Institutional Research Guidelines (Crypto)
crypto_sentiment_explanation = """Institutional crypto analysis focuses on 'Trust Variance' and 'Liquidity Flows'. We prioritize news regarding ETF net-inflows, corporate treasury allocations (custody), and network-level security resilience. We filter out retail-driven social media hype in favor of structural adoption metrics."""
crypto_momentum_explanation = """In the 24/7 crypto market, momentum serves as a proxy for institutional liquidity clusters. We use momentum to identify points of 'Forceful Accumulation' or 'Capitulation' that confirm fundamental network shifts."""

# summary
crypto_warmup_reason = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
crypto_test_reason = "Given the information of text and the summary of the cryptocurrency price movement. Please explain the reason why you make the investment decision."

# action
crypto_test_action_choice = "Given the information, please make an investment decision: buy the cryptocurrency, sell, and hold the cryptocurrency"

# final prompt
crypto_warmup_final_prompt = """### STRATEGY EXTRACTION (BITCOIN):
Analyze the provided catalyst and price action to define an **Institutional Liquidity Rule**.

### 🛠️ LOGICAL CONSTRAINTS:
1.  **LIQUIDITY TYPE**: Structural (ETF inflows/custody) or Psychological (FOMO/Fear).
2.  **TRUST VARIANCE**: Does the news improve or damage the network's perceived security?
3.  **PATTERN RULE**: Formulate a rule (e.g., "Institutional ETF accumulation overrides short-term exchange outflows").

### OUTPUT FORMAT (JSON):
{{
  "investment_decision": "BUY/SELL",
  "reasoning": "Rule: [If X then Y] | Evidence: [News catalyst vs Bitcoin response]."
}}"""


crypto_test_final_prompt = """### 🪙 CRYPTO LIQUIDITY COMMITTEE (BITCOIN):
Identify 'Forceful Accumulation' or 'Structural Capitulation'.

### 🕵️ ANALYSIS STEPS:
1. **TRUST VARIANCE**: Does the news strengthen or weaken the network's 'Safe Haven' mandate?
2. **LIQUIDITY FLOWS**: Identify ETF/Treasury movements (The 'Big Money' signal).
3. **MAGNITUDE**: Is this catalyst structural (ETF Inflow) or retail noise (Social Media)?
4. **DIRECTIONAL BLIAS**: Bitcoin rewards directional volatility. **Eliminate 'HOLD' bias.**

### 📋 COMPLIANCE:
- **DECISIVENESS**: You MUST capture alpha. **Passive HOLDING is for retail users, not professionals.**

### OUTPUT FORMAT (JSON):
{{
  "investment_decision": "{allowed_decisions}",
  "reasoning": "[Catalyst Type] | Net Flow Impact: [Positive/Negative] | Magnitude: [X/5] | Trust Score: [Current Delta]."
}}"""


class SingleAssetVLLMPromptConstructor(SingleAssetBasePromptConstructor):
    @staticmethod
    def __call__(
        cur_date: date,
        symbol: str,
        run_mode: RunMode,
        future_record: Union[float, None],
        short_memory: Union[List[str], None],
        short_memory_id: Union[List[int], None],
        mid_memory: Union[List[str], None],
        mid_memory_id: Union[List[int], None],
        long_memory: Union[List[str], None],
        long_memory_id: Union[List[int], None],
        reflection_memory: Union[List[str], None],
        reflection_memory_id: Union[List[int], None],
        momentum: Union[int, None] = None,
        character_string: str = "",
        cur_price: float = 0.0,
        history_prices: List[float] = None,
    ) -> str:
        
        # Inject persona
        prefix_persona = f"### YOUR ROLE:\n{character_string}\n\n" if character_string else ""
        
        # Day of the Week calculation
        day_of_week = cur_date.strftime("%A")
        tomorrow_day = (cur_date + timedelta(days=1)).strftime("%A")

        if symbol in {"TSLA"}:
            asset_type = "stock"
        elif symbol in {"BTC"}:
            asset_type = "crypto"
        else:
            # Default to stock logic for unknown symbols
            asset_type = "stock"

        # Format historical prices for the prompt
        historical_prices_text = ", ".join([f"${p:.2f}" for p in history_prices]) if history_prices else "N/A"

        def prune(ids, texts, limit=3):
            if texts and ids:
                return ids[:limit], [t[:1500] + "..." if len(t) > 1500 else t for t in texts[:limit]]
            return [], []

        if asset_type == "stock":
            # investment info + memories
            investment_info = prefix_persona + (
                stock_warmup_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date, 
                    day_of_week=day_of_week, tomorrow_day=tomorrow_day,
                    ground_truth_mandate=future_record,
                    cur_price=cur_price, historical_prices=historical_prices_text
                )
                if run_mode == RunMode.WARMUP
                else stock_test_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date, 
                    day_of_week=day_of_week, tomorrow_day=tomorrow_day,
                    cur_price=cur_price, historical_prices=historical_prices_text
                )
            )
            
            # Prune and add memories symmetrically - PRIORITY: Short-term > Mid-term > Long-term > Reflection
            if short_memory:
                p_id, p_txt = prune(short_memory_id, short_memory, limit=3)
                investment_info += "Short-term (TODAY'S CRITICAL NEWS):\n" + "\n".join([f"{i}. {t}" for i, t in zip(p_id, p_txt)]) + "\n\n"
                investment_info += stock_sentiment_explanation

            if mid_memory:
                p_id, p_txt = prune(mid_memory_id, mid_memory, limit=2)
                investment_info += "Mid-term (Recent):\n" + "\n".join([f"{i}. {t}" for i, t in zip(p_id, p_txt)]) + "\n\n"
            
            if long_memory:
                p_id, p_txt = prune(long_memory_id, long_memory, limit=1)
                investment_info += "Long-term (Context):\n" + "\n".join([f"{i}. {t}" for i, t in zip(p_id, p_txt)]) + "\n\n"

            if reflection_memory:
                p_id, p_txt = prune(reflection_memory_id, reflection_memory, limit=2)
                investment_info += "PREVIOUS THESIS (CONTEXT ONLY):\n" + "\n".join([f"{i}. {t}" for i, t in zip(p_id, p_txt)]) + "\n\n"
                investment_info += "Note: The above are your past thoughts. If today's news contradicts them, you MUST pivot your stance immediately.\n\n"

            # Cap total context to ~1800 tokens (approx 7500 chars) to ensure MANDATE at end is visible
            if len(investment_info) > 7500:
                investment_info = investment_info[:7500] + "\n...[Memories Truncated for Context Focus]...\n"

            if momentum:
                investment_info += stock_momentum_explanation
                investment_info = _add_momentum_info(momentum, investment_info)

            # Calculate trend_signal description
            trend_signal = "N/A (No historical price data available)"
            if history_prices:
                prev_price = history_prices[-1]
                diff = cur_price - prev_price
                pct = (diff / prev_price) * 100
                direction = "INCREASE" if diff > 0 else "DECREASE"
                trend_signal = f"Today's price of ${cur_price:.2f} is an {direction} of {abs(pct):.2f}% compared to yesterday's ${prev_price:.2f}."

            # Define Allowed Decisions based on Mandates
            is_weekend = cur_date.weekday() >= 5
            if is_weekend:
                allowed_decisions = "BUY/SELL/HOLD"
                mandate_warning = "Note: Weekend trading (Saturday/Sunday). HOLD is permitted."
            else:
                allowed_decisions = "BUY/SELL"
                mandate_warning = "STRICT MANDATE: Weekday trading. You MUST choose BUY or SELL. HOLD is prohibited."

            investment_info += f"\n### CURRENT TRADING MANDATE:\n{mandate_warning}\n\n"

            return (
                investment_info + stock_warmup_final_prompt.format(
                    trend_signal=trend_signal
                )
                if run_mode == RunMode.WARMUP
                else investment_info + stock_test_final_prompt.format(
                    cur_date=cur_date,
                    symbol=symbol,
                    trend_signal=trend_signal,
                    allowed_decisions=allowed_decisions
                )
            )
        else:
            # crypto
            investment_info = prefix_persona + (
                crypto_warmup_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date, 
                    day_of_week=day_of_week, tomorrow_day=tomorrow_day,
                    ground_truth_mandate=future_record,
                    cur_price=cur_price, historical_prices=historical_prices_text
                )
                if run_mode == RunMode.WARMUP
                else crypto_test_investment_info_prefix.format(
                    symbol=symbol, cur_date=cur_date, 
                    day_of_week=day_of_week, tomorrow_day=tomorrow_day,
                    cur_price=cur_price, historical_prices=historical_prices_text
                )
            )
            # Prune and add memories symmetrically - PRIORITY: Short-term > Mid-term > Long-term > Reflection
            if short_memory:
                p_id, p_txt = prune(short_memory_id, short_memory, limit=3)
                investment_info += "Short-term (TODAY'S CRITICAL NEWS):\n" + "\n".join([f"{i}. {t}" for i, t in zip(p_id, p_txt)]) + "\n\n"
                investment_info += crypto_sentiment_explanation

            if mid_memory:
                p_id, p_txt = prune(mid_memory_id, mid_memory, limit=2)
                investment_info += "Mid-term (Recent):\n" + "\n".join([f"{i}. {t}" for i, t in zip(p_id, p_txt)]) + "\n\n"
            
            if long_memory:
                p_id, p_txt = prune(long_memory_id, long_memory, limit=1)
                investment_info += "Long-term (Context):\n" + "\n".join([f"{i}. {t}" for i, t in zip(p_id, p_txt)]) + "\n\n"

            if reflection_memory:
                p_id, p_txt = prune(reflection_memory_id, reflection_memory, limit=2)
                investment_info += "PREVIOUS THESIS (CONTEXT ONLY):\n" + "\n".join([f"{i}. {t}" for i, t in zip(p_id, p_txt)]) + "\n\n"
                investment_info += "Note: The above are your past internal reflections. If today's institutional flows or ETF data contradict them, you MUST pivot.\n\n"

            # Cap total context to ~1800 tokens (approx 7500 chars) to ensure MANDATE at end is visible
            if len(investment_info) > 7500:
                investment_info = investment_info[:7500] + "\n...[Memories Truncated for Context Focus]...\n"

            if momentum:
                investment_info += crypto_momentum_explanation
                investment_info = _add_momentum_info(momentum, investment_info)

            # Calculate trend_signal description
            trend_signal = "N/A (No historical price data available)"
            if history_prices:
                prev_price = history_prices[-1]
                diff = cur_price - prev_price
                pct = (diff / prev_price) * 100
                direction = "INCREASE" if diff > 0 else "DECREASE"
                trend_signal = f"Today's price of ${cur_price:.2f} is an {direction} of {abs(pct):.2f}% compared to yesterday's ${prev_price:.2f}."

            # Define Allowed Decisions for Crypto (Always No-HOLD)
            allowed_decisions = "BUY/SELL"
            mandate_warning = "STRICT MANDATE: BTC always requires a high-conviction BUY or SELL decision. HOLD is prohibited."
            
            investment_info += f"\n### CURRENT TRADING MANDATE:\n{mandate_warning}\n\n"

            return (
                investment_info + crypto_warmup_final_prompt.format(
                    trend_signal=trend_signal
                )
                if run_mode == RunMode.WARMUP
                else investment_info + crypto_test_final_prompt.format(
                    trend_signal=trend_signal,
                    allowed_decisions=allowed_decisions
                )
            )

class MultiAssetsVLLMPromptConstructor(MultiAssetBasePromptConstructor):
    @staticmethod
    def __call__(
        cur_date: date,
        symbols: List[str],
        run_mode: RunMode,
        future_record: Dict[str, Union[float, None]],
        short_memory: Dict[str, Union[List[str], None]],
        short_memory_id: Dict[str, Union[List[int], None]],
        mid_memory: Dict[str, Union[List[str], None]],
        mid_memory_id: Dict[str, Union[List[int], None]],
        long_memory: Dict[str, Union[List[str], None]],
        long_memory_id: Dict[str, Union[List[int], None]],
        reflection_memory: Dict[str, Union[List[str], None]],
        reflection_memory_id: Dict[str, Union[List[int], None]],
        momentum: Dict[str, Union[int, None]],
        character_string: Dict[str, str] = None,
        cur_price: Dict[str, float] = None,
        history_prices: Dict[str, List[float]] = None,
    ) -> Dict[str, str]:

        prompts = {}
        character_string = character_string or {}

        for symbol in symbols:

            prompt = SingleAssetVLLMPromptConstructor.__call__(
                cur_date=cur_date,
                symbol=symbol,
                run_mode=run_mode,
                future_record=future_record.get(symbol),
                short_memory=short_memory.get(symbol),
                short_memory_id=short_memory_id.get(symbol),
                mid_memory=mid_memory.get(symbol),
                mid_memory_id=mid_memory_id.get(symbol),
                long_memory=long_memory.get(symbol),
                long_memory_id=long_memory_id.get(symbol),
                reflection_memory=reflection_memory.get(symbol),
                reflection_memory_id=reflection_memory_id.get(symbol),
                momentum=momentum.get(symbol),
                character_string=character_string.get(symbol, ""),
                cur_price=cur_price.get(symbol, 0.0) if cur_price else 0.0,
                history_prices=history_prices.get(symbol, []) if history_prices else [],
            )

            prompts[symbol] = prompt

        return prompts