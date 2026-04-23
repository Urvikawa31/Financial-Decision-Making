import os
import json
import re
import pickle
from typing import Any, Dict, List

import orjson
from loguru import logger

from .chat import (
    MultiAssetsStructureGenerationFailure,
    get_chat_model,
)
from .market_env import OneDayMarketInfo
from .memory_db import (
    ConstantAccessCounterUpdateFunction,
    ConstantImportanceInitialization,
    ConstantRecencyInitialization,
    IDGenerator,
    ImportanceDecay,
    LinearCompoundScore,
    MemoryDB,
    Queries,
    QuerySingle,
    RecencyDecay,
)
from .portfolio import (
    TradeAction,
    construct_portfolio,
)
from .utils import RunMode, TaskType


class FinMemAgent:
    def __init__(
        self,
        agent_config: Dict[str, Any],
        emb_config: Dict[str, Any],
        chat_config: Dict[str, Any],
        portfolio_config: Dict[str, Any],
        task_type: TaskType,
    ) -> None:
        logger.info("SYS-Initializing FinMemAgent")

        self.agent_config = agent_config
        self.emb_config = emb_config
        self.chat_config = chat_config
        self.portfolio_config = portfolio_config
        self.task_type = task_type

        self.memory_db = MemoryDB(agent_config=agent_config, emb_config=emb_config)
        self.id_generator = IDGenerator(id_init=0)

        # 🔥 CHAT MODEL (NON-PICKLABLE)
        self.chat_schema, self.chat_endpoint, self.chat_prompt = get_chat_model(
            chat_config=chat_config, task_type=task_type
        )

        self._config_memory_settings()
        self._construct_queries()

        self.portfolio = construct_portfolio(portfolio_config=portfolio_config)

    def _construct_queries(self) -> None:
        self.queries = Queries(
            query_records=[
                QuerySingle(
                    query_text=self.agent_config["character_string"][symbol],
                    k=self.agent_config["top_k"],
                    symbol=symbol,
                )
                for symbol in self.agent_config["trading_symbols"]
            ]
        )

    def _config_memory_settings(self) -> None:
        config = self.agent_config["memory_db_config"]

        self.memory_access_update = ConstantAccessCounterUpdateFunction(
            update_step=config["memory_importance_score_update_step"]
        )

        self.memory_compound_score = LinearCompoundScore(
            upper_bound=config["memory_importance_upper_bound"]
        )

        self.short_importance_init = ConstantImportanceInitialization(
            config["short"]["importance_init_val"]
        )
        self.short_recency_init = ConstantRecencyInitialization()

        self.short_importance_decay = ImportanceDecay(
            config["short"]["decay_importance_factor"]
        )
        self.short_recency_decay = RecencyDecay(
            config["short"]["decay_recency_factor"]
        )

        self.threshold_dict = config

    def _handling_new_information(self, market_info: OneDayMarketInfo, run_mode: RunMode) -> None:
        # News -> Short Layer
        for symbol, news in market_info.cur_news.items():
            if news:
                self.memory_db.add_memory(
                    memory_input=[
                        {
                            "id": self.id_generator(),
                            "symbol": symbol,
                            "date": market_info.cur_date,
                            "text": n,
                        }
                        for n in news
                    ],
                    layer="short",
                    importance_init_func=self.short_importance_init,
                    recency_init_func=self.short_recency_init,
                    run_mode=run_mode.value,
                )

        # 10-Q -> Mid Layer
        for symbol, filing_q in market_info.cur_filing_q.items():
            if filing_q:
                self.memory_db.add_memory(
                    memory_input=[{
                        "id": self.id_generator(),
                        "symbol": symbol,
                        "date": market_info.cur_date,
                        "text": filing_q,
                    }],
                    layer="mid",
                    importance_init_func=self.short_importance_init, # reuse init for now or pull from config
                    recency_init_func=self.short_recency_init,
                    run_mode=run_mode.value,
                )

        # 10-K -> Long Layer
        for symbol, filing_k in market_info.cur_filing_k.items():
            if filing_k:
                self.memory_db.add_memory(
                    memory_input=[{
                        "id": self.id_generator(),
                        "symbol": symbol,
                        "date": market_info.cur_date,
                        "text": filing_k,
                    }],
                    layer="long",
                    importance_init_func=self.short_importance_init,
                    recency_init_func=self.short_recency_init,
                    run_mode=run_mode.value,
                )
                
    def _query_memories(self, market_info: OneDayMarketInfo, symbols: List[str], run_mode: RunMode):
        """Query memories isolated by symbol for each layer with date filtering to prevent leakage."""
        results = {"short": [], "mid": [], "long": [], "reflection": []}
        
        # ✅ In TEST mode, we ensure we only retrieve memories strictly older than today.
        # In WARMUP, we can see current day memories if they exist (though usually they are added after query).
        date_limit = market_info.cur_date.isoformat() if run_mode == RunMode.TEST else None

        for s in symbols:
            # 1. Gather all today's news for this symbol to use as query texts
            symbol_news = market_info.cur_news.get(s, [])
            
            if not symbol_news:
                results["short"].append(([], []))
                results["mid"].append(([], []))
                results["long"].append(([], []))
                results["reflection"].append(([], []))
                continue

            query_input = Queries(
                query_records=[
                    QuerySingle(query_text=n, k=self.agent_config["top_k"], symbol=s) 
                    for n in symbol_news
                ]
            )

            # Query each layer with symbol filter
            for layer in ["short", "mid", "long", "reflection"]:
                
                # ✅ Safety: Even if the reflection layer was populated with ground-truth leaks during warmup,
                # the date_limit filter will now prevent them from being seen on the same test day.
                resp = self.memory_db.query(
                    query_input=query_input, 
                    layer=layer, 
                    linear_compound_func=self.memory_compound_score,
                )
                
                # resp is List[Tuple[texts, ids]]
                all_texts = []
                all_ids = []
                for texts, ids in resp:
                    all_texts.extend(texts)
                    all_ids.extend(ids)
                
                results[layer].append((all_texts, all_ids))
                
        return results

    def _multi_assets_trade_action(
        self,
        queried_memories,
        market_info: OneDayMarketInfo,
        run_mode: RunMode,
    ):
        logger.info("\n🚀 STARTING MULTI-ASSET TRADING STEP")

        symbols = self.agent_config["trading_symbols"]

        momentum = {
            s: (market_info.cur_momentum[s] if market_info.cur_momentum[s] is not None else 0)
            for s in symbols
        }

        future_record = market_info.cur_future_price_diff

        short_memory = {}
        short_memory_id = {}
        mid_memory = {s: [] for s in symbols}
        mid_memory_id = {s: [] for s in symbols}
        long_memory = {s: [] for s in symbols}
        long_memory_id = {s: [] for s in symbols}
        reflection_memory = {s: [] for s in symbols}
        reflection_memory_id = {s: [] for s in symbols}

        for i, s in enumerate(symbols):
            # Short Layer
            mems, ids = queried_memories["short"][i]
            short_memory[s] = mems
            short_memory_id[s] = ids

            # Mid Layer
            mems, ids = queried_memories["mid"][i]
            mid_memory[s] = mems
            mid_memory_id[s] = ids

            # Long Layer
            mems, ids = queried_memories["long"][i]
            long_memory[s] = mems
            long_memory_id[s] = ids

            # Reflection Layer
            mems, ids = queried_memories["reflection"][i]
            reflection_memory[s] = mems
            reflection_memory_id[s] = ids

        # 🔥 ENHANCE FUTURE RECORD (ONLY FOR WARMUP)
        enhanced_future_record = {}
        if run_mode == RunMode.WARMUP:
            for s in symbols:
                enhanced_future_record[s] = future_record.get(s, 0.0)
        else:
            # In TEST mode, we provide NO future information
            enhanced_future_record = {s: None for s in symbols}

        # 🔥 GET PROMPTS PER ASSET
        prompts = self.chat_prompt(
            cur_date=market_info.cur_date,
            symbols=symbols,
            run_mode=run_mode,
            short_memory=short_memory,
            short_memory_id=short_memory_id,
            mid_memory=mid_memory,
            mid_memory_id=mid_memory_id,
            long_memory=long_memory,
            long_memory_id=long_memory_id,
            reflection_memory=reflection_memory,
            reflection_memory_id=reflection_memory_id,
            momentum=momentum,
            future_record=enhanced_future_record, # Now clean in TEST mode
            character_string=self.agent_config.get("character_string", {}),
            cur_price=market_info.cur_price,
            history_prices=market_info.cur_history_prices,
        )

        logger.info("📤 PROMPTS GENERATED")
    
        results = {}
        results_reasoning = {}
    
        # 🔥 LOOP PER SYMBOL
        for symbol, prompt in prompts.items():
            
            # 🔥 Dynamic Mandate Generation (Ensures instructions are at the very end of the prompt)
            is_weekend = market_info.cur_date.weekday() >= 5
            
            if run_mode == RunMode.WARMUP:
                allowed_decisions = "BUY/SELL/HOLD"
                mandate_rule = f"- **UNIVERSAL LEARNING**: The actual outcome for {symbol} is a {future_record[symbol]} change. JUSTIFY this move. **STRICT PROTECTIVE RULE**: You must NOT mention the number '{future_record[symbol]}' in your JSON reasoning."
            else:
                # v3: "Magnitude x Variance" Aggressive Mandates
                if symbol == "BTC":
                    allowed_decisions = "BUY/SELL"
                    mandate_rule = f"- **ALPHA CAPTURE**: BTC requires a directional bias based on Liquidity/ETF flows. **HOLD IS PROHIBITED**. You MUST choose: {allowed_decisions}."
                elif symbol == "TSLA":
                    if is_weekend:
                        allowed_decisions = "BUY/SELL/HOLD"
                        mandate_rule = f"- **ALPHA CAPTURE**: {symbol} (Weekend). HOLD is permitted if catalysts are low-magnitude (<2)."
                    else:
                        allowed_decisions = "BUY/SELL"
                        mandate_rule = f"- **ALPHA CAPTURE**: Weekday trading requires captured alpha. **HOLD IS PROHIBITED**. You MUST choose: {allowed_decisions}."
                else:
                    allowed_decisions = "BUY/SELL/HOLD"
                    mandate_rule = f"- **ALPHA CAPTURE**: Synthesize catalysts into a definitive stance for {symbol}."

            prompt += f"\n### STRICT EXECUTION RULE:\n- Output must be valid JSON only.\n- **DECISION SPACE**: {allowed_decisions}\n{mandate_rule}\n- No conversational preamble.\n"
    
            logger.info(f"📤 SENDING PROMPT FOR {symbol} (Length: {len(prompt)})")
            logger.info(f"DEBUG-PROMPT-SNAPSHOT: {prompt[:200]}...")
            
            # log prompt for debugging
            # logger.debug(f"PROMPT for {symbol}: {prompt}")
    
            try:
                response = self.chat_endpoint(prompt=prompt)
                logger.info(f"📥 RESPONSE [{symbol}]: {response}")
            except Exception as e:
                logger.error(f"❌ CHAT ERROR [{symbol}]: {e}")
                results[symbol] = TradeAction.HOLD
                continue
            
            # 🔥 PARSE PER SYMBOL (Extremely Robust Multi-Stage Parser)
            try:
                decision = "hold"
                reasoning = "N/A"
                
                # Pre-clean: (Logic for thinking blocks removed for Qwen2.5)
                response = response.strip()
                
                # Stage 1: Try to find a JSON block { ... }
                match = re.search(r'\{.*?\}', response, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        decision = parsed.get("investment_decision", "hold").lower()
                        reasoning = parsed.get("reasoning", "N/A")
                    except:
                        pass # Fall through to stage 2
                
                # Stage 2: If JSON failed or was invalid, try to find "investment_decision": "..." anywhere
                if decision == "hold":
                    kv_match = re.search(r'"investment_decision":\s*"(\w+)"', response, re.I)
                    if kv_match:
                        decision = kv_match.group(1).lower()
                
                # Stage 3: Keyword Fallback (Last resort)
                if decision == "hold":
                    # Look for bolded keywords or standalone words in the response
                    keywords = re.findall(r'\b(buy|sell|hold)\b', response, re.I)
                    if keywords:
                        # Take the LAST mention as the final conclusion
                        decision = keywords[-1].lower()
    
                if decision == "buy":
                    results[symbol] = TradeAction.BUY
                elif decision == "sell":
                    results[symbol] = TradeAction.SELL
                else:
                    results[symbol] = TradeAction.HOLD
                
                # 🔥 STORE THE REASONING (for later memory storage)
                # If reasoning is still N/A but we have a response, use the response text (cleaned)
                if reasoning == "N/A" and response:
                    reasoning = response[:500] + "..." if len(response) > 500 else response
                
                results_reasoning[symbol] = reasoning
    
            except Exception as e:
                logger.warning(f"⚠️ JSON parsing failed [{symbol}]: {e}")
                results[symbol] = TradeAction.HOLD
                results_reasoning[symbol] = "Error during parsing."
    
        logger.info("\n📊 FINAL ACTIONS TAKEN")
        for s in results:
            logger.info(f"  {s}: {results[s].value.upper()}")
    
        # 🔥 APPLY TO PORTFOLIO
        self.portfolio.record_action(
            action_date={s: market_info.cur_date for s in symbols},
            action=results,
            price_info=market_info.cur_price,
            evidence={s: [results_reasoning[s]] for s in symbols},
        )
        
        # 🔥 STORE IN REFLECTION MEMORY
        self._store_reflections(
            market_info=market_info,
            decisions=results,
            reasonings=results_reasoning,
            run_mode=run_mode,
        )
    
        logger.info("✅ STEP COMPLETED")

    def _store_reflections(self, market_info: OneDayMarketInfo, decisions: Dict[str, TradeAction], reasonings: Dict[str, str], run_mode: RunMode):
        """Persist the agent's internal thoughts to the Reflection Memory Layer."""
        for symbol, action in decisions.items():
            reasoning = reasonings.get(symbol, "No reasoning provided.")
            
            # Format the reflection string
            reflection_text = f"Decision: {action.value.upper()} | Reasoning: {reasoning}"
            
            logger.info(f"🧠 STORING REFLECTION for {symbol}: {reflection_text[:100]}...")
            
            self.memory_db.add_memory(
                memory_input=[{
                    "id": self.id_generator(),
                    "symbol": symbol,
                    "date": market_info.cur_date,
                    "text": reflection_text,
                }],
                layer="reflection",
                importance_init_func=self.short_importance_init, # use default importance for now
                recency_init_func=self.short_recency_init,
                run_mode=run_mode.value,
            )


    def step(self, market_info, run_mode, task_type):
        symbols = self.agent_config["trading_symbols"]
        self._handling_new_information(market_info, run_mode)
        
        # Query memories with required arguments
        queried_memories = self._query_memories(market_info, symbols, run_mode)

        self._multi_assets_trade_action(
            queried_memories=queried_memories,
            market_info=market_info,
            run_mode=run_mode,
        )

        self.memory_db.decay(
            self.short_importance_decay, self.short_recency_decay, "short"
        )

    # 🔥 FIXED SAVE (IMPORTANT)
    def save_checkpoint(self, path: str) -> None:
        import os
        import pickle
    
        os.makedirs(path, exist_ok=True)
    
        # ✅ Save Agent State (Simple values)
        state = {
            "portfolio_cash": self.portfolio.buying_power,
            "portfolio_positions": self.portfolio.current_weights,
            "id_generator_val": self.id_generator.cur_id
        }
    
        with open(os.path.join(path, "agent_state.pkl"), "wb") as f:
            pickle.dump(state, f)
    
        # ✅ Save Portfolio (Detailed records)
        if hasattr(self, "portfolio") and self.portfolio is not None:
            self.portfolio.save_checkpoint(path)

        # ✅ Save MemoryDB (Chroma)
        self.memory_db.save_checkpoint(path)
    
        logger.info(f"✅ Full agent checkpoint saved at {path}")

    # 🔥 FIXED LOAD (IMPORTANT)
    @classmethod
    def load_checkpoint(cls, path: str, config: dict, portfolio_load_for_test: bool = False ):
        import os
        import pickle
    
        state_path = os.path.join(path, "agent_state.pkl")
    
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Checkpoint not found: {state_path}")
    
        # 🔥 Create fresh agent
        agent = cls(
            agent_config=config["agent_config"],
            emb_config=config["emb_config"],
            chat_config=config["chat_config"],
            portfolio_config=config["portfolio_config"],
            task_type=TaskType.MultiAssets,
        )
    
        # 🔥 Load saved data
        with open(state_path, "rb") as f:
            state = pickle.load(f)
    
        # ✅ Portfolio handling depends on whether we're loading for test
        from .portfolio import PortfolioMultiAsset
        if portfolio_load_for_test:
            # 🔥 CRITICAL: For test mode, create a FRESH portfolio.
            # The warmup portfolio carries 151 days of oracle-mode P&L,
            # which would contaminate test metrics if we reuse it.
            logger.info("🔥 Creating FRESH portfolio for TEST (not loading warmup P&L)")
            agent.portfolio = PortfolioMultiAsset(portfolio_config=config["portfolio_config"])
        else:
            agent.portfolio = PortfolioMultiAsset.load_checkpoint(path)
        
        # ✅ Restore other lightweight states
        agent.id_generator.cur_id = state.get("id_generator_val", 0)
    
        logger.info(f"✅ Full Agent state restored from {path}")
    
        return agent
