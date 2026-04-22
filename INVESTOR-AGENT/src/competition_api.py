import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from loguru import logger

# Import agent components
from .agent import FinMemAgent
from .market_env import OneDayMarketInfo
from .utils import RunMode, TaskType

load_dotenv()

# Configure logging to file
os.makedirs("logs", exist_ok=True)
logger.add("logs/competition_api.log", rotation="10 MB", level="INFO")

app = FastAPI(title="Investor Agent Competition API", version="2.0.0")

# Add CORS middleware for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global variables to hold agent and config
agent = None
config = None

def load_config(path: str) -> Dict:
    import orjson
    with open(path, "rb") as f:
        return orjson.loads(f.read())

def init_agent():
    global agent, config
    config_path = os.getenv("CONFIG_PATH", os.path.join("configs", "main.json"))
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        # Create a minimal config if missing, but ideally it should exist
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    config = load_config(config_path)
    
    # Allow environment overrides for competition deployment
    config["chat_config"]["chat_model"] = os.getenv("CHAT_MODEL", config["chat_config"].get("chat_model", "gpt-oss:120b"))
    config["chat_config"]["chat_endpoint"] = os.getenv("CHAT_ENDPOINT", "https://ollama.com/api/generate")
    config["chat_config"]["chat_model_inference_engine"] = os.getenv("CHAT_ENGINE", "ollama")
    
    # Increase default timeout for cloud models
    config["chat_config"]["chat_request_timeout"] = int(os.getenv("CHAT_TIMEOUT", "180"))

    # Path to the warmed-up agent checkpoint
    checkpoint_path = os.getenv("CHECKPOINT_PATH", os.path.join("outputs", "warmup", "agent"))
    
    try:
        logger.info(f"SYS-Loading agent from checkpoint: {checkpoint_path}")
        agent = FinMemAgent.load_checkpoint(
            path=checkpoint_path,
            config=config,
            portfolio_load_for_test=True
        )
        logger.info("SYS-Agent loaded successfully from checkpoint.")
    except Exception as e:
        logger.warning(f"SYS-Failed to load checkpoint: {e}. Initializing fresh agent.")
        agent = FinMemAgent(
            agent_config=config["agent_config"],
            emb_config=config["emb_config"],
            chat_config=config["chat_config"],
            portfolio_config=config["portfolio_config"],
            task_type=TaskType.MultiAssets if len(config["env_config"]["trading_symbols"]) > 1 else TaskType.SingleAsset
        )

# --- Competition Models ---

class HistoricalPrice(BaseModel):
    date: str
    price: float

class TradingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: str
    price: Dict[str, float]
    news: Dict[str, List[str]]
    symbol: List[str]
    momentum: Optional[Dict[str, str]] = None
    history_price: Dict[str, List[HistoricalPrice]] = Field(default_factory=dict, alias="history_price")
    ten_k: Optional[Dict[str, List[str]]] = Field(default=None, alias="10k")
    ten_q: Optional[Dict[str, List[str]]] = Field(default=None, alias="10q")

class TradingResponse(BaseModel):
    recommended_action: str

# --- Lifecycle ---

@app.on_event("startup")
async def startup_event():
    init_agent()
    # Pre-flight check for Ollama Cloud
    try:
        import requests
        chat_endpoint = config["chat_config"]["chat_endpoint"]
        # Try to reach the root or the endpoint itself with a head request
        logger.info(f"SYS-Checking connectivity to {chat_endpoint}...")
        requests.options(chat_endpoint, timeout=5) 
        logger.info(f"SYS-Chat endpoint is REACHABLE.")
    except Exception as e:
        logger.warning(f"SYS-Chat endpoint might be slow or unreachable: {e}. Proceeding anyway.")

@app.get("/")
async def home():
    return {"message": "Investor Agent Competition API (Task 3)"}

@app.get("/health")
async def health():
    chat_ok = False
    try:
        import requests
        chat_endpoint = config["chat_config"]["chat_endpoint"]
        # Determine base URL for health check
        base_url = "/".join(chat_endpoint.split("/")[:-2]) if "/" in chat_endpoint else chat_endpoint
        requests.get(base_url, timeout=2)
        chat_ok = True
    except:
        pass
    return {
        "status": "healthy" if agent else "starting",
        "agent_ready": agent is not None,
        "chat_reachable": chat_ok,
        "model": config["chat_config"]["chat_model"],
        "timestamp": datetime.now().isoformat()
    }

# --- Core Endpoint ---

@app.post("/trading_action/", response_model=TradingResponse)
async def get_trading_decision(request: TradingRequest):
    global agent
    if agent is None:
        init_agent()
        
    original_agent_symbols = None
    original_portfolio_symbols = None
    try:
        if not request.symbol:
            raise HTTPException(status_code=400, detail="No symbol provided")

        target_symbol = request.symbol[0]
        
        # 1. Prepare Market Info (Mapping competition format to agent format)
        try:
            cur_date = datetime.strptime(request.date, "%Y-%m-%d").date()
        except ValueError:
            # Try fallback if date format is different
            cur_date = datetime.now().date()

        # Convert history_price
        history_prices = {}
        for s, hps in request.history_price.items():
            history_prices[s] = [hp.price for hp in hps]
            
        # Convert momentum (bullish/bearish/neutral -> 1/-1/0)
        momentum_map = {"bullish": 1, "bearish": -1, "neutral": 0}
        agent_momentum = {}
        if request.momentum:
            for s, m in request.momentum.items():
                agent_momentum[s] = momentum_map.get(m.lower(), 0)
        
        # Fill missing momentum with 0
        for s in request.symbol:
            if s not in agent_momentum:
                agent_momentum[s] = 0

        # Handle 10k/10q
        filing_k = {}
        if request.ten_k:
            for s, texts in request.ten_k.items():
                filing_k[s] = texts[0] if texts else None
        
        filing_q = {}
        if request.ten_q:
            for s, texts in request.ten_q.items():
                filing_q[s] = texts[0] if texts else None

        market_info = OneDayMarketInfo(
            cur_date=cur_date,
            cur_price=request.price,
            cur_history_prices=history_prices,
            cur_news=request.news,
            cur_filing_k=filing_k,
            cur_filing_q=filing_q,
            cur_momentum=agent_momentum,
            cur_symbol=request.symbol,
            cur_future_price_diff={s: 0.0 for s in request.symbol},
            termination_flag=False
        )

        # 2. Dynamic Configuration (Sync symbols with the request)
        # We temporarily set the agent's symbols to only those in the request to avoid KeyErrors
        # while ensuring the agent is prepared for these symbols.
        original_agent_symbols = agent.agent_config["trading_symbols"]
        original_portfolio_symbols = agent.portfolio.trading_symbols
        
        agent.agent_config["trading_symbols"] = request.symbol
        agent.portfolio.trading_symbols = request.symbol
        
        for s in request.symbol:
            if s not in agent.agent_config["character_string"]:
                agent.agent_config["character_string"][s] = f"You are a professional analyst covering {s}."
            if s not in agent.portfolio.current_weights:
                agent.portfolio.current_weights[s] = 0
                agent.portfolio.asset_value[s] = []
                agent.portfolio.trading_price[s] = []
                agent.portfolio.asset_cash[s] = agent.portfolio.buying_power / len(request.symbol) # Simple allocation

        # Refresh queries for the new set of symbols
        agent._construct_queries()

        # 3. Execute Agent Step
        logger.info(f"🚀 Processing decision for {target_symbol} | Date: {request.date}")
        agent.step(market_info=market_info, run_mode=RunMode.TEST, task_type=agent.task_type)
        
        # 4. Extract Decision
        # In PortfolioMultiAsset.record_action, current_weights[s] is updated to 1 (BUY), -1 (SELL), or 0 (HOLD)
        weight = agent.portfolio.current_weights.get(target_symbol, 0)
        action_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        recommended_action = action_map.get(weight, "HOLD")
        
        # 5. Restore Symbols (Clean up)
        agent.agent_config["trading_symbols"] = original_agent_symbols
        agent.portfolio.trading_symbols = original_portfolio_symbols
        agent._construct_queries()
        
        logger.info(f"SYS-Decision for {target_symbol}: {recommended_action}")
        return TradingResponse(recommended_action=recommended_action)

    except Exception as exc:
        # Restore on error too
        if original_agent_symbols is not None:
            agent.agent_config["trading_symbols"] = original_agent_symbols
            agent.portfolio.trading_symbols = original_portfolio_symbols
            agent._construct_queries()
            
        logger.error(f"SYS-Error in trading_action: {exc}")
        # Default to HOLD on error as per competition policy
        return TradingResponse(recommended_action="HOLD")

if __name__ == "__main__":
    import uvicorn
    # Default port for competition often varies, using 62237 as per sample
    port = int(os.getenv("PORT", 62237))
    logger.info(f"Starting Competition API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
