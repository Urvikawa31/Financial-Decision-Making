import json
import re
from typing import Any, Dict, List, Union

import httpx
from loguru import logger
from pydantic import ValidationError

from ...portfolio import TradeAction
from .base import (
    MultiAssetsStructuredGenerationChatEndPoint,
    MultiAssetsStructureGenerationFailure,
    MultiAssetsStructureOutputResponse,
)


class MultiAssetsVLLMStructureGeneration(MultiAssetsStructuredGenerationChatEndPoint):

    def __init__(self, chat_config: Dict[str, Any]) -> None:

        self.chat_config = chat_config

        self.model = chat_config["chat_model"]

        self.endpoint = "http://localhost:11434/api/generate"

        self.timeout = chat_config["chat_request_timeout"]

        logger.info(f"CHAT-Ollama model: {self.model}")

    def __call__(

        self, prompt: str, schema: Any, symbols: List[str]

    ) -> Union[
        MultiAssetsStructureGenerationFailure,
        MultiAssetsStructureOutputResponse,
    ]:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
             "options": {
            "temperature":0,
             }
        }

        try:

            with httpx.Client(timeout=self.timeout) as client:

                response = client.post(self.endpoint, json=payload)

            response.raise_for_status()

            result_text = response.json()["response"]
            
            # 🔥 Remove <think> blocks from DeepSeek-R1 responses
            result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()

            response_dict = json.loads(result_text)

        except Exception as e:

            logger.error(f"Ollama generation failed: {e}")

            return MultiAssetsStructureGenerationFailure(
                investment_decision={symbol: TradeAction.HOLD for symbol in symbols}
            )

        try:

            summary_reason = {
                symbol: response_dict["symbols_summary"][f"{symbol}_summary_reason"]
                for symbol in symbols
            }

            investment_decision = {
                symbol: response_dict["symbols_summary"][
                    f"{symbol}_investment_decision"
                ]
                for symbol in symbols
            }

            return MultiAssetsStructureOutputResponse(
                investment_decision=investment_decision,
                summary_reason=summary_reason,
                short_memory_ids={},
                mid_memory_ids={},
                long_memory_ids={},
                reflection_memory_ids={},
            )

        except (ValidationError, KeyError):

            logger.error("CHAT parsing failed")

            return MultiAssetsStructureGenerationFailure(
                investment_decision={symbol: TradeAction.HOLD for symbol in symbols}
            )