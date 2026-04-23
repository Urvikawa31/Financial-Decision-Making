from typing import Dict, Tuple, Union

from loguru import logger
import os
import requests

# ONLY keep required imports
from .endpoint import (
    MultiAssetsStructuredGenerationChatEndPoint,
    MultiAssetsVLLMStructureGeneration,
    MultiAssetsStructureGenerationFailure,
    MultiAssetsStructureOutputResponse,
)

from .prompt import (
    MultiAssetBasePromptConstructor,
    MultiAssetsVLLMPromptConstructor,
)

from .structure_generation import (
    MultiAssetsBaseStructureGenerationSchema,
    MultiAssetsVLLMStructureGenerationSchema,
)

from ..utils import TaskType


# Only multi-asset (BTC + TSLA)
multi_asset_return_type = Tuple[
    MultiAssetsBaseStructureGenerationSchema,
    MultiAssetsStructuredGenerationChatEndPoint,
    MultiAssetBasePromptConstructor,
]


# ---------------- CHAT ENDPOINT (OLLAMA / HF / VLLM) ---------------- #
class OllamaChatEndpoint(MultiAssetsStructuredGenerationChatEndPoint):
    def __init__(self, chat_config: Dict):
        self.chat_config = chat_config
        self.endpoint = chat_config.get("chat_endpoint", "")
        self.model = chat_config["chat_model"]
        self.system_message = chat_config.get("chat_system_message", "")

    def __call__(self, prompt: str) -> str:
        try:
            config_timeout = self.chat_config.get("chat_request_timeout", 180)
            timeout = min(config_timeout, 300)
            
            # Support both Ollama and OpenAI-compatible endpoints
            is_openai = self.endpoint and "/v1" in self.endpoint
            
            if is_openai:
                messages = []
                if self.system_message:
                    messages.append({"role": "system", "content": self.system_message})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.chat_config.get("chat_parameters", {}).get("temperature", 0.2)
                }
                
                # Try multiple API key names
                api_key = (
                    os.getenv("HF_TOKEN") or 
                    os.getenv("HUGGINGFACE_API_KEY") or 
                    os.getenv("OPENAI_API_KEY") or 
                    os.getenv("OLLAMA_API_KEY", "")
                )
                headers = {"Authorization": f"Bearer {api_key}"}
            else:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
                # Support OLLAMA_API_KEY for official Ollama Cloud
                headers = {}
                ollama_key = os.getenv("OLLAMA_API_KEY")
                if ollama_key:
                    headers["Authorization"] = f"Bearer {ollama_key}"

            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            res_json = response.json()

            if is_openai:
                return res_json["choices"][0]["message"]["content"]
            else:
                return res_json.get("response", "HOLD")

        except requests.exceptions.Timeout:
            logger.error(f"CLOUD ERROR: Request timed out after {timeout}s")
            return "HOLD"
        except Exception as e:
            logger.error(f"CLOUD ERROR: {e}")
            return "HOLD"


# ---------------- LOCAL TRANSFORMERS ENDPOINT (OFFLINE) ---------------- #
class LocalTransformersChatEndpoint(MultiAssetsStructuredGenerationChatEndPoint):
    def __init__(self, chat_config: Dict):
        self.chat_config = chat_config
        self.model_id = chat_config["chat_model"]
        self.system_message = chat_config.get("chat_system_message", "")
        
        logger.info(f"SYS-Loading model {self.model_id} OFFLINE...")
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Use 4-bit or 8-bit if requested or default to auto
            # Note: Requires bitsandbytes and accelerate
            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "trust_remote_code": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=chat_config.get("chat_max_new_token", 2048),
                temperature=chat_config.get("chat_parameters", {}).get("temperature", 0.2),
                do_sample=True,
            )
            logger.info("✅ Local model loaded successfully")
            
        except ImportError as e:
            logger.error(f"FATAL: Missing dependencies for local inference: {e}")
            raise e
        except Exception as e:
            logger.error(f"FATAL: Failed to load local model: {e}")
            raise e

    def __call__(self, prompt: str) -> str:
        try:
            # Format using chat template if available, otherwise manual
            if self.tokenizer.chat_template:
                messages = []
                if self.system_message:
                    messages.append({"role": "system", "content": self.system_message})
                messages.append({"role": "user", "content": prompt})
                
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = f"{self.system_message}\n\n{prompt}" if self.system_message else prompt
            
            output = self.pipe(formatted_prompt)
            generated_text = output[0]["generated_text"]
            
            # Clean up: strip the prompt from the response
            if generated_text.startswith(formatted_prompt):
                response = generated_text[len(formatted_prompt):].strip()
            else:
                # Fallback if pipeline returns the full sequence
                response = generated_text.strip()
                
            return response
            
        except Exception as e:
            logger.error(f"LOCAL INFERENCE ERROR: {e}")
            return "HOLD"


# ---------------- MAIN FUNCTION ---------------- #
def get_chat_model(
    chat_config: Dict, task_type: TaskType
) -> multi_asset_return_type:

    logger.trace("SYS-Initializing chat model, prompt, and schema")

    engine = chat_config["chat_model_inference_engine"]

    # ✅ OLLAMA / HF / VLLM SUPPORT
    if engine in ["ollama", "vllm", "huggingface"]:
        logger.trace(f"SYS-Chat model is using {engine} engine")

        return (
            MultiAssetsVLLMStructureGenerationSchema(),
            OllamaChatEndpoint(chat_config=chat_config),
            MultiAssetsVLLMPromptConstructor(),
        )

    # ✅ LOCAL TRANSFORMERS SUPPORT (OFFLINE)
    elif engine in ["local", "offline"]:
        logger.trace(f"SYS-Chat model is using {engine} engine (Transformers)")

        return (
            MultiAssetsVLLMStructureGenerationSchema(),
            LocalTransformersChatEndpoint(chat_config=chat_config),
            MultiAssetsVLLMPromptConstructor(),
        )

    else:
        logger.error(
            f"SYS-Model {engine} not supported"
        )
        raise NotImplementedError(
            f"Model {engine} not implemented"
        )