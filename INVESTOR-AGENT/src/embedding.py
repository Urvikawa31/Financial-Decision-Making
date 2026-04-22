from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingModel(ABC):

    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        pass


class LocalEmbedding(EmbeddingModel):

    def __init__(self, emb_config: Dict):

        self.config = emb_config

        model_name = self.config["emb_model_name"]

        # map short name to real model
        if model_name == "bge-small":
            model_name = "BAAI/bge-small-en-v1.5"

        logger.info(f"EMB-Loading local embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: Union[List[str], str]) -> List[List[float]]:

        if isinstance(texts, str):
            texts = [texts]

        logger.trace("EMB-Generating embeddings locally")

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        return embeddings.tolist()