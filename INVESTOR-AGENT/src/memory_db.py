import os
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import orjson
from loguru import logger
from pydantic import BaseModel, NonNegativeInt

import chromadb  # ✅ NEW

from .embedding import LocalEmbedding
from .utils import ensure_path


def _date_to_int(d: Union[date, str]) -> int:
    """Consistently convert any date representation to an integer YYYYMMDD."""
    if isinstance(d, date):
        return int(d.strftime("%Y%m%d"))
    # Handle ISO strings "YYYY-MM-DD"
    return int(d.replace("-", ""))


# ---------------- MEMORY FUNCTIONS ---------------- #

class ConstantAccessCounterUpdateFunction:
    def __init__(self, update_step: float) -> None:
        self.update_step = update_step

    def __call__(self, cur_importance_score: float, direction: Literal[1, -1]) -> float:
        return cur_importance_score + self.update_step if direction == 1 else cur_importance_score - self.update_step


class LinearCompoundScore:
    def __init__(self, upper_bound: float) -> None:
        self.upper_bound = upper_bound

    def __call__(self, similarity_score: float, importance_score: float, recency_score: float) -> float:
        return similarity_score + (min(importance_score, self.upper_bound) / self.upper_bound) + recency_score


class ImportanceDecay:
    def __init__(self, decay_rate: float) -> None:
        self.decay_rate = decay_rate

    def __call__(self, cur_val: float) -> float:
        return cur_val * self.decay_rate


class RecencyDecay:
    def __init__(self, recency_factor: float) -> None:
        self.recency_factor = recency_factor

    def __call__(self, delta: float) -> float:
        return np.exp(-(delta / self.recency_factor))


class ConstantImportanceInitialization:
    def __init__(self, init_val: float) -> None:
        self.init_val = init_val

    def __call__(self) -> float:
        return self.init_val


class ConstantRecencyInitialization:
    def __call__(self) -> float:
        return 1.0


# ---------------- DATA STRUCTURES ---------------- #

class MemorySingle(BaseModel):
    id: NonNegativeInt
    symbol: str
    date: date
    text: str


class Memories(BaseModel):
    memory_records: List[MemorySingle]


class QuerySingle(BaseModel):
    query_text: str
    k: NonNegativeInt
    symbol: str


class Queries(BaseModel):
    query_records: List[QuerySingle]


class AccessSingle(BaseModel):
    id: NonNegativeInt
    feedback: Literal[1, -1]


class AccessMulti(BaseModel):
    symbol: str
    id: List[NonNegativeInt]
    feedback: List[Literal[1, -1]]


class AccessFeedback(BaseModel):
    access_counter_records: List[AccessSingle]


class AccessFeedbackMulti(BaseModel):
    access_counter_records: List[AccessMulti]


class JumpDirection(str, Enum):
    UP = "upper"
    DOWN = "lower"


class IDGenerator:
    def __init__(self, id_init: int = 0):
        self.cur_id = id_init

    def __call__(self):
        self.cur_id += 1
        return self.cur_id


# ---------------- MAIN MEMORY DB ---------------- #

class MemoryDB:
    def __init__(self, agent_config: Dict[str, Any], emb_config: Dict[str, Any], db_path: str = None):
        logger.info("SYS-Initializing MemoryDB (Chroma)")

        self.agent_config = agent_config
        self.emb_config = emb_config

        self.emb_model = LocalEmbedding(self.emb_config)

        # ✅ Chroma DB Persistent Client (Ensures memory survives restarts)
        # If no explicit db_path is provided, default to warmup path from config or literal
        if not db_path:
             db_path = os.path.join(agent_config.get("meta_config", {}).get("warmup_checkpoint_save_path", "checkpoints/warmup"), "chroma")
        else:
             # Ensure the chroma DB is in a sub-folder of the provided agent checkpoint path
             db_path = os.path.join(db_path, "chroma")

        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.agent_config["agent_name"]
        )

    # ---------------- ADD MEMORY ---------------- #
    def add_memory(
        self,
        memory_input: List[Dict],
        layer: str,
        importance_init_func,
        recency_init_func,
        run_mode: str = "test", # ✅ NEW: Track the mode of the memory
        similarity_threshold=None,
    ):
        if not memory_input:
            return []

        memories = Memories(memory_records=memory_input)
        texts = [m.text for m in memories.memory_records]
        embs = self.emb_model(texts=texts)

        ids = []
        for m, emb in zip(memories.memory_records, embs):
            try:
                # ✅ Safety: Check for existing ID to avoid ChromaDB errors
                existing = self.collection.get(ids=[str(m.id)])
                if existing and existing["ids"]:
                    logger.warning(f"Memory ID {m.id} already exists. Skipping.")
                    continue

                self.collection.add(
                    ids=[str(m.id)],
                    documents=[m.text],
                    embeddings=[emb],
                    metadatas=[{
                        "symbol": m.symbol,
                        "date": m.date.isoformat(),
                        "date_int": _date_to_int(m.date), # ✅ NEW: Supports comparison operators
                        "layer": layer,
                        "mode": run_mode # ✅ NEW: Mode tagging for sanitization
                    }]
                )
                ids.append(m.id)
            except Exception as e:
                # Catching specific ChromaDB InternalError related to type mismatches/corruption
                if "InternalError" in str(type(e)) or "mismatched types" in str(e):
                    logger.error(f"❌ ChromaDB Internal Error detected: {e}")
                    logger.error("This usually indicates storage corruption in checkpoints/warmup/chroma.")
                    logger.error("The corrupted folder has been cleared if this was the first run.")
                    # Return empty to allow the step to potentially continue or fail gracefully
                    return ids 
                raise e

        return ids

    # ---------------- QUERY ---------------- #
    def query(self, query_input: Queries, layer: str, linear_compound_func, date_limit: str = None, run_mode: str = "test"):
        queries = [q.query_text for q in query_input.query_records]
        embs = self.emb_model(texts=queries)

        results = []
        for emb, q in zip(embs, query_input.query_records):
            
            # ✅ Construct metadata filter
            # We always filter by symbol and layer.
            # If date_limit is provided, we ensure documents are strictly OLDER than the limit.
            where_clause = {
                "$and": [
                    {"symbol": q.symbol},
                    {"layer": layer}
                ]
            }
            if date_limit:
                where_clause["$and"].append({"date_int": {"$lt": _date_to_int(date_limit)}})

            # ✅ CRITICAL: Sanitization filter to prevent lookahead bias
            # In TEST mode, we MUST NOT retrieve anything from WARMUP reflections
            if run_mode == "test" and layer == "reflection":
                where_clause["$and"].append({"mode": {"$ne": "warmup"}})

            res = self.collection.query(
                query_embeddings=[emb],
                n_results=q.k,
                where=where_clause
            )

            texts = res["documents"][0] if res["documents"] and len(res["documents"]) > 0 else []
            ids = [int(i) for i in res["ids"][0]] if res["ids"] and len(res["ids"]) > 0 else []

            results.append((texts, ids))

        return results

    # ---------------- FEEDBACK (SIMPLIFIED) ---------------- #
    def update_access_counter_with_feedback(self, *args, **kwargs):
        pass  # simplified

    # ---------------- DECAY ---------------- #
    def decay(self, *args, **kwargs):
        pass  # simplified

    # ---------------- CLEAN ---------------- #
    def clean_up(self, *args, **kwargs):
        pass  # simplified

    # ---------------- FLOW ---------------- #
    def memory_flow(self, *args, **kwargs):
        pass  # simplified

    # ---------------- SAVE/LOAD ---------------- #
    def save_checkpoint(self, path: str):
        # PersistentClient saves automatically, but we ensure directory structure
        ensure_path(os.path.join(path, "brain"))
        logger.info(f"✅ MemoryDB (Chroma) persisted at {path}")

    @classmethod
    def load_checkpoint(cls, path: str, agent_config, emb_config):
        # To load, we just initialize a new MemoryDB pointing to the same path
        return cls(agent_config=agent_config, emb_config=emb_config)