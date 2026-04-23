import os
import shutil
from enum import Enum

from loguru import logger


class RunMode(str, Enum):
    WARMUP = "warmup"
    TEST = "test"


class TaskType(str, Enum):
    SingleAsset = "single_asset"
    MultiAssets = "multi_assets"


def ensure_path(save_path: str) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        logger.debug(f"Path already exists: {save_path}")
    logger.info(f"Path created: {save_path}")