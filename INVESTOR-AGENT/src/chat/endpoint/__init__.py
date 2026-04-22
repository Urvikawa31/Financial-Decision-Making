from .base import (
    MultiAssetsStructuredGenerationChatEndPoint,
    MultiAssetsStructureGenerationFailure,
    MultiAssetsStructureOutputResponse,
)

from .vllm import (
    MultiAssetsVLLMStructureGeneration,
)

__all__ = [
    "MultiAssetsStructuredGenerationChatEndPoint",
    "MultiAssetsStructureGenerationFailure",
    "MultiAssetsStructureOutputResponse",
    "MultiAssetsVLLMStructureGeneration",
]
