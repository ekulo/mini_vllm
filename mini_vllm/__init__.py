"""mini-vLLM: a small, readable LLM inference engine.

The model is the hand-written Transformer in ``手写/`` (128 hidden, 4 layers,
8 heads, 1000-token vocab). This package wraps it in the same moving parts a
real serving engine has:

    api_server.py   OpenAI-ish HTTP endpoints (FastAPI + SSE)
    engine_core.py  the step() loop that ties scheduler + runner + kv together
    scheduler.py    continuous batching, FCFS admission, recompute preemption
    kv_manager.py   paged KV-cache block allocator
    model_runner.py loads the model, builds attention metadata, samples tokens
"""

from ._env import ensure_local_libs

# Must run before torch / fastapi are imported anywhere in this package.
ensure_local_libs()

from .config import EngineConfig
from .sequence import SamplingParams, Sequence, SequenceStatus
from .kv_manager import KVCacheManager
from .scheduler import Scheduler
from .model_runner import ModelRunner
from .engine_core import EngineCore, RequestOutput
from .llm_engine import LLMEngine

__all__ = [
    "EngineConfig",
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
    "KVCacheManager",
    "Scheduler",
    "ModelRunner",
    "EngineCore",
    "RequestOutput",
    "LLMEngine",
]

__version__ = "0.1.0"
