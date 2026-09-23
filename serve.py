#!/usr/bin/env python
"""Launch the mini-vLLM OpenAI-compatible server.

    python serve.py                       # 127.0.0.1:8000, auto device
    python serve.py --device cuda --port 8000
    python serve.py --weights checkpoints/model.pt
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mini_vllm._env import ensure_local_libs  # noqa: E402

ensure_local_libs()

from mini_vllm.api_server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
