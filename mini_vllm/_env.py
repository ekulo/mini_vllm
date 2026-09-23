"""Make the workspace-local dependency folder importable.

``tools/fetch_cuda_torch.py`` and ``tools/fetch_pypi_wheels.py`` unpack wheels
into ``<repo>/.pylibs`` (pip itself is unusable in this sandbox). Putting that
directory at the front of ``sys.path`` here means the CUDA build of PyTorch
shadows any CPU-only install, and FastAPI/uvicorn are found, whichever way the
package is launched.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_LIBS = os.path.join(_ROOT, ".pylibs")

_done = False


def ensure_local_libs() -> str | None:
    """Prepend ``.pylibs`` to ``sys.path`` once. Returns the path if it exists."""
    global _done
    if _done:
        return _LOCAL_LIBS if os.path.isdir(_LOCAL_LIBS) else None
    _done = True
    if os.path.isdir(_LOCAL_LIBS):
        if _LOCAL_LIBS in sys.path:
            sys.path.remove(_LOCAL_LIBS)
        sys.path.insert(0, _LOCAL_LIBS)
        return _LOCAL_LIBS
    return None
