"""Install pure-python wheels without pip.

pip's temp-directory juggling trips the file sandbox, but HTTPS + zipfile work
fine from Python. This resolves the newest wheel for each requested package
(plus its non-extra dependencies) from the PyPI JSON API and unpacks everything
into ``.pylibs``, which you then put on ``PYTHONPATH``.

Usage:
    python tools/fetch_pypi_wheels.py fastapi "uvicorn[standard]"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / ".pylibs"
CACHE = TARGET / "_wheels"

# Already present in the interpreter we run with; never re-download.
SKIP = {
    "numpy", "pydantic", "pydantic-core", "pydantic_core", "typing-extensions",
    "typing_extensions", "setuptools", "pip", "wheel", "torch", "filelock",
    "sympy", "networkx", "jinja2", "fsspec", "markupsafe", "mpmath",
}

UA = {"User-Agent": "mini-vllm-bootstrap/0.1"}


def _get(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)


def _pick_wheel(meta: dict, py_tag: str) -> str | None:
    urls = meta.get("urls") or []
    for f in urls:
        if f.get("packagetype") == "bdist_wheel" and f["filename"].endswith(".whl"):
            return f["url"]
    return None


def _deps(meta: dict) -> list[str]:
    out = []
    for req in meta.get("info", {}).get("requires_dist") or []:
        if ";" in req and "extra" in req.split(";", 1)[1]:
            continue  # optional extra
        name = req.split(";")[0].split("(")[0].split("[")[0].split("<")[0]
        name = name.split(">")[0].split("=")[0].split("!")[0].split("~")[0].strip()
        if name:
            out.append(name)
    return out


def resolve(name: str, seen: set[str], order: list[str]) -> None:
    key = name.lower().replace("_", "-")
    if key in seen or key in SKIP:
        return
    seen.add(key)
    try:
        meta = json.load(_get(f"https://pypi.org/pypi/{name}/json"))
    except Exception as exc:  # pragma: no cover
        print(f"[warn] cannot resolve {name}: {exc}")
        return
    url = _pick_wheel(meta, "")
    if url is None:
        print(f"[warn] no wheel for {name}")
        return
    order.append(name)
    for dep in _deps(meta):
        resolve(dep, seen, order)
    _DOWNLOADS[name] = url


_DOWNLOADS: dict[str, str] = {}


def fetch_and_unpack(name: str, url: str) -> None:
    fname = url.rsplit("/", 1)[-1]
    dest = CACHE / fname
    if not dest.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[get] {fname}")
        with _get(url) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    else:
        print(f"[skip] {fname}")
    with zipfile.ZipFile(dest) as z:
        z.extractall(TARGET)
    print(f"[ok] {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="+")
    args = ap.parse_args()

    TARGET.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    order: list[str] = []
    for pkg in args.packages:
        resolve(pkg.split("[")[0], seen, order)

    # Unpack dependencies first so a partial run still leaves a usable tree.
    for name in order:
        fetch_and_unpack(name, _DOWNLOADS[name])
    print(f"\n[done] {len(order)} wheels -> {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
