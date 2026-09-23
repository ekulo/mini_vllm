"""Download + unpack a CUDA build of PyTorch without pip.

The sandbox blocks pip's temporary-directory churn, but plain HTTPS from Python
works, so we resolve the wheel URL from the PyTorch index and unzip it ourselves.

Usage:
    python tools/fetch_cuda_torch.py                 # newest cu128 cp312 win wheel
    python tools/fetch_cuda_torch.py --index cu128 --py cp312
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / ".pylibs"
CACHE = ROOT / ".pylibs" / "_wheels"


def resolve_url(index: str, pytag: str, plat: str) -> str:
    page = f"https://download.pytorch.org/whl/{index}/torch/"
    html = urllib.request.urlopen(page, timeout=60).read().decode("utf-8", "ignore")
    # Filenames appear both url-encoded (2B) and raw (+) in the index.
    pat = re.compile(
        rf"torch-(\d+(?:\.\d+)+)(?:%2B|\+)([A-Za-z0-9.]+)-{pytag}-{pytag}-{plat}\.whl"
    )
    versions = {tuple(int(p) for p in m.group(1).split(".")): m.group(2) for m in pat.finditer(html)}
    if not versions:
        raise SystemExit(f"no wheel for {pytag}/{plat} on {page}")

    ver, tag = max(versions.items(), key=lambda kv: kv[0])
    ver_s = ".".join(str(p) for p in ver)
    name = f"torch-{ver_s}+{tag}-{pytag}-{pytag}-{plat}.whl"
    # The CDN rejects a literal '+' in the path (403); it must be %2B encoded.
    return f"https://download.pytorch.org/whl/{index}/{name.replace('+', '%2B')}", name


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded ({dest.stat().st_size / 2**30:.2f} GiB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        step = 256 * 1024 * 1024
        nxt = step
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got >= nxt:
                pct = f" ({got / total * 100:.1f}%)" if total else ""
                print(f"  ... {got / 2**30:.2f} GiB{pct}", flush=True)
                nxt += step
    tmp.replace(dest)
    print(f"[ok] downloaded {dest.name} ({dest.stat().st_size / 2**30:.2f} GiB)")


def unpack(wheel: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {wheel.name} -> {target}")
    with zipfile.ZipFile(wheel) as z:
        members = [m for m in z.infolist() if not m.filename.endswith("/")]
        for i, m in enumerate(members, 1):
            z.extract(m, target)
            if i % 2000 == 0:
                print(f"  ... {i}/{len(members)}", flush=True)
    print(f"[ok] unpacked {len(members)} entries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="cu128")
    ap.add_argument("--py", default=f"cp{sys.version_info.major}{sys.version_info.minor}")
    ap.add_argument("--plat", default="win_amd64")
    args = ap.parse_args()

    url, name = resolve_url(args.index, args.py, args.plat)
    print(f"[pick] {name}")
    wheel = CACHE / name
    download(url, wheel)
    unpack(wheel, TARGET)
    print("\nNow run with:  $env:PYTHONPATH='M:\\dsh\\.pylibs'")
    print("  python -c \"import torch;print(torch.__version__, torch.cuda.is_available())\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
