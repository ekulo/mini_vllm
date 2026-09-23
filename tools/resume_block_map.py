"""Report the paragraph-property signature of every top-level block.

Blocks that share a signature are formatted identically, which tells us which
template can safely be cloned when adding bullets.

    python tools/resume_block_map.py 简历-杜之初.backup.docx
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ET.register_namespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "简历-杜之初.backup.docx"
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    body = ET.fromstring(xml).find(f"{W}body")
    blks = list(body)
    print(f"{len(blks)} top-level blocks")
    groups: dict[str, list[int]] = {}
    for i, b in enumerate(blks):
        tag = b.tag.replace(W, "")
        ppr = b.find(f"{W}pPr")
        sig = hashlib.md5(ET.tostring(ppr, encoding="unicode").encode()).hexdigest()[:8] if ppr is not None else "none"
        groups.setdefault(sig, []).append(i)
        if i <= 22 or tag != "p":
            text = "".join(x.text or "" for x in b.iter(f"{W}t"))[:30]
            runs = len([r for r in b if r.tag == f"{W}r"])
            print(f"{i:3d} {tag:<5} pPr={sig} runs={runs} {text!r}")
    print()
    for sig, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  pPr={sig}  -> blocks {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
