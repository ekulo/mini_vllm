"""Dump the raw OOXML of specific top-level body blocks of the resume.

Used to derive exact run/paragraph-property templates before doing a surgical,
format-preserving text substitution.

    python tools/dump_resume_blocks.py 简历-杜之初.backup.docx 1 4 8 12 18
"""

from __future__ import annotations

import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ET.register_namespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")


def main() -> int:
    path = sys.argv[1]
    wanted = {int(x) for x in sys.argv[2:]} if len(sys.argv) > 2 else None
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    body = ET.fromstring(xml).find(f"{W}body")

    for i, blk in enumerate(body):
        tag = blk.tag.replace(W, "")
        text = "".join(t.text or "" for t in blk.iter(f"{W}t"))
        if wanted is not None and i not in wanted:
            if tag == "tbl" or not wanted:
                pass
            continue
        print("=" * 78)
        print(f"block {i}  <{tag}>  text={text[:50]!r}")
        print("-" * 78)
        s = ET.tostring(blk, encoding="unicode")
        s = s.replace("><", ">\n<")
        print(s[:3000])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
