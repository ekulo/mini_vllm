"""Dump the border / shading / table-style definitions of the original resume.

These are the design elements (left blue accent bar, summary box, table borders)
that a lossy docx round-trip would silently drop.

    python tools/dump_resume_borders.py 简历-杜之初.backup.docx
"""

from __future__ import annotations

import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SIDES = ("top", "left", "bottom", "right", "between", "insideH", "insideV", "bar")


def q(node, name):
    return node.get(f"{W}{name}")


def describe(el) -> list[str]:
    out = []
    for pb in el.iter(f"{W}pBdr"):
        parts = []
        for side in SIDES:
            n = pb.find(f"{W}{side}")
            if n is not None:
                parts.append(f"{side}(sz={q(n, 'sz')} val={q(n, 'val')} color={q(n, 'color')} space={q(n, 'space')})")
        out.append("pBdr -> " + ", ".join(parts))
    for sh in el.iter(f"{W}shd"):
        out.append(f"shd(fill={q(sh, 'fill')} val={q(sh, 'val')} color={q(sh, 'color')})")
    for tb in el.iter(f"{W}tblBorders"):
        parts = []
        for side in SIDES:
            n = tb.find(f"{W}{side}")
            if n is not None:
                parts.append(f"{side}(sz={q(n, 'sz')} val={q(n, 'val')} color={q(n, 'color')})")
        out.append("tblBorders -> " + ", ".join(parts))
    for ts in el.iter(f"{W}tblStyle"):
        out.append(f"tblStyle(val={q(ts, 'val')})")
    return out


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "简历-杜之初.backup.docx"
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    body = ET.fromstring(xml).find(f"{W}body")

    print(f"=== {path} ===")
    for i, blk in enumerate(body):
        tag = blk.tag.replace(W, "")
        if tag == "sectPr":
            print(f"{i:3d} sectPr (final section properties)")
            continue
        txt = "".join(t.text or "" for t in blk.iter(f"{W}t"))[:34]
        d = describe(blk)
        if tag != "p" or d:
            print(f"{i:3d} {tag:<6} {txt!r}")
            for x in d:
                print(f"          {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
