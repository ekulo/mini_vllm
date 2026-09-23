"""Compare the exported resume .docx against the original, structurally.

The Univer Doc imports as "Modern" (pageless), so print-pdf collapses to one
page and cannot prove the two-page layout. The .docx itself still carries the
page break, so this script reads both packages (read-only) and reports what
survived: page breaks, section properties, fonts, the photo table, and the
per-paragraph text.

    python tools/check_resume_docx.py 简历-杜之初.backup.docx .dsh-tmp/简历-杜之初.new.docx
"""

from __future__ import annotations

import re
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def load(path: str):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        xml = z.read("word/document.xml").decode("utf-8")
    return names, xml


def paragraphs(xml: str) -> list[dict]:
    root = ET.fromstring(xml)
    body = root.find(f"{W}body")
    out = []
    for p in body.iter(f"{W}p"):
        texts = [t.text or "" for t in p.iter(f"{W}t")]
        text = "".join(texts)
        fonts = {r.get(f"{W}ascii") for r in p.iter(f"{W}rFonts") if r.get(f"{W}ascii")}
        ea = {r.get(f"{W}eastAsia") for r in p.iter(f"{W}rFonts") if r.get(f"{W}eastAsia")}
        sizes = {sz.get(f"{W}val") for sz in p.iter(f"{W}sz")}
        bolds = {b.get(f"{W}val") for b in p.iter(f"{W}b")}
        brks = [b.get(f"{W}type") for b in p.iter(f"{W}br")]
        out.append({
            "text": text, "fonts": fonts, "ea": ea, "sizes": sizes,
            "bold": bolds, "breaks": brks,
        })
    return out


def summary(path: str):
    names, xml = load(path)
    root = ET.fromstring(xml)
    body = root.find(f"{W}body")
    ps = paragraphs(xml)
    sect = list(body.iter(f"{W}sectPr"))
    page_breaks = len(re.findall(r'<w:br [^>]*w:type="page"', xml))
    last_rendered = len(re.findall(r"lastRenderedPageBreak", xml))
    tables = list(body.iter(f"{W}tbl"))
    return {
        "path": path,
        "zip_entries": len(names),
        "has_document_xml": "word/document.xml" in names,
        "paragraphs": len(ps),
        "nonempty_paragraphs": sum(1 for p in ps if p["text"].strip()),
        "page_breaks": page_breaks,
        "last_rendered_page_breaks": last_rendered,
        "sectPr_count": len(sect),
        "tables": len(tables),
        "table_rows": [len(list(t.iter(f"{W}tr"))) for t in tables],
        "fonts_ascii": sorted({f for p in ps for f in p["fonts"] if f}),
        "fonts_eastasia": sorted({f for p in ps for f in p["ea"] if f}),
        "sizes": sorted({s for p in ps for s in p["sizes"] if s}, key=lambda x: int(x)),
        "ps": ps,
    }


def main() -> int:
    old_path = sys.argv[1] if len(sys.argv) > 1 else "简历-杜之初.backup.docx"
    new_path = sys.argv[2] if len(sys.argv) > 2 else ".dsh-tmp/简历-杜之初.new.docx"
    a, b = summary(old_path), summary(new_path)

    keys = ["zip_entries", "has_document_xml", "paragraphs", "nonempty_paragraphs",
            "page_breaks", "last_rendered_page_breaks", "sectPr_count", "tables", "table_rows"]
    print(f"{'metric':<28}{'ORIGINAL':>22}{'NEW':>22}   same")
    print("-" * 78)
    for k in keys:
        same = "ok" if a[k] == b[k] else "DIFF"
        print(f"{k:<28}{str(a[k]):>22}{str(b[k]):>22}   {same}")
    for k in ("fonts_ascii", "fonts_eastasia", "sizes"):
        same = "ok" if a[k] == b[k] else "DIFF"
        print(f"{k:<28}{str(a[k]):>22}{str(b[k]):>22}   {same}")

    print()
    print("=== page-break position check ===")
    for label, s in (("ORIGINAL", a), ("NEW", b)):
        idx = next((i for i, p in enumerate(s["ps"]) if "page" in (p["breaks"] or [])), None)
        ctx = ""
        if idx is not None:
            before = [p["text"].strip() for p in s["ps"][:idx] if p["text"].strip()][-1:]
            after = [p["text"].strip() for p in s["ps"][idx:] if p["text"].strip()][:1]
            ctx = f" after={before!r} before={after!r}"
        print(f"  {label:<9} page-break paragraph index = {idx}{ctx}")

    print()
    print("=== text diff ===")
    ot = [p["text"] for p in a["ps"]]
    nt = [p["text"] for p in b["ps"]]
    if ot == nt:
        print("  identical")
    else:
        import difflib
        for line in difflib.unified_diff(ot, nt, "original", "new", lineterm="", n=0):
            print("  " + line[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
