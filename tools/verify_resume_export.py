"""Verify a Univer-exported resume .docx against the pristine original.

Checks that nothing was lost or restyled beyond the intended content changes:
paragraph inventory, page-break/section properties, fonts, the photo table, and
the bold lead-in pattern of each bullet.

    python tools/verify_resume_export.py 简历-杜之初.backup.docx .dsh-tmp/resume-v3.docx
"""

from __future__ import annotations

import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def blocks(path: str):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        xml = z.read("word/document.xml").decode("utf-8")
        settings = z.read("word/settings.xml").decode("utf-8") if "word/settings.xml" in names else ""
        styles = z.read("word/styles.xml").decode("utf-8") if "word/styles.xml" in names else ""
    body = ET.fromstring(xml).find(f"{W}body")
    paras = []
    for p in body.iter(f"{W}p"):
        runs = []
        for r in p.iter(f"{W}r"):
            txt = "".join(t.text or "" for t in r.iter(f"{W}t"))
            rpr = r.find(f"{W}rPr")
            bold = False
            fonts = set()
            sizes = set()
            if rpr is not None:
                b = rpr.find(f"{W}b")
                bold = b is not None and b.get(f"{W}val") not in ("0", "false")
                for rf in rpr.iter(f"{W}rFonts"):
                    for k in ("ascii", "eastAsia"):
                        if rf.get(f"{W}{k}"):
                            fonts.add(rf.get(f"{W}{k}"))
                for sz in rpr.iter(f"{W}sz"):
                    if sz.get(f"{W}val"):
                        sizes.add(int(sz.get(f"{W}val")) // 2)
            if txt:
                runs.append({"t": txt, "b": bold, "fonts": fonts, "sizes": sizes})
        paras.append({
            "text": "".join(x["t"] for x in runs),
            "runs": runs,
        })
    return {
        "path": path,
        "zip": sorted(n for n in names if not n.endswith("/")),
        "document_xml": xml,
        "paras": paras,
        "tables": len(list(body.iter(f"{W}tbl"))),
        "tbl_cells": [len(list(t.iter(f"{W}tc"))) for t in body.iter(f"{W}tbl")],
        "sectPr": len(list(body.iter(f"{W}sectPr"))),
        "page_breaks": xml.count('w:type="page"'),
        "settings_len": len(settings),
        "styles_len": len(styles),
        "has_fontTable": "word/fontTable.xml" in names,
    }


def main() -> int:
    old_p = sys.argv[1] if len(sys.argv) > 1 else "简历-杜之初.backup.docx"
    new_p = sys.argv[2] if len(sys.argv) > 2 else ".dsh-tmp/resume-v3.docx"
    a, b = blocks(old_p), blocks(new_p)
    fails: list[str] = []

    def check(label, ok, extra=""):
        print(f"  [{'ok ' if ok else 'FAIL'}] {label} {extra}")
        if not ok:
            fails.append(label)

    print("== structural parity ==")
    check("paragraph count", len(a["paras"]) == len(b["paras"]), f"{len(a['paras'])} -> {len(b['paras'])}")
    check("tables", a["tables"] == b["tables"], f"{a['tables']} -> {b['tables']}")
    check("table cells", a["tbl_cells"] == b["tbl_cells"], f"{a['tbl_cells']} -> {b['tbl_cells']}")
    check("sectPr (page-break container)", a["sectPr"] == b["sectPr"], f"{a['sectPr']} -> {b['sectPr']}")
    check("explicit page breaks", a["page_breaks"] == b["page_breaks"], f"{a['page_breaks']} -> {b['page_breaks']}")

    print("== content diff (should be your edits only) ==")
    at = [p["text"] for p in a["paras"]]
    bt = [p["text"] for p in b["paras"]]
    changed = [i for i, (x, y) in enumerate(zip(at, bt)) if x != y]
    for i in changed:
        print(f"  para {i}:")
        print(f"    - {at[i][:70]}")
        print(f"    + {bt[i][:70]}")
    print(f"  {len(changed)} paragraphs changed")

    print("== bold lead-in pattern of the new bullets ==")
    expected = {14: "• 引擎架构：", 15: "• 分页 KV Cache：", 16: "• 连续批处理与调度：",
                17: "• 性能实测与调优：", 18: "• 正确性保障：", 19: "• 真实 vLLM 对照：",
                23: "• 推理引擎与部署："}
    for idx, lead in expected.items():
        runs = b["paras"][idx]["runs"]
        if not runs:
            check(f"para {idx} has runs", False)
            continue
        head = runs[0]
        ok = head["b"] and head["t"] == lead and all(not r["b"] for r in runs[1:])
        check(f"para {idx} lead bold == {lead!r}", ok,
              f"got bold={head['b']} text={head['t'][:22]!r} runs={len(runs)}")

    print("== fonts / sizes preserved ==")
    def fontset(blk):
        return sorted({f for p in blk["paras"] for r in p["runs"] for f in r["fonts"]})
    def sizeset(blk):
        return sorted({s for p in blk["paras"] for r in p["runs"] for s in r["sizes"]})
    check("font families", fontset(a) == fontset(b), f"{fontset(a)} vs {fontset(b)}")
    check("font sizes", sizeset(a) == sizeset(b), f"{sizeset(a)} vs {sizeset(b)}")

    print("== package parts ==")
    print(f"  original entries: {len(a['zip'])}   new entries: {len(b['zip'])}")
    missing = [n for n in a["zip"] if n not in b["zip"]]
    if missing:
        print(f"  missing in export: {missing}")
    check("styles.xml not gutted", b["styles_len"] >= a["styles_len"] * 0.5,
          f"{a['styles_len']} -> {b['styles_len']} bytes")
    check("settings.xml not gutted", b["settings_len"] >= a["settings_len"] * 0.5,
          f"{a['settings_len']} -> {b['settings_len']} bytes")
    check("fontTable.xml present", b["has_fontTable"], "yes" if b["has_fontTable"] else "MISSING")

    print()
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
