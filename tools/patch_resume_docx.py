"""Surgically update the resume .docx, preserving every other byte of the package.

Why not the normal document toolchain: the Univer docx exporter drops the
paragraph immediately after a table and never writes ``w:pBdr``, so the blue
accent bar and the five section underlines would be lost. This script instead
edits ``word/document.xml`` as text: it only rewrites the ``<w:r>``/``<w:t>``
runs of the nine paragraphs that change, cloning run properties from templates
that already exist in the document. Styles, settings, font table, borders,
section properties and all other package parts are untouched.

    python tools/patch_resume_docx.py 简历-杜之初.backup.docx 简历-杜之初.docx
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

PARA_RE = re.compile(r"<w:p(?:\s[^>]*)?>.*?</w:p>", re.S)
T_RE = re.compile(r"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)", re.S)

BULLET_ANCHOR = "• 性能成果："          # any existing bullet, used as the run template
TITLE_ANCHOR = "迷你 LLM 推理引擎（个人项目"
SUMMARY_ANCHOR = "计算机科学与技术硕士，研究方向多模态大模型安全"
OLD_VLLM_ANCHOR = "vLLM 推理引擎源码研读与部署实践"

# (anchor, bold lead, body).  lead + body is the whole paragraph text.
BULLETS = [
    ("• 链路：", "• 引擎架构：", "从零实现与 vLLM 对应的五层结构 —— API Server（FastAPI，OpenAI 兼容 /v1/completions 与 SSE 流式）、EngineCore（step 主循环）、Scheduler（连续批处理）、KVManager（分页 KV 块分配）、ModelRunner（模型前向与采样）；引擎核心约 1.3k 行 Python，模型侧沿用此前手写的 4 层 Transformer（自定义 tokenizer + pre-LN decoder），并以 torch C++ extension + pybind11 手写 GEMM 替换 nn.Linear"),
    ("• CUDA 算子：", "• 分页 KV Cache：", "把 KV 显存切成固定大小 block 池，每条序列只持有 block table（逻辑位置 → 物理槽位映射），按需分配、不复制不搬迁；实测 4096 block × 16 token = 128 MiB 承载 65536 个 token 槽位，并保证序列复用旧 block 时不读到脏数据（长度以内先写后读，长度以外被 mask）"),
    ("• 性能剖析：", "• 连续批处理与调度：", "实现 iteration 级批处理（prefill 与 decode 混合同一步）、token 预算驱动的 chunked prefill、块池耗尽时的重计算抢占（还块并重放 prompt 与已生成 token）；32 请求下较原版逐请求串行循环（torch.cat 拼接式 KV Cache）吞吐提升约 10×，且 32/32 条序列输出与其逐 token 一致"),
    (OLD_VLLM_ANCHOR, "• 性能实测与调优：", "RTX 5060 Ti / bf16 下 batch 64 时 prefill 达 8×10⁵ token/s（0.001 ms/token），decode 7×10³ token/s（0.13 ms/token），单 token 成本相差约 100×，实测印证 prefill 算力密集、decode 访存密集；用 torch.profiler 拆分 decode 单步，定位瓶颈在 CPU 算子派发与 host↔device 同步而非算力，并把注意力元数据由每层重建改为每步构建一次、4 层共享，decode 单步 12.2 → 4.4 ms"),
    ("• 机制理解：", "• 正确性保障：", "50 项测试覆盖「prefill + N 步 decode 的 logits == 整段一次前向」「批间隔离」「非连续块表」「块复用」「抢占后不泄漏」等不变量；过程中定位并修复手写注意力 kernel 的 softmax 竞态（缺 __syncthreads 与输出偏移错误）与 tokenizer 特殊 token 序号错位"),
    ("• 部署实践：", "• 真实 vLLM 对照：", "研读 vLLM 调度与显存管理源码（PagedAttention 的 Block 分配、Continuous Batching 请求级调度、Block Manager 引用计数），并用 Docker 容器化部署真实 vLLM 服务，跑通 OpenAI 兼容 API、配置张量并行与 gpu_memory_utilization / max_model_len 显存参数完成并发验证"),
]

NEW_TITLE = "迷你 vLLM 推理引擎（从零实现核心机制 ｜ CUDA + PyTorch）"

NEW_SUMMARY_BOLD = "计算机科学与技术硕士，研究方向多模态大模型安全，具备 3 年企业工作经历。"
NEW_SUMMARY_REST = "底层到服务化都能做：从零手写 CUDA 算子并走完性能调优闭环（FlashAttention 于 RTX 5060 Ti 实现 320 倍加速与 9.0 TFLOPS）；从零实现 mini-vLLM 推理引擎（分页 KV Cache、连续批处理、抢占式调度、OpenAI 兼容 API），实测 prefill 达 8×10⁵ token/s、较原版串行循环吞吐提升约 10×；以第一作者完成大模型安全评测论文，在单张 12GB 显卡上跑完 4B 参数模型训练。"

SKILLS_ANCHOR = "• 推理引擎与部署："
SKILLS_LEAD = "• 推理引擎与部署："
SKILLS_BODY = "从零实现 vLLM 核心机制（PagedAttention 分页 KV Cache 与 block 分配、Continuous Batching、chunked prefill、重计算抢占调度）；FastAPI 服务化与 OpenAI 兼容 API / SSE 流式；Docker 容器化部署、张量并行与显存参数配置；prefill/decode 分阶段基准与 TTFT/TPOT 口径设计"


def find_para(xml: str, anchor: str, count: int = 1) -> tuple[int, int, str]:
    hits = [m for m in PARA_RE.finditer(xml) if anchor in m.group(0)]
    if len(hits) != count:
        raise SystemExit(f"anchor {anchor!r}: expected {count} paragraph(s), found {len(hits)}")
    m = hits[0]
    return m.start(), m.end(), m.group(0)


def rpr_of(run_xml: str) -> str:
    m = re.search(r"<w:rPr>.*?</w:rPr>", run_xml, re.S)
    if not m:
        raise SystemExit("run has no <w:rPr>")
    return m.group(0)


def runs_of(para_xml: str) -> list[str]:
    return [m.group(0) for m in re.finditer(r"<w:r>.*?</w:r>", para_xml, re.S)]


def open_tag_of(para_xml: str) -> str:
    """The paragraph's own ``<w:p ...>`` opening tag, attributes included."""
    return re.match(r"<w:p(?:\s[^>]*)?>", para_xml).group(0)


def set_bold(rpr: str, bold: bool) -> str:
    """Rewrite the run's bold flag, inserting one if the run has none."""
    target = "<w:b />" if bold else '<w:b w:val="0" />'
    if re.search(r"<w:b\s*/>", rpr):
        return re.sub(r"<w:b\s*/>", target, rpr, count=1)
    if re.search(r'<w:b\s+w:val="[^"]*"\s*/>', rpr):
        return re.sub(r'<w:b\s+w:val="[^"]*"\s*/>', target, rpr, count=1)
    return rpr.replace("<w:sz", target + "<w:sz", 1)


def build_run(rpr_template: str, text: str, bold: bool) -> str:
    return f'<w:r>{set_bold(rpr_template, bold)}<w:t xml:space="preserve">{text}</w:t></w:r>'


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "简历-杜之初.backup.docx")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "简历-杜之初.docx")
    if dst.exists():
        shutil.copy2(dst, dst.with_suffix(".before-patch.docx"))

    with zipfile.ZipFile(src) as z:
        parts = {n: z.read(n) for n in z.namelist()}
        infos = {n: z.getinfo(n) for n in z.namelist()}
    original_xml = parts["word/document.xml"].decode("utf-8")
    xml = original_xml

    # ---- templates, taken verbatim from paragraphs already in the document ----
    _, _, bullet_para = find_para(xml, BULLET_ANCHOR)
    bullet_runs = runs_of(bullet_para)
    assert len(bullet_runs) == 2, f"bullet template has {len(bullet_runs)} runs"
    BOLD_RPR = rpr_of(bullet_runs[0])
    NORM_RPR = rpr_of(bullet_runs[1])
    bullet_ppr = re.search(r"<w:pPr>.*?</w:pPr>", bullet_para, re.S).group(0)

    _, _, summary_para = find_para(xml, SUMMARY_ANCHOR)
    summary_rpr = rpr_of(runs_of(summary_para)[0])
    assert "F2F6FB" in summary_rpr, "summary run lost its light-blue shading"

    log: list[str] = []

    # ---- 1..6: the six engine bullets (the 4th replaces the old vLLM title) ----
    for anchor, lead, body in BULLETS:
        s, e, para = find_para(xml, anchor)
        ts = list(T_RE.finditer(para))
        if anchor == OLD_VLLM_ANCHOR:
            # a section title becomes a bullet: swap in the bullet templates
            new_para = (f"{open_tag_of(para)}{bullet_ppr}"
                        f"{build_run(BOLD_RPR, lead, True)}{build_run(NORM_RPR, body, False)}</w:p>")
            log.append(f"paragraph({anchor!r}) title -> bullet, {len(ts)} <w:t> replaced")
        else:
            assert len(ts) == 2, f"{anchor!r}: expected 2 <w:t>, got {len(ts)}"
            new_para = para
            for m, text in zip(reversed(ts), [body, lead]):
                new_para = new_para[:m.start(2)] + text + new_para[m.end(2):]
            log.append(f"paragraph({anchor!r}) -> {lead!r} ({len(body)} chars body)")
        xml = xml[:s] + new_para + xml[e:]

    # ---- entry title ----
    s, e, para = find_para(xml, TITLE_ANCHOR)
    ts = list(T_RE.finditer(para))
    assert len(ts) >= 1, "title paragraph has no <w:t>"
    m = ts[0]
    if m.group(2) != NEW_TITLE:
        para = para[:m.start(2)] + NEW_TITLE + para[m.end(2):]
    xml = xml[:s] + para + xml[e:]
    log.append(f"title -> {NEW_TITLE!r}")

    # ---- summary: single run -> bold lead run + normal run, shading kept ----
    s, e, para = find_para(xml, SUMMARY_ANCHOR)
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", para, re.S).group(0)
    new_para = (
        f"{open_tag_of(para)}{ppr}"
        f"{build_run(summary_rpr, NEW_SUMMARY_BOLD, True)}"
        f"{build_run(summary_rpr, NEW_SUMMARY_REST, False)}"
        f"</w:p>"
    )
    xml = xml[:s] + new_para + xml[e:]
    log.append("summary -> bold lead sentence + normal remainder (shading preserved)")

    # ---- skills line ----
    s, e, para = find_para(xml, SKILLS_ANCHOR)
    ts = list(T_RE.finditer(para))
    assert len(ts) == 2, f"skills line: expected 2 <w:t>, got {len(ts)}"
    new_para = para
    for m, text in zip(reversed(ts), [SKILLS_BODY, SKILLS_LEAD]):
        new_para = new_para[:m.start(2)] + text + new_para[m.end(2):]
    xml = xml[:s] + new_para + xml[e:]
    log.append("skills '推理引擎与部署' replaced")

    # ---- sanity: nothing but the intended paragraphs changed ----
    assert xml.count("<w:pBdr>") == original_xml.count("<w:pBdr>"), "paragraph borders changed!"
    assert xml.count("<w:tcBorders>") == original_xml.count("<w:tcBorders>"), "cell borders changed!"
    assert xml.count("<w:tblBorders>") == original_xml.count("<w:tblBorders>"), "table borders changed!"
    assert xml.count("<w:sectPr") == original_xml.count("<w:sectPr"), "section properties changed!"
    # The summary goes from one run to two, so it gains one run-level shading --
    # but no shading colour may appear or disappear.
    fills = lambda s: sorted(set(re.findall(r'<w:shd[^>]*w:fill="([^"]+)"', s)))
    assert fills(xml) == fills(original_xml), f"shading colours changed: {fills(original_xml)} -> {fills(xml)}"
    assert xml.count("<w:shd") >= original_xml.count("<w:shd"), "shading was lost!"
    para_count = lambda s: len(re.findall(r"<w:p(?:\s[^>]*)?>", s))
    assert para_count(xml) == para_count(original_xml), \
        f"paragraph count changed: {para_count(original_xml)} -> {para_count(xml)}"

    # ---- write the package, byte-identical except document.xml ----
    parts["word/document.xml"] = xml.encode("utf-8")
    tmp = dst.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            info = infos[name]
            zi = zipfile.ZipInfo(name, date_time=info.date_time)
            zi.compress_type = info.compress_type
            zi.external_attr = info.external_attr
            z.writestr(zi, data)
    tmp.replace(dst)

    print(f"patched {src.name} -> {dst.name}")
    for line in log:
        print("  -", line)
    print(f"  document.xml {len(original_xml)} -> {len(xml)} chars; "
          f"{len(parts)} package parts rewritten unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
