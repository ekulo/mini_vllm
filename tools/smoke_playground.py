"""Verify the browser playground: the HTML's JS contract and the SSE wire format.

Checks that the page's element ids all exist, that the JSON bodies it builds are
accepted, and that the raw SSE byte stream parses exactly the way its JS does.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
fails: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        fails.append(name)


print("== page ==")
html = urllib.request.urlopen(BASE + "/", timeout=30).read().decode()
for eid in ("sub", "prompt", "max_tokens", "temperature", "top_p", "go", "clr",
            "stream", "ignore_eos", "out", "status", "ttft", "total", "ntok", "tps",
            "stats", "bstats", "bmetrics"):
    check(f"element #{eid} exists", f'id="{eid}"' in html)
check("no external resources (CSP-safe / offline OK)",
      not re.search(r'(src|href)\s*=\s*["\']https?://', html))

print("== non-stream (what the page does with the box unchecked) ==")
req = urllib.request.Request(
    BASE + "/v1/completions",
    data=json.dumps({"prompt": "word_12 word_40", "max_tokens": 6, "ignore_eos": True,
                     "temperature": 0, "top_p": 1, "stream": False}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
obj = json.loads(urllib.request.urlopen(req, timeout=60).read())
check("choices[0].text present", isinstance(obj["choices"][0]["text"], str),
      repr(obj["choices"][0]["text"][:40]))
check("usage.completion_tokens == 6", obj["usage"]["completion_tokens"] == 6)
check("finish_reason present", obj["choices"][0]["finish_reason"] is not None,
      obj["choices"][0]["finish_reason"])

print("== stream: parse the raw bytes the way the page's JS does ==")
req = urllib.request.Request(
    BASE + "/v1/completions",
    data=json.dumps({"prompt": "word_7 word_8", "max_tokens": 6, "ignore_eos": True,
                     "temperature": 0, "top_p": 1, "stream": True}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
resp = urllib.request.urlopen(req, timeout=60)
check("content-type is text/event-stream",
      "text/event-stream" in resp.headers.get("Content-Type", ""),
      resp.headers.get("Content-Type", ""))

raw = b""
deltas: list[str] = []
finish = None
saw_done = False
usage_tokens = None
buf = ""
for chunk in resp:  # browsers read via ReadableStream; same bytes either way
    raw += chunk
    buf += chunk.decode("utf-8")
    while "\n\n" in buf:
        line, buf = buf.split("\n\n", 1)
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            saw_done = True
            continue
        ev = json.loads(payload)
        ch = ev["choices"][0]
        if ch["text"]:
            deltas.append(ch["text"])
        if ch["finish_reason"]:
            finish = ch["finish_reason"]
        if ev.get("usage"):
            usage_tokens = ev["usage"]["completion_tokens"]

check("frames are 'data: ...\\n\\n'", re.match(rb"data: \{.*\}\n\n", raw) is not None)
check("stream ends with [DONE]", saw_done)
check("got text deltas", 1 <= len(deltas) <= 6, f"{len(deltas)}: {deltas}")
check("usage.completion_tokens == 6 (counts special tokens too)",
      usage_tokens == 6, f"usage={usage_tokens}, visible={len(deltas)}")
check("finish_reason delivered", finish is not None, str(finish))
print(f"       assembled text: {''.join(deltas)!r}")

print("== early EOS still stops the stream (ignore_eos off) ==")
req = urllib.request.Request(
    BASE + "/v1/completions",
    data=json.dumps({"prompt": "word_7 word_8", "max_tokens": 64, "ignore_eos": False,
                     "temperature": 0, "stream": True}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
resp = urllib.request.urlopen(req, timeout=60)
n_tok = 0
finish2 = None
for chunk in resp:
    for line in chunk.decode().split("\n"):
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        ch = json.loads(line[6:])["choices"][0]
        if ch["text"]:
            n_tok += 1
        if ch["finish_reason"]:
            finish2 = ch["finish_reason"]
check("early EOS produces finish_reason=stop", finish2 == "stop",
      f"{n_tok} tokens, finish={finish2}")

print("== error path the page reports ==")
try:
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps({"prompt": "word_1", "max_tokens": 99999}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=30)
    check("overlong rejected with 400", False)
except urllib.error.HTTPError as e:
    check("overlong rejected with 400", e.code == 400, f"HTTP {e.code}")

print()
print("FAILURES:", fails if fails else "none")
raise SystemExit(1 if fails else 0)
