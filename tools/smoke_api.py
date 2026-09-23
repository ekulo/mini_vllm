"""Smoke-test the HTTP surface of a running mini-vLLM server.

    python tools/smoke_api.py --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def post(base: str, path: str, body: dict, stream: bool = False):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    r = urllib.request.urlopen(req, timeout=120)
    if not stream:
        return json.loads(r.read().decode())
    events = []
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data: "):
            events.append(line[6:])
    return events


def get(base: str, path: str, raw: bool = False):
    r = urllib.request.urlopen(base + path, timeout=30)
    data = r.read().decode()
    return data if raw else json.loads(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    failures = []

    def check(name, cond, extra=""):
        print(f"  [{'ok ' if cond else 'FAIL'}] {name} {extra}")
        if not cond:
            failures.append(name)

    print("== GET /health ==")
    h = get(base, "/health")
    check("health", h.get("status") == "ok", str(h))

    print("== GET /v1/models ==")
    m = get(base, "/v1/models")
    check("models", len(m["data"]) == 1, m["data"][0]["id"])

    print("== POST /v1/completions ==")
    out = post(base, "/v1/completions", {"prompt": "word_12 word_40", "max_tokens": 5, "ignore_eos": True})
    check("completion shape", out["object"] == "text_completion")
    check("completion usage", out["usage"]["completion_tokens"] == 5, str(out["usage"]))
    print("       text:", repr(out["choices"][0]["text"]))

    print("== POST /v1/completions (stream) ==")
    events = post(base, "/v1/completions", {"prompt": "word_7", "max_tokens": 4, "stream": True, "ignore_eos": True}, stream=True)
    check("stream terminates", events[-1] == "[DONE]", f"{len(events)} events")
    payloads = [json.loads(e) for e in events[:-1]]
    deltas = [p["choices"][0]["text"] for p in payloads]
    check("stream deltas", len(deltas) >= 4, repr(deltas))
    check("stream finish", payloads[-1]["choices"][0]["finish_reason"] == "length",
          str(payloads[-1]["choices"][0]["finish_reason"]))

    print("== POST /v1/chat/completions ==")
    chat = post(base, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "word_3 word_9"}],
        "max_tokens": 4, "ignore_eos": True,
    })
    check("chat shape", chat["object"] == "chat.completion")
    print("       text:", repr(chat["choices"][0]["message"]["content"]))

    print("== stop string ==")
    first = post(base, "/v1/completions", {"prompt": "word_1", "max_tokens": 1, "stream": False, "ignore_eos": True})
    token = first["choices"][0]["text"].split()[0]
    stopped = post(base, "/v1/completions", {"prompt": "word_1", "max_tokens": 20,
                                             "stop": [token], "stream": False, "ignore_eos": True})
    check("stop truncates", token not in stopped["choices"][0]["text"], repr(stopped["choices"][0]["text"]))

    print("== GET /stats ==")
    s = get(base, "/stats")
    check("stats has prefill", s["prefill_tokens"] > 0)
    check("stats has decode", s["decode_tokens"] > 0)
    check("kv drained", s["kv_cache"]["blocks_used"] == 0, str(s["kv_cache"]))

    print("== GET /metrics ==")
    metrics = get(base, "/metrics", raw=True)
    check("metrics text", "minivllm_decode_tokens_total" in metrics)

    print("== validation ==")
    try:
        post(base, "/v1/completions", {"prompt": "word_1", "max_tokens": 100000})
        check("rejects overlong", False)
    except urllib.error.HTTPError as e:
        check("rejects overlong", e.code == 400, f"HTTP {e.code}")

    print()
    print("FAILURES:", failures if failures else "none")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
