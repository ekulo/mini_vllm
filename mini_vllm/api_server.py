"""OpenAI-compatible HTTP server for the mini-vLLM engine.

Endpoints
---------
    GET  /                    browser playground (type a prompt, watch it stream)
    GET  /health              liveness + device
    GET  /v1/models           the (single) served model
    POST /v1/completions      text completion, ``stream`` supported
    POST /v1/chat/completions chat completion, ``stream`` supported
    GET  /stats               engine internals: KV usage, prefill/decode rates
    GET  /metrics             Prometheus text format
    GET  /docs                Swagger UI

Run it:
    python -m mini_vllm.api_server --port 8000
or  python serve.py --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

from mini_vllm._env import ensure_local_libs

ensure_local_libs()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .config import EngineConfig  # noqa: E402
from .llm_engine import AsyncEngine, LLMEngine  # noqa: E402
from .sequence import SamplingParams  # noqa: E402

MODEL_NAME = "handwritten-128"

# Self-contained playground so the server can be verified from a browser with no
# extra tooling. No CDN, no build step, no dependencies -- one inline page.
PLAYGROUND_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-vLLM playground</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; background: #14161a; color: #e6e8eb;
         font: 14px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .wrap { max-width: 940px; margin: 0 auto; }
  h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  .sub { color: #8b93a1; font-size: 12px; margin-bottom: 18px; }
  .sub b { color: #7dd3a0; font-weight: 600; }
  .card { background: #1b1e24; border: 1px solid #2a2f38; border-radius: 10px;
          padding: 16px; margin-bottom: 14px; }
  label { display: block; color: #8b93a1; font-size: 12px; margin-bottom: 6px; }
  textarea, input[type=number] { width: 100%; background: #12141a; color: #e6e8eb;
          border: 1px solid #2a2f38; border-radius: 7px; padding: 9px 11px;
          font: inherit; outline: none; }
  textarea:focus, input:focus { border-color: #4c7dff; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
  .row > div { flex: 1 1 90px; }
  button { background: #4c7dff; color: #fff; border: 0; border-radius: 7px;
           padding: 9px 20px; font: inherit; font-weight: 600; cursor: pointer; }
  button:hover { background: #5f8bff; }
  button:disabled { background: #333a46; color: #6c7482; cursor: default; }
  button.ghost { background: #232830; color: #c3cad6; font-weight: 400; padding: 9px 14px; }
  button.ghost:hover { background: #2c323c; }
  #out { white-space: pre-wrap; word-break: break-word; min-height: 84px;
         background: #12141a; border: 1px solid #2a2f38; border-radius: 7px;
         padding: 12px; color: #d7dbe2; }
  .meta { display: flex; gap: 16px; flex-wrap: wrap; color: #8b93a1;
          font-size: 12px; margin-top: 10px; }
  .meta span b { color: #c9d1e0; font-weight: 600; }
  .ex { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .ex a { color: #8fb4ff; cursor: pointer; font-size: 12px;
          border-bottom: 1px dashed #40506e; text-decoration: none; }
  .err { color: #ff8a8a; }
  details summary { cursor: pointer; color: #8b93a1; font-size: 12px; }
  pre { background: #12141a; border: 1px solid #2a2f38; border-radius: 7px;
        padding: 12px; overflow: auto; max-height: 320px; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>mini-vLLM playground</h1>
  <div class="sub" id="sub">loading…</div>

  <div class="card">
    <label for="prompt">prompt（这是词表级模型，词表里是 word_0 … word_995 这种 token）</label>
    <textarea id="prompt" rows="2">word_12 word_40 word_53</textarea>
    <div class="ex">
      <a data-p="word_961">word_961</a>
      <a data-p="word_498 word_783">word_498 word_783</a>
      <a data-p="word_12 word_40 word_53 word_343 word_242">长一点</a>
      <a data-p="word_961 hello word_498">含未知词</a>
      <a data-p="">空 prompt</a>
    </div>
    <div class="row" style="margin-top:14px">
      <div><label for="max_tokens">max_tokens</label><input id="max_tokens" type="number" value="24" min="1"></div>
      <div><label for="temperature">temperature</label><input id="temperature" type="number" value="0" min="0" step="0.1"></div>
      <div><label for="top_p">top_p</label><input id="top_p" type="number" value="1" min="0.01" max="1" step="0.05"></div>
      <div style="flex:0 0 auto"><button id="go">生成</button></div>
      <div style="flex:0 0 auto"><button id="clr" class="ghost">清空</button></div>
    </div>
    <div class="meta">
      <label style="margin:0"><input type="checkbox" id="stream" checked> 流式 (SSE)</label>
      <label style="margin:0"><input type="checkbox" id="ignore_eos" checked> 忽略 &lt;EOS&gt;（跑满 max_tokens）</label>
    </div>
  </div>

  <div class="card">
    <label>输出</label>
    <div id="out"></div>
    <div class="meta">
      <span>状态 <b id="status">idle</b></span>
      <span>首 token <b id="ttft">-</b></span>
      <span>总耗时 <b id="total">-</b></span>
      <span>token 数 <b id="ntok">-</b></span>
      <span>tok/s <b id="tps">-</b></span>
    </div>
  </div>

  <div class="card">
    <div class="row">
      <div style="flex:0 0 auto"><button class="ghost" id="bstats">刷新 /stats</button></div>
      <div style="flex:0 0 auto"><button class="ghost" id="bmetrics">/metrics</button></div>
      <div style="flex:0 0 auto"><button class="ghost" id="bdocs" onclick="window.open('/docs','_blank')">/docs</button></div>
    </div>
    <details style="margin-top:12px"><summary>引擎内部状态</summary><pre id="stats">-</pre></details>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const out = $('out');

fetch('/health').then(r => r.json()).then(h => {
  $('sub').innerHTML = 'model <b>' + h.model + '</b> · device <b>' + h.device +
                       '</b> · dtype <b>' + h.dtype + '</b> · unfinished <b>' + h.unfinished + '</b>';
}).catch(e => { $('sub').textContent = 'health 请求失败: ' + e; });

document.querySelectorAll('.ex a').forEach(a => a.onclick = () => { $('prompt').value = a.dataset.p; });
$('clr').onclick = () => { out.textContent = ''; reset(); };

function reset() {
  $('status').textContent = 'idle'; $('ttft').textContent = '-';
  $('total').textContent = '-'; $('ntok').textContent = '-'; $('tps').textContent = '-';
}

$('go').onclick = async () => {
  const btn = $('go'); btn.disabled = true;
  out.textContent = ''; out.className = ''; reset();
  $('status').textContent = 'running';
  const body = {
    prompt: $('prompt').value,
    max_tokens: parseInt($('max_tokens').value || '16', 10),
    temperature: parseFloat($('temperature').value || '0'),
    top_p: parseFloat($('top_p').value || '1'),
    ignore_eos: $('ignore_eos').checked,
    stream: $('stream').checked,
  };
  const t0 = performance.now(); let first = null, n = 0;
  try {
    const res = await fetch('/v1/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    if (!res.ok) { throw new Error('HTTP ' + res.status + ': ' + (await res.text())); }

    if (!body.stream) {
      const obj = await res.json();
      out.textContent = obj.choices[0].text;
      if (first === null) { first = performance.now(); $('ttft').textContent = (first - t0).toFixed(0) + ' ms'; }
      n = obj.usage.completion_tokens;
      $('status').textContent = 'finish: ' + obj.choices[0].finish_reason;
    } else {
      const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') continue;
          const obj = JSON.parse(payload);
          const ch = obj.choices[0];
          if (ch.text) {
            if (first === null) { first = performance.now(); $('ttft').textContent = (first - t0).toFixed(0) + ' ms'; }
            out.textContent += ch.text; n++;
          }
          if (ch.finish_reason) $('status').textContent = 'finish: ' + ch.finish_reason;
          // The final frame carries the authoritative token count (it includes
          // special tokens that render as no text).
          if (obj.usage && typeof obj.usage.completion_tokens === 'number') n = obj.usage.completion_tokens;
        }
      }
    }
  } catch (e) {
    out.className = 'err'; out.textContent = '错误: ' + e.message;
    $('status').textContent = 'error';
  } finally {
    const dt = performance.now() - t0;
    $('total').textContent = dt.toFixed(0) + ' ms';
    $('ntok').textContent = n;
    $('tps').textContent = n > 1 && dt > 0 ? ((n - 1) / (dt / 1000)).toFixed(1) : '-';
    if ($('status').textContent === 'running') $('status').textContent = 'done';
    btn.disabled = false;
  }
};

$('bstats').onclick = async () => { $('stats').textContent = JSON.stringify(await (await fetch('/stats')).json(), null, 2); };
$('bmetrics').onclick = async () => { $('stats').textContent = await (await fetch('/metrics')).text(); };
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
#  request / response schemas
# --------------------------------------------------------------------------- #
class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]] = ""
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False
    ignore_eos: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False
    ignore_eos: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
#  text assembly (word-level vocab: tokens are joined by single spaces)
# --------------------------------------------------------------------------- #
class TextAccumulator:
    """Rebuilds text from generated ids and applies stop strings.

    ``num_tokens`` counts every sampled id; ``words`` holds only the ones that
    produced visible text. They differ whenever the model emits a special token
    (<PAD>/<BOS>/<EOS>/<UNK>), and ``usage.completion_tokens`` must follow
    ``num_tokens`` -- counting visible words under-reports generated tokens.
    """

    def __init__(self, tokenizer, stop: Optional[Union[str, List[str]]]) -> None:
        self.tokenizer = tokenizer
        if stop is None:
            self.stop: List[str] = []
        elif isinstance(stop, str):
            self.stop = [stop] if stop else []
        else:
            self.stop = [s for s in stop if s]
        self.words: List[str] = []
        self.num_tokens = 0
        self.text = ""

    def push(self, token_id: int) -> str:
        """Append one token; returns the text delta (may be empty)."""
        self.num_tokens += 1
        word = self.tokenizer.decode_token(token_id)
        if not word:
            return ""
        delta = (" " if self.words else "") + word
        self.words.append(word)
        self.text += delta
        return delta

    def stop_index(self) -> int | None:
        """Index in ``text`` where a stop string starts, if any."""
        best: int | None = None
        for s in self.stop:
            i = self.text.find(s)
            if i != -1 and (best is None or i < best):
                best = i
        return best

    def trim_delta_for_stop(self, delta: str) -> str:
        """Cut ``delta`` so the emitted text ends right before the stop string."""
        idx = self.stop_index()
        if idx is None:
            return delta
        keep = idx - (len(self.text) - len(delta))
        return delta[: max(0, keep)]


def _to_sampling_params(req) -> SamplingParams:
    try:
        sp = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            ignore_eos=req.ignore_eos,
            seed=req.seed,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return sp


def _render_chat(messages: List[ChatMessage]) -> str:
    """A minimal chat template: ``role: content`` lines, then ``assistant:``."""
    lines = [f"{m.role}: {m.content}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def _usage(prompt_tokens: int, completion: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "total_tokens": prompt_tokens + completion,
    }


# --------------------------------------------------------------------------- #
#  app
# --------------------------------------------------------------------------- #
def create_app(engine: LLMEngine | None = None, config: EngineConfig | None = None) -> FastAPI:
    state: Dict[str, Any] = {"engine": engine, "config": config}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["engine"] is None:
            state["engine"] = LLMEngine(state["config"] or EngineConfig())
        eng: LLMEngine = state["engine"]
        eng.warmup()
        async_engine = AsyncEngine(eng)
        async_engine.start(asyncio.get_running_loop())
        app.state.async_engine = async_engine
        app.state.engine = eng
        try:
            yield
        finally:
            async_engine.stop()
            eng.shutdown()

    app = FastAPI(title="mini-vLLM", version="0.1.0", lifespan=lifespan)

    def core():
        ae: AsyncEngine = app.state.async_engine
        return ae.engine.core, ae

    # ------------------------------------------------------------------ meta
    @app.get("/", include_in_schema=False)
    async def playground():
        """Browser playground -- open http://127.0.0.1:8000/ to use it."""
        return HTMLResponse(PLAYGROUND_HTML)

    @app.get("/health")
    async def health():
        eng: LLMEngine = app.state.engine
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "device": str(eng.core.device),
            "dtype": str(eng.core.model_runner.dtype).replace("torch.", ""),
            "unfinished": eng.core.num_unfinished(),
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
        }

    @app.get("/stats")
    async def stats():
        return core()[0].stats()

    @app.get("/metrics")
    async def metrics():
        s = core()[0].stats()
        lines = [
            "# HELP minivllm_kv_blocks_used KV cache blocks currently allocated",
            "# TYPE minivllm_kv_blocks_used gauge",
            f"minivllm_kv_blocks_used {s['kv_cache']['blocks_used']}",
            f"minivllm_kv_blocks_total {s['kv_cache']['blocks_total']}",
            "# HELP minivllm_running Sequences currently decoding",
            "# TYPE minivllm_running gauge",
            f"minivllm_running {s['scheduler']['running']}",
            f"minivllm_waiting {s['scheduler']['waiting']}",
            f"minivllm_prefill_tokens_total {s['prefill_tokens']}",
            f"minivllm_decode_tokens_total {s['decode_tokens']}",
            f"minivllm_prefill_tokens_per_s {s['prefill_tokens_per_s']}",
            f"minivllm_decode_tokens_per_s {s['decode_tokens_per_s']}",
            f"minivllm_steps_total {s['steps']}",
        ]
        return JSONResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    # ----------------------------------------------------------- completions
    def _sse(payload: dict | str) -> str:
        if isinstance(payload, str):
            return f"data: {payload}\n\n"
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _run_completion(
        prompt_text: str, sp: SamplingParams, stop, stream: bool, request_id: str,
        object_name: str, model_name: str,
    ):
        eng: LLMEngine = app.state.engine
        ae: AsyncEngine = app.state.async_engine
        prompt_ids = eng.encode(prompt_text)
        acc = TextAccumulator(eng.tokenizer, stop)
        created = int(time.time())

        def envelope(text: str, finish: str | None, prompt_tokens: int, done: bool):
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
            if object_name == "chat.completion":
                choice = {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }
            return {
                "id": request_id,
                "object": object_name,
                "created": created,
                "model": model_name,
                "choices": [choice],
                "usage": _usage(prompt_tokens, acc.num_tokens) if done else None,
            }

        if not stream:
            last = await ae.complete(prompt_ids, sp, meta={"prompt": prompt_text})
            for token_id in (last.output_token_ids if last else []):
                acc.push(token_id)
            idx = acc.stop_index()
            reason = "stop"
            if idx is not None:
                acc.text = acc.text[:idx]
            elif last and last.finish_reason:
                reason = last.finish_reason
            return envelope(acc.text, reason, len(prompt_ids), done=True)

        async def event_stream() -> AsyncIterator[str]:
            finish_reason = "length"
            try:
                async for out in ae.stream(prompt_ids, sp, meta={"prompt": prompt_text}):
                    delta = acc.push(out.token_id)
                    if delta:
                        idx = acc.stop_index()
                        if idx is not None:
                            delta = acc.trim_delta_for_stop(delta)
                            if delta:
                                yield _sse(envelope(delta, None, len(prompt_ids), False))
                            finish_reason = "stop"
                            break
                        yield _sse(envelope(delta, None, len(prompt_ids), False))
                    if out.finished:
                        finish_reason = out.finish_reason or "length"
                        break
                yield _sse(envelope("", finish_reason, len(prompt_ids), True))
                yield _sse("[DONE]")
            except asyncio.CancelledError:  # client hung up mid-stream
                raise

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        eng: LLMEngine = app.state.engine
        if isinstance(req.prompt, list):
            raise HTTPException(status_code=400, detail="list prompts are not supported; send one per request")
        sp = _to_sampling_params(req)
        return await _run_completion(
            req.prompt, sp, req.stop, req.stream,
            f"cmpl-{uuid.uuid4().hex[:24]}", "text_completion", req.model or MODEL_NAME,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        sp = _to_sampling_params(req)
        return await _run_completion(
            _render_chat(req.messages), sp, req.stop, req.stream,
            f"chatcmpl-{uuid.uuid4().hex[:24]}", "chat.completion", req.model or MODEL_NAME,
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error"}}, status_code=400)

    return app


# --------------------------------------------------------------------------- #
#  cli
# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace) -> EngineConfig:
    cfg = EngineConfig()
    if args.model_dir:
        cfg.model_dir = args.model_dir
    cfg.weights_path = args.weights
    cfg.device = args.device
    cfg.dtype = args.dtype
    cfg.num_blocks = args.num_blocks
    cfg.block_size = args.block_size
    cfg.max_num_seqs = args.max_num_seqs
    cfg.max_num_batched_tokens = args.max_num_batched_tokens
    cfg.max_model_len = args.max_model_len
    cfg.chunked_prefill = not args.no_chunked_prefill
    cfg.num_layers = args.num_layers
    cfg.hidden_size = args.hidden_size
    cfg.num_heads = args.num_heads
    cfg.time_steps = not args.no_timing
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM OpenAI-compatible server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    p.add_argument("--model-dir", default=None, help="folder holding vocab_infer.json (default: ./手写)")
    p.add_argument("--weights", default=None, help="checkpoint .pt; omit for random init")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--num-blocks", type=int, default=2048, help="KV cache blocks in the pool")
    p.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--no-timing", action="store_true", help="skip per-step CUDA sync")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None) -> int:
    ensure_local_libs()
    import uvicorn

    args = parse_args(argv)
    app = create_app(config=build_config(args))
    print(
        f"[mini-vllm] http://{args.host}:{args.port}  "
        f"device={args.device} block_size={args.block_size} blocks={args.num_blocks}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
