# mini-vLLM

一个**可读、可运行、可度量**的迷你 LLM 推理引擎。用约 2200 行 Python 复刻了 vLLM 的核心机制——分页 KV Cache、连续批处理、Chunked Prefill、抢占与重算——并套上一个 OpenAI 兼容的 HTTP 服务。模型是一个 128 维、4 层、8 头、1000 词表的手写 Transformer（见 [`手写/`](手写/)），权重可直接与原始 `Infra` 模型互相加载。

> 设计目标不是跑赢 vLLM，而是让 **scheduler / paged attention / prefill-vs-decode 成本结构** 这些概念在几千行代码里看得见、测得出。

## 核心特性

- **分页 KV Cache**：固定块池 + 每序列 block table，`(2, layers, blocks, block_size, heads, head_dim)` 单张缓存张量，无 `torch.cat` 逐序列增长
- **连续批处理**：每步先准入新请求（prefill），再让常驻序列各解码一个 token，长生成不阻塞新请求
- **Chunked Prefill**：长 prompt 按 token 预算分块计算，可与 decode 混在同一个 token 预算内
- **Recompute 抢占**：块池不足时驱逐序列并重算，`num_computed_tokens` 归零、整条 token 列表重放
- **prefill / decode 分开度量**：`/stats` 实时给出两阶段的 tokens/s 与耗时，差距即推理引擎的固有成本结构
- **OpenAI 兼容 API**：`/v1/completions`、`/v1/chat/completions`，支持 SSE 流式；自带浏览器 playground，无 CDN、零构建
- **离线基准**：`bench.py` 分阶段测 prefill / decode 吞吐，数据全部来自引擎真实 `step()`

## 快速开始

依赖：`torch`、`fastapi`、`uvicorn`、`pydantic`。启动会自动探测 CUDA / CPU，GPU 默认 bf16，CPU 默认 fp32。

```bash
# 启动 OpenAI 兼容服务（默认 127.0.0.1:8000）
python serve.py

# 指定设备与端口
python serve.py --device cuda --port 8000

# 加载训练好的权重（.pt / .safetensors）
python serve.py --weights checkpoints/model.pt
```

启动后在浏览器打开 `http://127.0.0.1:8000/` 即可使用自带的 playground 流式生成。

### 离线生成

```python
from mini_vllm import LLMEngine, SamplingParams

engine = LLMEngine()            # 自动从 ./手写 读取词表
engine.warmup()

result = engine.generate(
    "word_12 word_40 word_53",
    SamplingParams(max_tokens=32, temperature=0.0),
)
print(result.text)              # 词表级模型，token 形如 word_0 ... word_995
print(result.prompt_tokens, result.finish_reason, f"{result.latency*1e3:.1f} ms")
```

### 基准与测试

```bash
# prefill vs decode 吞吐对比（引擎真实调度路径）
python -m mini_vllm.bench --device cuda --batch-sizes 1,8,32 --prompt-len 128 --gen-len 32

# 单元测试（scheduler / kv_manager / paged attention / engine / api）
pytest tests/ -v
```

## API 端点

| 端点 | 说明 |
|---|---|
| `GET /` | 浏览器 playground（内联页面，无外部依赖） |
| `GET /health` | 存活检查 + device / dtype / 未完成序列数 |
| `GET /v1/models` | 当前服务模型 |
| `POST /v1/completions` | 文本补全，支持 `stream`（SSE）与 `stop` |
| `POST /v1/chat/completions` | 聊天补全，支持 `stream`（SSE） |
| `GET /stats` | 引擎内部状态：KV 用量、prefill/decode 步数与速率 |
| `GET /metrics` | Prometheus 文本格式指标 |
| `GET /docs` | Swagger UI |

## 架构

```
                 schedule()
+-----------+  ------------->  +-----------+  +-------------+
| Scheduler |  prefills(块化)  | EngineCore|->| ModelRunner |  前向 (B, vocab)
|           |  decodes(1 token)|  step()   |  | (模型+采样)  |----> logits
+-----------+                  +-----------+  +-------------+
      |                            ^                |
      | allocate_slots             | kv_cache       | sample
      v                            |                v
+-------------+                    |          下一个 token id
| KVCacheMgr  | <------------------+
| (块池+块表)  |
+-------------+
```

一个 `step()` = 至多两次前向（prefill 组一次、decode 组一次）。刻意拆分：decode 组每序列 `query_len == 1`，零 padding 浪费，且让两阶段成本差异可直接测量。

### 模块职责

| 模块 | 职责 |
|---|---|
| `config.py` | 扁平 `EngineConfig` dataclass，所有可调参数一处集中 |
| `sequence.py` | `Sequence` 生命周期、`SamplingParams`、重算前的状态复位 |
| `kv_manager.py` | 分页 KV 块分配器：块表、空闲栈、slot mapping |
| `scheduler.py` | FCFS 准入 + token 预算；块不足即抢占（重算） |
| `model_runner.py` | 打包 padded 张量、构建注意力元数据、前向、greedy/top-k/top-p 采样 |
| `engine_core.py` | `step()` 循环，串起 scheduler / runner / kv，统计两阶段成本 |
| `llm_engine.py` | 离线 `LLMEngine` + 后台线程 `AsyncEngine`（结果经 asyncio 队列流式输出） |
| `api_server.py` | FastAPI 服务：OpenAI 兼容端点、SSE、playground、/stats、/metrics |
| `models/handwritten.py` | 手写 Transformer 重写为分页推理 |

## 关键设计决策

### 手写模型的 4 处修正（对照 `手写/` 原实现）

1. **注意力读取分页 KV cache**：原实现逐序列 `torch.cat` 增长 KV，每步都要重新分配；现改为写入/收集固定块池
2. **真正的 causal mask**：原实现以 `mask=None` 调用注意力，prompt token 能看到未来 token，导致 prefill 与 decode 对同一位置结论不一致
3. **弃用自带 CUDA kernel**：`multi_head_attn.cu` 在 softmax 写读之间缺 `__syncthreads()`（存在竞态），`out_off` 对部分线程写错；改用 `torch.nn.functional.scaled_dot_product_attention`
4. **不做 `torch.cat` / `F.normalize`**：与 `Infra` 实际使用的 `transform_mask.py` 行为保持一致

参数名与原始 `Infra` 模型完全一致，`load_state_dict(..., strict=True)` 可直接互换权重（见 `tools/check_weight_compat.py`）。

### KV Cache 内存

单张张量 `(2, layers, blocks, block_size, heads, head_dim)`。默认配置下：

- CUDA + fp16：约 **64 MiB**
- CPU + fp32：约 **128 MiB**

块大小为 16 token，块池 2048 块，理论最大并发 32 序列 × 1024 token。

### 正确性不变量

分页缓存复用块且**不做零填充**：块表内低于序列长度的每个槽位，在读取前必被该序列写入；等于/高于长度的槽位在注意力中被 mask 掉。

## 目录结构

```
mini_vllm/
  api_server.py    OpenAI 兼容 HTTP 服务 + playground + /stats /metrics
  bench.py         prefill vs decode 分阶段基准
  config.py        引擎配置（单一数据类）
  engine_core.py   step() 主循环
  kv_manager.py    分页 KV cache 分配器
  llm_engine.py    离线/异步引擎入口
  model_runner.py  模型加载、元数据构建、采样
  scheduler.py     连续批处理调度器
  sequence.py      序列状态机与采样参数
  tokenizer.py     词表级分词（word_0 ... word_995，无 padding）
  models/          手写 Transformer（分页推理重写版）
手写/               原始手写模型与训练相关代码
tests/             单元测试（scheduler / kv / paged attention / engine / api）
tools/             工具：权重兼容性检查、profiling、冒烟测试、离线依赖拉取等
serve.py           服务启动入口
```

## 测试

`pytest tests/ -v` 覆盖：分页注意力正确性、KV 块分配/释放、调度器准入与抢占、引擎端到端生成、API 层解析。所有测试在 CPU 上以 fp32 运行，保证数值精确。

## 许可

尚未指定。若需对外分发请补充 LICENSE 文件。
