# HyperGraphRAG — sag-benchmark External Method

Uses HKUDS HyperGraphRAG for hypergraph-based QA.

## Setup

Run all commands from this project directory (`external/hypergraphrag/` of the
repository root):

```bash
uv sync --frozen
```

Configuration is read from the repository root `.env` (loaded with
`--env-file ../../.env`). The following variables are honored; defaults are
shown and are safe to override:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `Qwen3.6-35B-A3B-FP8` | LLM model name |
| `LLM_BASE_URL` | `http://localhost:32768/v1` | LLM OpenAI-compatible endpoint |
| `LLM_API_KEY` | `1` | LLM API key (回退 `OPENAI_API_KEY`) |
| `LLM_TIMEOUT` | `180` | QA 路径 LLM 超时（秒），默认档 |
| `LLM_INDEX_TIMEOUT` | `600` | 建图路径 LLM 超时（秒），`Step_1` 入口 `set_index_mode()` 切换到该档 |
| `LLM_CONNECT_TIMEOUT` | `5` | LLM 端点连接超时（秒），端点不通时快速失败 |
| `LLM_ENABLE_THINKING` | unset (=0) | Set to `1` to keep the model's thinking chain; otherwise the caller disables it (`enable_thinking` / `chat_template_kwargs.enable_thinking`) |
| `EMBEDDING_MODEL_NAME` | `text-embedding-bge-large-en-v1.5` | Embedding model name |
| `EMBEDDING_BASE_URL` | `http://localhost:9990/v1` | Embedding OpenAI-compatible endpoint |
| `EMBEDDING_API_KEY` | `1` | Embedding API key |
| `EMBEDDING_DIM` | `1024` | Embedding vector dimension |
| `EMBEDDING_MAX_SEQ_LEN` | `512` | Embedding max sequence length (falls back to `EMBED_MAX_TOKENS`); truncation keeps 2 tokens headroom → 510 |
| `EMBEDDING_FUNC_MAX_ASYNC` | `16` | Max concurrent embedding calls |
| `LLM_FUNC_MAX_ASYNC` | `32` | Max concurrent LLM calls (falls back to `MAX_ASYNC`; kernel default 32) |

### LLM sampling & behavior (built-in)

LLM 采样/行为参数不再从 `.env` 读取，内建在 `hypergraphrag_config.py` 的
`llm_model_func`（与 pipeline `settings.py` 默认值对齐）：

- **temperature 分路径**：`system_prompt` 为 None（建图/关键词抽取）用 `0.7`；非 None（QA 答案生成）用 `0.0`（确定性）。
- **中性采样参数**：`max_tokens=30000`、`top_p=1.0`、`frequency_penalty=0.0`、`presence_penalty=0.0`；`top_k`/`min_p`/`repetition_penalty` 保持关闭值、不下发。
- **`LLM_ENABLE_THINKING`** 是唯一保留的 env 覆盖开关（默认 0 = 关闭思考链），需要时临时设 `1`。
- **LLM 超时**是运行环境参数（非采样参数），故保留 env 覆盖：默认走 QA 档 `LLM_TIMEOUT=180`，
  `Step_1` 建图进程通过 `set_index_mode()` 切到 `LLM_INDEX_TIMEOUT=600`。connect 超时固定
  `LLM_CONNECT_TIMEOUT=5` 不随 read 超时放大，端点不通时快速失败。

The dataset source files are always read from the repository-root `dataset/`
directory, never from this project.

## 3-Step Pipeline

```bash
uv run --frozen --env-file ../../.env python reproduce/Step_0_load_dataset.py --data_name test_hotpotqa
uv run --frozen --env-file ../../.env python reproduce/Step_1_build_index.py --data_name test_hotpotqa
uv run --frozen --env-file ../../.env python reproduce/Step_3_response.py --data_name test_hotpotqa
```

## Output Locations

索引（图/向量库）在 `caches/<ds>/`，QA 响应在 `outputs/<ds>/response/<run_id>/`——同一张图可以跑多次 QA，每次一个 `run_id` 目录，`manifest.json` 记录本次响应用了哪个索引和什么检索参数。

| Artifact | Path |
|----------|------|
| Index | `caches/<ds>/`（graphml/vdb/kv_store_*） |
| Response | `outputs/<ds>/response/<run_id>/hybrid_<ds>_result.json` |
| Run manifest | `outputs/<ds>/response/<run_id>/manifest.json` |

### 同一图谱跑多个 Response

Step_1 建的图（`caches/<ds>/`）是共享的，Step_3 只读它——可以在**同一张图**上跑多次 QA，每次生成一个独立的 `run_id` 目录，互不覆盖。

```bash
# 新建一个具名 run（推荐：名字体现本次检索/生成配置，便于区分）
uv run --frozen --env-file ../../.env python reproduce/Step_3_response.py \
    --data_name test_hotpotqa --run-id exp_v2 --no_resume

# 或不命名，自动新建一个时间戳 run（YYYYMMDD_HHMMSS）
uv run --frozen --env-file ../../.env python reproduce/Step_3_response.py \
    --data_name test_hotpotqa --no_resume

# 中断后继续某个 run（只补跑该 run 里失败/缺漏的题）
uv run --frozen --env-file ../../.env python reproduce/Step_3_response.py \
    --data_name test_hotpotqa --resume-run-id exp_v2
```

要点：

- **不带参数 = 续跑最近的时间戳 run**。若该 run 已全部成功，会直接退出（`全部已完成，无需重跑`）——所以另起新 run 务必加 `--no_resume`。
- `--no_resume` 阻止接管旧 run；`--run-id` 决定输出落在哪个目录。
- Judge 默认取**最新的时间戳 run**；具名 run（字母名）需用 `--source-run-id exp_v2` 显式指定。

## Convert & Judge

After Step 3, run both commands from the repository root. Omit
`--source-run-id` to select the newest timestamped response run.

```bash
# 1. Convert the newest Step 3 response.
uv run --frozen --env-file .env python scripts/run_llm_judge.py convert \
  --project hypergraphrag \
  --datasets <ds>

# 2. Run generation and retrieval Judge metrics.
uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate \
  --project hypergraphrag \
  --dataset <ds>
```

需要覆盖默认发现路径时，`--input-root` 应指向包含 QA 响应的 `outputs/`。

CLI parameter: `--top-k K` limits Judge evaluation to the first K rows of the canonical predictions file, preserving file order; omit it to evaluate all rows.

Judge results are written to
`evaluation/hypergraphrag/<ds>/<source_run_id>/llmjudge/<judge-model>/<judge-run-id>/`.
