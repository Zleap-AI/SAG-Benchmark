# 搜索基准运行说明

本文档对应当前版本的 `scripts/run_search_benchmark.py`。脚本会加载指定数据集，使用一个搜索策略检索每道题的候选片段，并计算 Recall、Precision、F1 等指标。

## 当前支持的策略

`--strategy` 的公开实验选项为以下四个值：

| 策略 | 实现/用途 | 特殊参数 |
| --- | --- | --- |
| `atomic` | Atomic 检索与实体扩展，使用 `AtomicConfig` | `--top-k` |
| `sag2` | SAG2 图、事件和实体链路检索 | `--sag2-*` |
| `vector` | 纯向量检索，使用 `VectorConfig` | `--top-k` |
| `bm25` | Elasticsearch BM25 检索，使用 `BM25Config` | `--top-k` |

SAG2 事件候选池是一个独立的实验变体：命令行仍使用 `--strategy sag2`，但加上 `--sag2-scope` 后，先按查询相似度构造有界事件候选池，再执行 SAG2 的召回、扩展和重排。它应在实验记录中与 `vector`、`atomic` 等策略并列；`bm25` 也可以直接作为一级检索策略运行。

## 运行前提

请在仓库根目录运行脚本，并确保项目依赖已经安装。运行前在 `.env` 中配置 LLM、embedding、Elasticsearch/MySQL 等依赖。数据集对应的 `source_info.json` 默认从以下目录按时间戳选择最新上传结果：

```text
pipeline/evaluation/source/SAG/<LLM_MODEL>/<dataset-name>/<timestamp>/source_info.json
```

如果不希望自动选择，使用 `--source-config-id` 指定确切的 `source_config_id`。脚本会根据 source 信息激活对应的 embedding 维度和物理索引。

## 基本命令

```bash
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy atomic
```

示例：

```bash
# SAG2 检索，并发处理 4 道题
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique --strategy sag2 \
  --max-concurrency 4 --bench-size 100

# SAG2 启用事件候选池，并显式设置 rerank 和最终保留数量
uv run python scripts/run_search_benchmark.py \
  --dataset-name hotpotqa --strategy sag2 \
  --sag2-scope --sag2-event-top-k 500 \
  --sag2-rerank-top-k 10 --sag2-max-results 10

# BM25 关键词检索基线
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique --strategy bm25 --top-k 10

# 只跑第 0 到 49 题（端点包含）
uv run python scripts/run_search_benchmark.py \
  --dataset-name test_hotpotqa --strategy bm25 \
  --limit 0 49 --k-values 1,5,10

# 使用 MLflow 记录实验，并从 Prompt Registry 读取搜索提示词
uv run python scripts/run_search_benchmark.py \
  --dataset-name musique --strategy vector \
  --use-mlflow --mlflow-url http://127.0.0.1:5000 \
  --mlflow-experiment sag-benchmark \
  --use-mlflow-prompts --mlflow-prompt-alias latest
```

## 命令行参数

| 参数 | 默认值/取值 | 使用方法 |
| --- | --- | --- |
| `--dataset-name` | 必填 | 数据集名称，例如 `musique`、`hotpotqa`、`test_hotpotqa`。 |
| `--strategy` | 必填 | 公开实验选项为 `atomic`、`sag2`、`vector`、`bm25`。 |
| `--top-k` | `10` | 通用候选数量；对 SAG2 之外的策略直接限制最终检索结果，对 SAG2 用作默认 section 上限。 |
| `--sag2-rerank-top-k` | `None`（读取配置，默认 10） | SAG2 rerank 模型保留的最高 K 个结果。不能大于 `--sag2-max-results`。 |
| `--sag2-max-results` | `None`（读取配置，默认 10） | SAG2 最终返回的事项数；rerank 不足时按 embedding 相似度补齐。 |
| `--k-values` | `1,2,5,10` | 计算哪些 K 值的指标，逗号分隔，例如 `1,5,10`。 |
| `--max-concurrency` | `1` | 并发处理的问题数；外部 LLM/ES 服务不稳定时保持 `1`。 |
| `--limit` | 不限制 | 一个整数 `N` 表示前 `N` 题；两个整数 `S E` 表示 0-based 且包含端点的题目区间；最多两个整数。 |
| `--bench-size` | `5` | 每处理 N 道题打印一次累积统计并触发进度回调；它不限制题目总数，也不改变并发度。 |
| `--sag2-scope` | 关闭 | 开启 SAG2 事件候选池变体；不传则使用普通 SAG2 检索范围。该配置应与 vector、atomic 等策略并列比较。 |
| `--sag2-event-top-k` | `1000` | 候选池大小 `k_pool`：在 SAG2 scope 中按 query 相似度选取的事件数量，例如 `500`。 |
| `--sag2-bootstrap-entity-limit` | `0` | 初始化时加载的 event-entity 关系上限；`0` 表示不限制。 |
| `--sag2-no-event-content` | 关闭 | 不把事件正文保留在内存候选全集中；只在需要降低内存时开启。 |
| `--output-dir` | `output/<dataset>/<strategy>/<timestamp>/` | 指定输出目录；不传时按默认路径和当前时间戳创建。 |
| `--source-config-id` | 自动查找 | 直接指定 source 配置 ID，跳过最新 `source_info.json` 查找。 |
| `--allow-embedding-mismatch` | 关闭 | 允许当前 embedding 维度与索引维度不一致。仅用于排查问题，不建议用于可比实验。 |
| `--use-mlflow` | 关闭 | 开启 MLflow 实验记录。 |
| `--mlflow-url` | `.env` 中的 `MLFLOW_URL` | MLflow Tracking Server 地址；命令行值优先。 |
| `--mlflow-experiment` | `sag-benchmark` | MLflow 实验名称；实际名称会带本机 IP 前缀。 |
| `--use-mlflow-prompts` | 关闭 | 从 MLflow Prompt Registry 加载搜索提示词；未注册时回退到代码常量，服务不可达则报错。 |
| `--mlflow-prompt-alias` | `latest` | 与 `--use-mlflow-prompts` 一起使用的 Prompt Registry alias。 |

### SAG2 参数关系

`--sag2-rerank-top-k` 是 rerank 阶段上限，`--sag2-max-results` 是最终返回数量；前者不能大于后者。若不显式设置，两者读取 SAG2 配置（当前默认均为 10）。`--sag2-scope` 打开后，再用 `--sag2-event-top-k`、`--sag2-bootstrap-entity-limit` 和 `--sag2-no-event-content` 控制事件候选全集的规模和内存内容。

## 输出文件

默认目录为：

```text
output/<dataset-name>/<strategy>/<timestamp>/
```

也可以用 `--output-dir` 指定。目录内主要有：

- `run.log`：本次运行日志。
- `search_results.json`：逐题检索结果，每条记录包含 `question_index`、`question` 和 `retrieved_docs`；后者对应脚本内部的 `sections`（`title\ncontent` 字符串列表）。
- `benchmark_results.json`：最终指标和运行元数据，包含 `metrics`、`statistics`、`timings`、`search_diagnostics`、`llm_token_usage`、`metadata` 等字段。

`metadata` 会记录数据集、策略、`top_k`、`k_values`、题目数、搜索耗时和时间戳；开启 MLflow 后还会记录命令、实际配置、source_config_id 和输出目录。

新实验优先使用 `sag2`。

## 指标与复现实验注意事项

脚本会按 `--k-values` 计算 Precision@K、Recall@K、F1@K，并输出完整召回、部分召回、零召回和成功检索数量等统计。比较不同实验时应固定数据集、source_config_id、embedding 模型/维度、策略参数和 `k-values`。

不要在正式对比实验中使用 `--allow-embedding-mismatch`；它会放宽索引维度校验，结果可能不可比。
