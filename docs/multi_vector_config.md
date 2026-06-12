# multi_vector.py 配置说明

本文档说明 `pipeline/modules/search/multi_vector.py` 当前实际使用的配置项。

配置类位置：

```text
pipeline/modules/search/config.py
MultiConfig
```

注意：`MultiConfig` 仍被旧版 `multi.py` / `multi1` / `hopllm` 流程复用，所以部分字段会保留兼容，但不一定被 `multi_vector.py` 使用。

## 当前模式

`multi_vector.py` 现在只有两个模式：

```python
MultiConfig(
    strategy="multi",
    mode="fast",      # fast 或 precise
    max_sections=10,
)
```

### fast 模式

快速模式使用 BM25 从实体索引召回实体，然后用少量 seed event 做一跳扩展，最后统一按 query-content 向量相似度排序取 chunk。

```text
query
-> entity_vectors.name BM25 召回 entity，取 entity_top_k=20
-> 取前 fast_entity_k=5 个 entity
-> entity -> event 候选，最多 fast_entity_event_candidate_k=20
-> 候选 event 按 query-content 相似度粗排，取 event1=fast_entity_event_k=20
-> query -> event 直接向量召回，取 event2=fast_query_event_k=20
-> event1 + event2 合并
-> seed_score 选第一跳 seed event=fast_answer_k=5
-> seed event -> entity
-> expand
-> 扩展 event 按 query-content 相似度粗排，取 fast_expand_answer_k=5
-> seed + expanded 按 query-content 相似度统一排序
-> event -> chunk
```

### precise 模式

精准模式保留原本的 LLM 过滤流程，只把入口实体召回从 spaCy/实体向量改成 query BM25。

```text
query
-> entity_vectors.name BM25 召回 entity，取 entity_top_k=20
-> entity -> event，最多 40
-> query -> event 直接向量召回，取 multi_top_k=20
-> 两路 event 合并
-> 初始 event -> entity
-> expand
-> 初始 event + 扩展 event 合并
-> query-content 粗排，最多 max_events=100
-> LLM 过滤，取 max_sections=10
-> event -> chunk
-> 如果 chunk 不足 max_sections，用 query -> chunk 原生向量检索补齐
```

## 通用配置

这些字段会被 `multi_vector.py` 直接读取。

| 字段 | 默认值 | 使用位置 | 说明 |
| --- | ---: | --- | --- |
| `mode` | `"fast"` | 搜索入口 | `fast` 使用快速 seed-expand 流程，`precise` 使用 LLM 过滤流程 |
| `entity_top_k` | `20` | Step1 | query -> entity BM25 最大返回实体数量 |
| `multi_top_k` | `20` | precise Step3 | query -> event 向量召回数量；precise 的 entity -> event 通道固定使用 40 |
| `similarity_threshold` | `0.4` | event 向量召回 | query -> event 最低相似度阈值 |
| `max_hops` | `1` | expand | 扩展跳数；`0` 表示不扩展 |
| `max_expand_events_per_hop` | `2000` | expand | 每跳 expand 阶段 entity -> event 最大召回数量 |
| `max_events` | `100` | precise 粗排 | precise 模式最多保留多少个候选 event 给 LLM |
| `max_sections` | `10` | 输出阶段 | 最终返回 chunk 数量上限，也是 precise LLM 过滤的 top_k |

## fast 专用配置

这些字段只影响 `mode="fast"`。

| 字段 | 默认值 | 说明 |
| --- | ---: | --- |
| `fast_entity_k` | `5` | BM25 实体召回后，只取前 N 个 entity 作为 seed entity |
| `fast_entity_event_candidate_k` | `20` | seed entity -> event 初始候选数量 |
| `fast_entity_event_k` | `20` | entity 候选 event 按 query-content 粗排后保留数量，即 event1 |
| `fast_query_event_k` | `20` | query -> event 直接向量召回数量，即 event2 |
| `fast_answer_k` | `5` | event1 + event2 合并后，按 seed_score 选择第一跳 event 数量 |
| `fast_expand_answer_k` | `5` | 扩展 event 粗排后保留数量 |
| `fast_vector_weight` | `0.85` | seed_score 中 query-content 归一化向量分权重 |
| `fast_entity_weight` | `0.15` | seed_score 中实体命中增强权重；命中任一 seed entity 记 1，否则记 0 |
| `fast_channel_weight` | `0.05` | seed_score 中双通道命中奖励权重 |

fast 的 `seed_score` 公式：

```text
seed_score =
  fast_vector_weight  * vector_score_norm
+ fast_entity_weight  * entity_hit_score
+ fast_channel_weight * channel_score
```

其中：

```text
vector_score_norm: 当前候选集合内归一化后的 query-content 相似度
entity_hit_score: 命中任意 seed entity 记 1，否则记 0
channel_score: 同时来自 entity 和 query 两个通道记 1，否则记 0
```

`seed_score` 只用于 fast 模式 expand 前选择第一跳 seed event。最终 chunk 顺序仍按 query-content 相似度统一排序。

## 兼容字段

这些字段在 `MultiConfig` 中保留，但当前 `multi_vector.py` 主搜索入口不使用。

| 字段 | 说明 |
| --- | --- |
| `spacy_model` | 旧 spaCy 实体抽取流程兼容字段；当前入口实体召回已经改为 BM25 |
| `key_similarity_threshold` | 旧实体向量召回阈值；当前 BM25 实体召回只按 top_k 截断 |
| `strategy` | 旧版 `multi.py` / benchmark 策略标识使用，`multi_vector.py` 自己只看 `mode` |
| `rerank_top_k` | 旧版 rerank 字段；`multi_vector.py` precise 模式使用 `max_sections` 控制 LLM 输出数量 |
| `max_events_a` | `multi1` / `hopllm` 阶段 A 使用 |
| `max_events_b` | `multi1` / `hopllm` 阶段 B 使用 |
| `max_hop_retries` | `multi1` / `hopllm` 动态扩跳使用 |

## 关键日志

常见日志名：

| 日志 | 含义 |
| --- | --- |
| `[multi_es.start]` | 当前模式、ranking、entity_top_k、multi_top_k、entity_event_top_k |
| `[entity.bm25]` | query -> entity BM25 输入/候选/输出数量 |
| `[entity.bm25.entities]` | BM25 召回的实体明细、原始 `_score` 和是否保留 |
| `[event.seed]` | fast 模式第一跳 seed event 选择统计 |
| `[event.recall]` | precise 模式 entity->event 与 query->event 双通道召回统计 |
| `[event.rank.coarse]` | event 按 query-content 向量相似度粗排统计 |
| `[fast.rank]` | fast 模式 seed + expanded 最终统一排序的输入/输出/分数范围 |
| `[precise.done]` | precise 模式 LLM 过滤和 chunk 返回统计 |

## 示例配置

### 快速模式

```python
from pipeline.modules.search.config import MultiConfig

config = MultiConfig(
    strategy="multi",
    mode="fast",
    max_sections=10,
    entity_top_k=20,
    max_hops=1,
    fast_entity_k=5,
    fast_entity_event_candidate_k=20,
    fast_entity_event_k=20,
    fast_query_event_k=20,
    fast_answer_k=5,
    fast_expand_answer_k=5,
)
```

### 精准模式

```python
from pipeline.modules.search.config import MultiConfig

config = MultiConfig(
    strategy="multi",
    mode="precise",
    max_sections=10,
    entity_top_k=20,
    multi_top_k=20,
    max_hops=1,
    max_events=100,
)
```

## 常用命令

单独测试 `multi_vector.py`：

```bash
python scripts/test_multi_vector.py \
  --source-config-id musique-20260512_213908 \
  --mode fast \
  --output summary
```

benchmark 中使用 `multi_es`：

```bash
python scripts/run_search_benchmark.py \
  --dataset-name musique \
  --strategy multi_es \
  --mode fast \
  --top-k 10 \
  --limit 1
```
