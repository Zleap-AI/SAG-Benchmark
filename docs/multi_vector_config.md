# multi_vector.py 配置说明

本文档只说明 `pipeline/modules/search/multi_vector.py` 当前实际读取的配置项。

配置类位于：

```text
pipeline/modules/search/config.py
MultiConfig
```

当前 `MultiConfig` 也被旧版 `multi.py` / `multi1` / `hopllm` 流程复用，所以有些字段虽然在 `MultiConfig` 中存在，但并不被 `multi_vector.py` 使用。

## 当前检索模式

`multi_vector.py` 通过 `mode` 区分快速模式和精准模式：

```python
MultiConfig(
    strategy="multi",
    mode="fast",      # fast 或 precise
    max_sections=10,
)
```

### fast 模式

快速模式用于少量 seed event 加一跳扩展，最后按 query-content 相似度排序后取 chunk。

流程：

```text
query
-> spaCy 抽实体
-> entity 向量检索
-> 取前 fast_entity_k 个 entity
-> entity -> event 候选
-> 候选 event 按 query-content 相似度取 event1
-> query -> event 直接向量召回取 event2
-> event1 + event2 合并
-> seed_score 选择第一跳 seed event
-> seed event -> entity
-> expand
-> 扩展 event 粗排
-> seed + expanded 按 query-content 相似度统一排序
-> event -> chunk
```

### precise 模式

精准模式保留原来的 LLM 过滤流程。

流程：

```text
query
-> spaCy 抽实体
-> entity 向量检索
-> entity -> event + query -> event 双通道召回
-> 初始 event -> entity
-> expand
-> 初始 event + 扩展 event 合并
-> query-content 粗排
-> LLM 过滤
-> event -> chunk
```

## 通用配置

这些字段会被 `multi_vector.py` 直接读取。

| 字段 | 默认值 | 使用位置 | 说明 |
| --- | ---: | --- | --- |
| `mode` | `"fast"` | 搜索入口 | `fast` 使用快速 seed-expand 流程，`precise` 使用 LLM 过滤流程 |
| `spacy_model` | `"en_core_web_sm"` | Step1 | spaCy NER 模型名 |
| `entity_top_k` | `20` | Step2 | 每个 query entity 最多召回多少个相似实体 |
| `key_similarity_threshold` | `0.9` | Step2 | entity 向量检索最低相似度阈值 |
| `multi_top_k` | `20` | precise Step3 | query -> event 直接向量召回数量 |
| `similarity_threshold` | `0.4` | event 向量召回 | event 向量召回最低分数 |
| `max_hops` | `1` | Step5 | 扩展跳数，`0` 表示不扩展 |
| `max_events` | `100` | precise Step6 | precise 粗排最多保留多少个 event 给 LLM |
| `max_expand_events_per_hop` | `2000` | Step5 | 每跳 entity -> event 最多召回多少个新 event |
| `max_sections` | `10` | 输出阶段 | 最终返回 chunk 数量上限，也是 precise LLM 过滤的 top_k |

## fast 专用配置

这些字段只影响 `mode="fast"`。

| 字段 | 默认值 | 说明 |
| --- | ---: | --- |
| `fast_entity_k` | `5` | Step2 后只取前 N 个 entity 作为 seed entity |
| `fast_entity_event_candidate_k` | `2000` | seed entity -> event 初始候选数量 |
| `fast_entity_event_k` | `20` | entity 候选 event 按 query-content 粗排后保留数量，即 event1 |
| `fast_query_event_k` | `20` | query -> event 直接召回数量，即 event2 |
| `fast_answer_k` | `5` | event1 + event2 合并后，按 seed_score 选择第一跳 event 数量 |
| `fast_expand_answer_k` | `5` | 扩展 event 粗排后保留数量 |
| `fast_vector_weight` | `0.85` | seed_score 中 query-content 归一化向量分权重 |
| `fast_entity_weight` | `0.15` | seed_score 中是否命中 seed entity 的加分权重 |
| `fast_channel_weight` | `0.05` | seed_score 中同时命中 entity/query 双通道的加分权重 |

fast 的 seed_score 公式：

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

注意：`seed_score` 只用于 fast 模式 expand 前选择第一跳 seed event。最终 chunk 顺序仍按 query-content 相似度统一排序。

## multi_vector.py 不使用的 MultiConfig 字段

这些字段在 `MultiConfig` 中存在，但当前 `multi_vector.py` 不读取。

| 字段 | 说明 |
| --- | --- |
| `strategy` | 旧版 `multi.py` / benchmark 策略标识使用，`multi_vector.py` 自己只看 `mode` |
| `rerank_top_k` | 旧版 LLM rerank 使用，`multi_vector.py` 的 precise 模式使用 `max_sections` 控制 LLM 输出数量 |
| `max_events_a` | `multi1` / `hopllm` 阶段 A 使用 |
| `max_events_b` | `multi1` / `hopllm` 阶段 B 使用 |
| `max_hop_retries` | `multi1` / `hopllm` 动态扩跳使用 |

如果只服务 `multi_vector.py`，这些字段可以不出现在 multi_es 的日志或实验参数里。

不要直接从 `MultiConfig` 删除这些字段，除非先把旧版 `multi.py`、`multi1`、`hopllm` 和 `multi_vector.py` 的配置类拆开。

## 示例配置

### 快速模式

```python
from pipeline.modules.search.config import MultiConfig

config = MultiConfig(
    strategy="multi",
    mode="fast",
    max_sections=10,
    entity_top_k=20,
    key_similarity_threshold=0.9,
    max_hops=1,
    fast_entity_k=5,
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
    key_similarity_threshold=0.9,
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

## 配置删减建议

短期建议：

```text
保留 MultiConfig 字段不动
从 multi_es 的日志 / MLflow 参数中去掉 multi_vector.py 不使用的字段
```

长期建议：

```text
新增 MultiESConfig
只给 pipeline/modules/search/multi_vector.py 使用
旧 MultiConfig 继续给 multi.py / multi1 / hopllm 使用
```

这样可以避免一个配置类同时承载多套检索流程，减少后续维护时的歧义。
