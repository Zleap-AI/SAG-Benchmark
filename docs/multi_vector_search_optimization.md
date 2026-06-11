# multi_vector.py 搜索阶段优化检查

## 当前结论

`pipeline/modules/search/multi_vector.py` 当前搜索阶段已经完成了比较关键的一步优化：

- query 和 query entities 使用一次 batch embedding 调用生成向量。
- `query_vector` 复用于 Step3 和 Step6。
- `entity_query_vectors` 复用于 Step2。
- `step2_entity_vector` 现在包含 batch embedding + entity ES 检索 + 过滤去重耗时。

因此，当前主要优化空间不再是重复 embedding，而是召回规模控制、ES 返回字段控制、以及 LLM rerank 输入规模控制。

## 搜索阶段流程

```text
Step1  实体提取
Step2  batch embedding + entity vector 检索
Step3  双通道事项召回
       - entity -> event
       - query  -> event
Step4  批量获取 event 详情和 entity_ids
Step5  单阶段固定跳数扩展
Step6  query_vector 粗排序
Step7  LLM 过滤 / 精选
Step8  event -> chunk，保留 MySQL
```

## 优化建议

### 1. 限制 Event 详情返回字段

位置：

```text
pipeline/core/storage/repositories/event_repository.py
get_events_by_ids()
```

当前 Step4/Step5 实际只需要：

```text
event_id
title
content
entity_ids
```

建议 ES 查询使用 `_source.includes`，只返回这些字段。这样可以避免拉取 `content_vector`、`title_vector` 等大字段，减少 ES 网络传输和 Python 反序列化成本。

建议目标：

```python
"_source": {
    "includes": ["event_id", "title", "content", "entity_ids"]
}
```

### 2. 控制 Step5 每跳扩展规模

位置：

```text
pipeline/modules/search/multi_vector.py
step5_expand()
```

当前逻辑：

```text
new_entity_ids -> get_event_ids_by_entity_ids(size=10000) -> new_event_ids -> get_events_by_ids()
```

如果实体很常见，单跳可能拉出大量 event，导致 Step4/Step5 明显变慢。

建议新增配置：

```text
max_expand_entities_per_hop
max_expand_events_per_hop
```

用于限制：

```text
每跳最多扩展多少新实体
每跳最多拉取多少新事项
```

### 3. Step3 entity -> event 上限配置化

位置：

```text
pipeline/core/storage/repositories/event_entity_repository.py
get_event_ids_by_entity_ids()
```

当前默认：

```python
size=10000
```

这符合“entity -> event 不受 multi_top_k 限制”的要求，但会影响后续 Step4/Step5 的规模。

建议保留默认 10000，同时在 `MultiConfig` 中增加配置项，例如：

```text
entity_event_recall_size
```

这样测试时可以灵活比较 500、1000、3000、10000 的速度和效果。

### 4. 限制 LLM rerank 候选数量

位置：

```text
pipeline/modules/search/multi_vector.py
step7_llm_rerank()
```

当前 Step6 粗排返回 `max_events` 个候选，然后 Step7 交给 LLM 精选。

如果 `max_events` 较大，LLM prompt 会变长，耗时和 token 成本都会上升。

建议新增配置：

```text
llm_candidate_top_k
```

例如：

```text
Step6 粗排 100 个
Step7 只给 LLM 前 30 个
```

### 5. 复用 search_for_rerank 的 query_vector

位置：

```text
pipeline/modules/search/multi_vector.py
search_for_rerank(query_vector=...)
```

当前接口有 `query_vector` 参数，但实际没有使用。

如果上游已经生成 query embedding，可以考虑让 `search()` 接收可选 `query_vector`，避免重复生成完整 query 的 embedding。

注意：当前搜索还需要 query entities 的 embedding，所以只传 `query_vector` 不能完全替代 batch embedding，除非接口同时传入：

```text
query_entities
entity_query_vectors
```

### 6. Native 补充段落复用 query_vector

位置：

```text
pipeline/modules/search/multi_vector.py
search_for_rerank()
search_chunks()
```

当多元搜索返回段落不足 `max_sections` 时，会调用 `search_chunks()` 补充段落。

`search_chunks()` 当前会重新生成 query embedding。可以改成接收可选 `query_vector`，复用主流程生成的向量。

## 推荐优先级

建议按下面顺序做：

```text
P0: get_events_by_ids 只返回必要字段
P1: Step5 增加每跳扩展上限
P2: Step7 增加 llm_candidate_top_k
P3: search_chunks 复用 query_vector
P4: search_for_rerank 支持外部传入 query/entity vectors
```

## 当前最值得先改的两个点

```text
1. get_events_by_ids 只返回 event_id/title/content/entity_ids
2. Step5 增加每跳扩展上限配置
```

这两个改动风险较低，且最直接影响搜索阶段耗时和扩展规模。
