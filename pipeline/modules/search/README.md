# 搜索模块（Search Module）

## 📁 目录结构

```
search/
├── __init__.py          # 导出接口
├── config.py            # 配置文件
├── searcher.py          # 统一搜索入口（SAGSearcher）
├── vector.py            # VECTOR 策略：纯向量检索
├── atomic.py            # ATOMIC 策略：原子三元组事项检索
├── multi.py             # MULTI / MULTI1 / HOPLLM 策略（按 strategy 参数切换）
├── multi_vector.py      # MULTI_ES 策略：multi.py 的 ES-First 版本
└── step5_strategies.py  # Step5 多跳扩展策略（Multi / Multi1 / HopLLM）
```

> 说明：`MULTI1` / `HOPLLM` 并非独立文件，而是 `multi.py` 中通过 `MultiConfig(strategy=...)`
> 选择的策略分支，其扩跳差异实现于 `step5_strategies.py`。


## 🔄 搜索策略

| 策略 | 说明 |
|------|------|
| `VECTOR` | 纯向量检索段落，跳过实体召回，仅支持 PARAGRAPH 返回类型 |
| `ATOMIC` | 原子三元组事项检索 + LLM 精选 |
| `MULTI` | 多元事项检索（多实体 + 固定跳数扩展 + LLM 精选） |
| `MULTI1` | 双阶段扩跳：固定1跳 eventset + 动态扩跳 eventset1，合并后一次 LLM 精选 |
| `HOPLLM` | 双阶段多跳：阶段A粗排后以粗排结果为种子进行阶段B扩跳 |
| `MULTI_ES` | `MULTI` 的 ES-First 版本（`multi_vector.py`），用 Elasticsearch 替换部分 MySQL JOIN |

## 💻 使用示例

```python
from pipeline.modules.search import SAGSearcher, SearchConfig, RerankStrategy
from pipeline.core.prompt.manager import PromptManager

searcher = SAGSearcher(prompt_manager=PromptManager())

config = SearchConfig(
    query="人工智能的最新进展",
    source_config_ids=["source_123"],
)

result = await searcher.search(config)
print(f"找到 {len(result['sections'])} 个段落")
```

## 📊 返回结果

```python
{
    "sections": [...],   # 段落列表
    "clues": [...],      # 线索列表
    "stats": {...},      # 统计信息（含耗时）
    "query": {
        "original": "...",
        "current": "...",
        "rewritten": False
    }
}
```
