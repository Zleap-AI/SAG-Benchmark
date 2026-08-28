# 统一数据集读取与 External Step 0

## 唯一原始数据源

默认目录固定为仓库根下的 `dataset/`。每个数据集必须成对存在：

```text
dataset/<name>.json
dataset/<name>_corpus.json
```

各方法只读取这一个公开数据源。方法自己的 `caches/<name>/` 是规范化运行产物，不是第二份原始数据源。

当前共享目录可发现：`musique`、`hotpotqa`、`2wikimultihopqa`、`sample`、`test_hotpotqa`、`narrativeqa`。实际可用集合由 QA/corpus 文件配对动态确定。

## Python API

```python
from pipeline.evaluation.utils.load_utils import DatasetLoader

loader = DatasetLoader("test_hotpotqa")
loader.validate_source_pair()
docs = loader.get_docs()
questions = loader.get_question_records()
```

统一 question schema：

```json
{
  "id": "...",
  "question": "...",
  "gold_answers": ["..."],
  "gold_docs": ["title\ncontent"],
  "gold_ref": "title\ncontent"
}
```

`get_docs()` 支持 corpus 的 `text` 与 `paragraph_text` 字段，并按首次出现顺序去重。`get_question_records()` 保证 QA、答案和 gold docs 同序；空问题或数量错位会明确报错。

## External Step 0

HippoRAG2：

```bash
cd external/hipporag2
uv run python reproduce/Step_0_load_dataset.py --data-name test_hotpotqa
```

HyperGraphRAG、Hyper-RAG、LightRAG 使用相同参数：

```bash
uv run python reproduce/Step_0_load_dataset.py --list
uv run python reproduce/Step_0_load_dataset.py --data-name test_hotpotqa
```

GraphRAG：

```bash
cd external/graphrag
uv run python reproduce/Step_0_prepare_dataset.py --dataset test_hotpotqa --smoke
```

只有显式传入 `--dataset-root` 时才覆盖共享目录，例如测试临时数据；正常运行无需指定。

## 统一缓存产物

每个方法在自己的项目目录生成：

```text
caches/<name>/
├── contexts/<name>_corpus_docs.json
├── questions/<name>_questions.json
└── dataset_manifest.json
```

HyperGraphRAG、Hyper-RAG、LightRAG 为兼容现有评分脚本，额外生成：

```text
questions/<name>_stage.json
questions/<name>_stage_ref.json
```

manifest 记录唯一源文件的绝对路径、SHA-256、文档/问题数与输出路径。JSON 采用同目录临时文件加原子替换写入，避免中断后留下半文件。

GraphRAG 同时保留其算法所需的：

```text
gr_<name>/input/<name>.csv
gr_<name>/questions.jsonl
```

统一读取不改变任何方法的 chunk、抽取、建图、检索或 QA 算法。
