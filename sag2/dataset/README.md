# 数据集

本页命令和相对路径均以 `SAG-Benchmark/sag2/` 为当前目录。

保留 MuSiQue、2WikiMultiHopQA、HotpotQA 的问题和完整语料，共六个 JSON 文件。每组 1,000 个问题。

数据来自 HippoRAG 2 发布：`osunlp/HippoRAG_2`，固定修订 `5ec05b38deecc3318bb432c69865959c56058990`。下载入口：

```bash
uv run python scripts/run_sag2_benchmark.py --download --index-only
```

数据格式沿用上游发布。HotpotQA 支持段落的句子直接连接，2WikiMultiHopQA 用空格连接，MuSiQue 使用 `is_supporting` 标记；这些规则与论文评测一致。

EnterpriseRAG 使用项目外 `../../datasets/EnterpriseRAG-Bench` 的完整官方数据，包括 511,962 个源文档和 500 道问题；其中 470 题有召回标签。数据集：<https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench>。

第三方数据遵循各上游数据集的许可，完整语料不提交到 Git。
