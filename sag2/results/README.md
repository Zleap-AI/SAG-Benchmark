# 最终论文结果

最佳 Agentic SAG2 的三个数据集全量结果、Token和费用另见 [Agentic SAG2 结果](agentic/README.md)。下表保留原 SAG2 论文结果。

本地Rerank的模型与流程消融见[Rerank结果](rerank/README.md)。

表中数值为百分比，从最终论文的数据快照复制，并由逐题记录复核。

| 指标 | MuSiQue | 2WikiMultiHopQA | HotpotQA | EnterpriseRAG |
|---|---:|---:|---:|---:|
| 评测题数 | 1,000 | 1,000 | 1,000 | 470 |
| R@1 | 34.30 | 43.55 | 48.30 | 63.05 |
| R@2 | 62.18 | 85.50 | 93.60 | 71.68 |
| R@5 | 80.68 | 97.35 | 98.45 | 77.74 |
| R@10 | 86.62 | 97.43 | 99.00 | 81.10 |

EnterpriseRAG 按源文档 ID 计算召回，其余三组按支持段落计算。EnterpriseRAG 另外 30 题没有召回标签，不进入分母。

## 保留的对照

MuSiQue 的八组全量记录分别是：完整 SAG2、匹配判断预算的 BM25+JEV、移除上下文、仅最终 JEV、RM3、RM3+最终 JEV、GAR-BM25、GAR-BM25+最终 JEV。H0/H1/H2 分阶段结果来自同一完整 SAG2 运行。

`paper/main-results.json` 保留论文主表，包括引用的原 SAG 及其他基线分数。`paper/bm25-reference.json` 为论文采用的固定 BM25 参考值。它们与本次对照运行中的 BM25 结果分别记录，不混用。

## 数据来源

`paper/` 保存最终论文使用的数据快照。逐题结果、原始记录 SHA256、最终汇总和复核报告一并保留，评测字段保持原值。

`provenance.json` 使用仓库内的相对路径列出结果来源。`sha256` 校验公开文件，`source_sha256` 标识导出前的原始汇总文件；逐题记录中的 `source_record_sha256` 标识原始结果记录。
