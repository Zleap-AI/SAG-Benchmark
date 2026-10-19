# Rerank 最终消融

保留7组1,000题评测的最终汇总和可取得的逐题结果。全量评分使用302接口；本地运行方法见[部署说明](../../docs/local-rerank.md)。

`summary.json` 包含原始比例指标、模型、语料、流程、来源校验值和复核状态。`records` 为 `null` 的 Qwen 0.6B 组只保留已核对的汇总；其余6组可通过 `scripts/verify_rerank_results.py` 离线重算。
