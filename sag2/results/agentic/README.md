# Agentic SAG2 最佳结果

只保留 gpt-6.1-sol、high思考、一次SAG2扩展、最多三次工具调用这一方案。三个目录各包含1000题。

`questions.jsonl` 保存问题身份、检索原文、召回率、完整证据指标、模型用量和原始记录校验值；`summary.json` 保存汇总分数、Token及费用。`provenance.json` 保存参数、数据校验值和评测运行方式。

这些数据来自原生MCP全量评测。仓库通过通用模型API运行Agentic工具调用，不依赖原执行程序或其账号；每次运行结果独立保存。

在仓库的 sag2/ 目录运行 `uv run python scripts/verify_agentic_results.py` 可使用原始数据复算全部结果，不调用模型。
