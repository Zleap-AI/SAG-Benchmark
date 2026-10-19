# 本地 Rerank：离线运行 SAG2

本页命令在仓库的 `sag2/` 目录执行。[SAG2中文首页](../README-CN.md) · [项目首页](../../README-CN.md)

本地模式保留 SAG2 的 SQLite FTS5、BM25、词项扩展与最终多样性排序，将每轮 Jev 评分替换为本机 Rerank 推理。原始语料不添加批注；完成依赖、权重与语料准备后，查询过程无网络调用，不需要模型 API key。

## 模型与运行环境

| 模型 | 后端 | 用途 | 权重许可 |
|---|---|---|---|
| Qwen/Qwen3-Reranker-0.6B | Transformers，CPU / CUDA / MPS | 通用本地部署；本次 MuSiQue Rerank 对照中 R@10 最高 | Apache-2.0 |
| jinaai/jina-reranker-v3.5 | Transformers，CPU / CUDA / MPS | 多语言联合排序；已有三个数据集的完整对照 | CC BY-NC 4.0 |
| jinaai/jina-reranker-v3.5-mlx | MLX，Apple Silicon | Mac 本地推理 | CC BY-NC 4.0 |

许可与模型接口见官方说明：[Qwen](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)、[Jina](https://huggingface.co/jinaai/jina-reranker-v3.5)、[Jina MLX](https://huggingface.co/jinaai/jina-reranker-v3.5-mlx)。Jina 的商业使用需另行获得许可。权重由使用者下载，不包含在本仓库中。

## 安装与下载

在有网络的准备阶段安装依赖和下载权重。默认 SAG2 安装不会引入本地模型依赖。

```bash
# Qwen / Jina 的 Transformers 后端
uv sync --frozen --group local

# 固定模型修订，下载后可以断网运行
uv run --group local hf download Qwen/Qwen3-Reranker-0.6B \
  --revision e61197ed45024b0ed8a2d74b80b4d909f1255473 \
  --local-dir models/qwen3-reranker-0.6b

# 可选：Jina Transformers 权重（包含官方模型代码）
uv run --group local hf download jinaai/jina-reranker-v3.5 \
  --revision e8a93f33f0b22108f8c2364f8484ce3422552fbc \
  --local-dir models/jina-reranker
```

Apple Silicon 可使用官方 MLX 版本：

```bash
uv sync --frozen --group local-mlx
uv run --group local-mlx hf download jinaai/jina-reranker-v3.5-mlx \
  --revision 3dd4ac901ccdcac85abe3815df0a0aaaf44e4a21 \
  --local-dir models/jina-reranker-mlx
```

准备固定评测语料：

```bash
uv run python scripts/run_sag2_benchmark.py --download --index-only
```

Jina 使用模型目录中的官方代码；应使用上面的固定修订。离线入口只接受存在的本地权重目录，禁用 Hugging Face 在线访问和遥测，不会自动下载模型或切换云端服务。

## 完全本地评测

```bash
# Qwen：MuSiQue 前10题，完整语料、两次扩展
uv run --offline --group local python scripts/run_sag2_benchmark.py \
  --mode local --reranker qwen --model-path models/qwen3-reranker-0.6b \
  --dataset musique --limit 10 --expansions 2 --concurrency 20

# 三个数据集全量；不请求 Jev 或其他云端模型
uv run --offline --group local python scripts/run_sag2_benchmark.py \
  --mode local --reranker qwen --model-path models/qwen3-reranker-0.6b \
  --datasets all --expansions 2 --concurrency 20

# Mac：Jina MLX
uv run --offline --group local-mlx python scripts/run_sag2_benchmark.py \
  --mode local --reranker jina --backend mlx --model-path models/jina-reranker-mlx \
  --dataset musique --limit 10 --expansions 2

# 单轮消融：原始问题 + 标准相关性提示词，不扩展、不做最终DPP
uv run --offline --group local python scripts/run_sag2_benchmark.py \
  --mode local --reranker qwen --model-path models/qwen3-reranker-0.6b \
  --dataset musique --expansions 0 --rerank-protocol standard
```

`--expansions 0` 默认仍使用 SAG2 的证据提示词和最终 DPP；要复现标准单轮设置，还需 `--rerank-protocol standard`。单轮与多轮对照同时改变提示词、查询上下文和 DPP，差值不能全部归因于扩展次数。

本地问题并发上限默认20；单个模型实例的推理依次执行，避免多个长上下文同时占用显存。Qwen 的 `--batch-size` 控制同时评分的段落数量，默认4；`--device` 可指定 `cpu`、`cuda` 或 `mps`。MLX 使用 GPU。长上下文计算量和显存需求随候选数量增加。

## 检索自己的语料

准备 JSON 文件，每项包含 `title` 和 `text`：

```json
[
  {"title": "项目说明", "text": "项目由甲公司承建。"},
  {"title": "公司资料", "text": "甲公司的总部位于杭州。"}
]
```

运行本地查询：

```bash
uv run --offline --group local python scripts/search_local.py \
  --corpus my-corpus.json --query "承建该项目的公司总部在哪里？" \
  --reranker qwen --model-path models/qwen3-reranker-0.6b --expansions 2
```

程序返回前十个去重原文段落、顺序和相关性分数，适用于接入本地问答程序。该入口执行证据检索，不生成最终答案。

## 输入、评分与恢复

多轮模式每次传入原问题、已有证据及完整候选原文，使用现有 SAG2 证据评分提示词。Qwen 使用 yes/no 概率；Jina 的原始余弦分数按 `(score + 1) / 2` 转换为0–1，供扩展权重和最终 DPP 使用。内部兼容字段 `jev_score` 在本地模式中保存 Rerank 分数，不代表请求了 Jev。

本地代码不截断原文或共享上下文。输入超过模型上下文时直接报错；可减少 `--candidate-k` 或调整分段后重新评测。Jina MLX 的共享查询上限已提高到模型上下文容量，避免官方默认查询长度截断已有证据；官方分组算法保持不变。Transformers 后端使用完整单组输入。MLX与Transformers的模型格式和提示词存在差异。

默认评测结果位于 `sag2/output/sag2/<运行时间>/<数据集>/`；`--output-dir` 指定的是运行根目录。评测输出的 `config.json` 记录模型文件 SHA256、后端、设备与评分设置。`requests/<题号>/` 保存本地评分输入和原始响应；`details.jsonl` 保存逐题检索结果，`reranker_usage` 记录推理次数和实际输入Token。`model_api_requests` 和 `jev_requests` 均为0。这里的API费用为0，硬件与电力成本未统计。

恢复时使用原参数及 `--resume output/sag2/<运行目录>`，已完成题目及相同模型、相同输入的本地响应直接复用。失败会报错，不会自动切换模型或服务商。

## 消融结果

[最终消融数据](../results/rerank/summary.json)保留7组完成的评测及可取得的逐题记录；没有加入100题预检、失败重试日志或早期分批实现。模型 `qwen3-rerank` 是302提供的独立型号，参数规模未确认，不能与 Qwen 0.6B 混称。

**表中全量检索分数来自US上的302接口。** 本地入口使用相同检索方法及评分规则。

同批 MuSiQue 对照中的 Jev R@5 / R@10 为80.59% / 86.61%，与主表采用的独立论文结果分别保留。每项均为同一固定1,000题；完成率指前十段包含全部标准证据的题目比例。

离线复核：

```bash
uv run python scripts/verify_rerank_results.py
uv run pytest -q
```

历史 Jina MLX 速度测试在 M4 Pro / 64GB、BF16 上，预热后三次测量中位数为：512个输入Token 0.082秒、4,096个Token 1.054秒、32,768个Token 41.616秒，加载约2.02秒。输入为合成英文候选；完整检索还包含多轮评分与词法扩展。
