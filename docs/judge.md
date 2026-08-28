# LLM Judge 运行说明

本文档对应当前版本的 `scripts/run_llm_judge.py`。脚本把外部方法（或 SAG 自身）的原生输出转换为统一 predictions 格式，并运行 generation、retrieval 以及可选的 indexing 指标。

## 子命令概览

| 入口 | 用途 | 必填参数 |
| --- | --- | --- |
| 无子命令（一键） | convert → 逐数据集 evaluate 一次完成 | `--project`、`--dataset`/`--datasets` 至少一个 |
| `convert` | 只做转换，把原生输出转为统一 predictions | `--project` |
| `evaluate` | generation 和/或 retrieval 评估，`--metrics` 按名称路由；含 `indexing` 时加跑图索引指标 | 无硬性必填，实际取决于转换产物是否已存在 |
| `indexing` | 单独向已有 Judge run 补跑图索引指标，不重跑 generation/retrieval | `--framework`、`--base-path` |

旧 `generation`、`retrieval`、`all` 子命令已删除，统一由 `evaluate` + `--metrics` 表达。

## 运行前提

请在仓库根目录运行，并确认 `.env` 提供评测模型配置：`JUDGE_LLM_API_KEY`、`JUDGE_LLM_MODEL`。两者缺失时，仅当 `JUDGE_ALLOW_FALLBACK=true` 才回退到主 LLM。评测调用固定 `temperature=0`，thinking 默认关闭（`JUDGE_LLM_ENABLE_THINKING` 可开）。ground truth 来自 `--dataset-dir`（默认仓库根 `dataset/`）。

命令模板：

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py ...
```

## 指标与路由

| 阶段 | 指标 | 是否需要 LLM |
| --- | --- | --- |
| generation | `answer_correctness`、`coverage_score`、`qa_em`、`qa_f1`、`rouge_score` | `qa_em`/`qa_f1`/`rouge_score` 为确定性指标，不需要；其余需要 |
| retrieval | `context_relevancy`、`evidence_recall` | 需要 |
| indexing | `indexing` 是 `--metrics` 中的哨兵 token，不是指标；触发图索引阶段 | 不需要 |

省略 `--metrics` 时默认评测全部 generation 指标 + 全部 retrieval 指标；数据集无 evidence 能力时（如 NarrativeQA）自动剔除 `evidence_recall`。

## 基本命令

一键完成转换、generation 与 retrieval：

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py \
  --project hypergraphrag \
  --dataset musique
```

分步：先转换，再按需要的指标评估。`evaluate` 的 `--metrics` 可混选两个阶段的任意指标并自动路由：

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py convert \
  --project hypergraphrag \
  --datasets musique

uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate \
  --project hypergraphrag \
  --dataset musique \
  --metrics context_relevancy,evidence_recall
```

核心 SAG 使用 `--project sag`。该入口固定 `sag2` 策略，自动选择 `output/<dataset>/sag2/<run_id>/` 下最新的**完整**运行：同一目录必须同时存在 `search_results.json` 和 `qa_*/qa_results.json`，因此请先运行 QA benchmark：

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py \
  --project sag \
  --dataset narrativeqa
```

在 `--metrics` 中加入 `indexing`，在同一次 Judge run 中额外运行图索引指标（需要 `--framework` 和 `--base-path`）：

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate \
  --project hypergraphrag \
  --dataset musique \
  --metrics context_relevancy,indexing \
  --framework graphml \
  --base-path /path/to/graph/output
```

续跑既有 Judge run 补算图索引指标：

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py indexing \
  --project hypergraphrag \
  --dataset musique \
  --framework graphml \
  --base-path /path/to/graph/output \
  --resume-run-id YOUR_JUDGE_RUN_ID
```

恢复失败的 Judge 样本（必须与 `--resume-run-id` 同用）：

```bash
uv run --frozen --env-file .env python scripts/run_llm_judge.py evaluate \
  --project hypergraphrag \
  --dataset musique \
  --metrics context_relevancy \
  --resume-run-id YOUR_JUDGE_RUN_ID \
  --retry-failed
```

## 命令行参数

适用范围缩写：A = 无子命令一键模式，C = `convert`，E = `evaluate`，I = `indexing`。同一参数在不同子命令默认值或约束不同时拆行呈现。

| 参数 | 类型 | 默认值 | 作用 | 适用 |
| --- | --- | --- | --- | --- |
| `--project` | str | C: argparse 必填；A: 必填（运行时校验）；E/I: 可选 | 项目名，决定使用的 adapter（graphrag / hipporag2 / hypergraphrag / hyperrag / lightrag / sag） | 全部 |
| `--dataset` | str | None | 单个数据集 | A/E/I |
| `--datasets` | str×N | None | 数据集列表；独立 convert 省略时自动发现 input-root 下的子目录 | A/C |
| `--input-root` | path | None → `external/<project>/outputs`（sag 为 `output/`） | 原生结果输入根目录 | A/C |
| `--output-root` | path | None | 一键模式专属的 artifact 根目录别名；与 `--artifact-run-root` 互斥 | A |
| `--artifact-run-root` | path | None → 仓库根 | 评测镜像根目录；evaluate 省略时从 predictions 路径推断，推断失败报错 | 全部 |
| `--data-file` | path | None | 显式指定 predictions 文件，跳过镜像解析 | A/E |
| `--judge-run-id` | str | None | 指定本次新 Judge run ID；与 `--resume-run-id` 互斥；一键模式不支持（运行时报错） | E/I |
| `--resume-run-id` | str | None | 续跑既有 Judge run；与 `--judge-run-id` 互斥；一键模式不支持（运行时报错） | E/I |
| `--max-concurrent` | int | `3` | Judge 样本级并发数 | A/E |
| `--context-top-k` | int | `5` | 每项指标读取的 contexts 条数上限（图格式 context 不截断） | A/E |
| `--num-samples` / `--top-k` | int | None（全部） | 只评测前 N 行；两个名字同一参数，命令行后出现的覆盖先出现的 | A/E |
| `--force` | flag | 关闭 | 覆盖既有 run 或既有 indexing 结果；与 `--retry-failed`、`--force-metrics` 互斥；convert 无此参数 | A（仅影响 evaluate 阶段）/E/I |
| `--dry-run` | flag | 关闭 | convert 只打印选中的 source 与输出路径，不写文件；一键模式 dry-run 到 convert 为止 | A/C |
| `--dataset-dir` | path | 仓库根 `dataset/` | 源数据集（ground truth）目录 | 全部 |
| `--source-run-id` | str | None → 最新 | 显式选择时间戳源 run（如 `<source_run_id>`） | 全部 |
| `--mode` | choice | None，取值 `naive`/`hyper`/`hyper-lite` | Hyper-RAG 源模式；源目录存在多模式时必填 | A/C |
| `--predictions-dir` | path | None | 高级：覆盖 predictions 输出目录；必须位于 `--artifact-run-root` 内 | A/C |
| `--metrics` | str | None | 逗号分隔指标列表，按名称路由到 generation/retrieval；含 `indexing` 触发图索引阶段 | A/E |
| `--only-metrics` | str | None | 旧参数别名（legacy），与 `--metrics` 互斥 | A/E |
| `--framework` | str | I: 必填；A/E: None | 图索引框架：`microsoft_graphrag` / `lightrag` / `fast_graphrag` / `hipporag2` / `graphml` | 全部 |
| `--base-path` | path | I: 必填；A/E: None | 图数据根目录；A/E 中仅当 `--metrics` 含 `indexing` 时必填 | 全部 |
| `--folder-name` | str | None | 图子目录名（`hipporag2` 框架必需） | A/E/I |
| `--retry-failed` | flag | 关闭 | 仅重跑上次失败的样本；必须与 `--resume-run-id` 同用 | E |
| `--force-metrics` | flag | 关闭 | 仅重算并替换所选指标；必须与 `--resume-run-id` 同用且需 `--metrics`/`--only-metrics` | E |
| `--judge-model` | str | None → `indexing` | indexing 结果目录名 `llmjudge/<judge-model>/` | I |

### 参数约束

1. `--resume-run-id` 与 `--judge-run-id` 互斥。
2. `--retry-failed` 必须与 `--resume-run-id` 同用；排除 `--force`、`--force-metrics`。
3. `--force-metrics` 必须与 `--resume-run-id` 同用且需 `--metrics`/`--only-metrics`；排除 `--force`；所选指标不能只有 `indexing`。
4. `--metrics` 与 `--only-metrics` 互斥。
5. 一键模式：`--resume-run-id`/`--judge-run-id` 会被运行时拒绝；`--retry-failed` 不被该模式接受。
6. `--metrics` 含 `indexing` 时必须有 `--framework` + `--base-path`，在任何 LLM 调用前 fail-fast。
7. convert 的 `--predictions-dir` 与 `--artifact-run-root` 同时显式给出时，强制要求前者位于后者内。
8. indexing 续跑时输入契约（framework / base_path / folder_name / lineage）与持久化 run 不符需 `--force`。

## 输出文件

评测产物按 `<project>/<dataset>/<source_run_id>` 三层镜像组织，predictions 与 llmjudge 同层：

```text
<artifact_root>/evaluation/<project>/<dataset>/<source_run_id>/
├── predictions/
│   ├── predictions_<dataset>.json        # 统一 predictions
│   └── conversion_manifest.json          # 转换元数据
└── llmjudge/<judge-model>/
    ├── latest.json                       # 指向最新 run 的指针
    └── <judge-run-id>/
        ├── generation_results.json       # generation 指标逐样本结果
        ├── retrieval_results.json        # retrieval 指标逐样本结果
        ├── indexing_results.json         # 图索引指标（如运行）
        ├── summary.json                  # 汇总指标
        └── run_manifest.json             # 本次 run 的配置与 lineage
```

默认 `<artifact_root>` 为仓库根。run_id 默认格式 `YYYYmmdd_HHMMSS_ffffff`。持有多层镜像出现之前的旧平铺产物（`evaluation/predictions/`、`evaluation/llmjudge/`）时，脚本仍会扫描读取。

## 注意事项

- `--top-k` 是评测行数上限（等价 `--num-samples`），不是检索 k；不要与 `--context-top-k` 混淆。
- resume 时 judge model 目录取自已持久化 run 的模型名，忽略当前 `.env` 的 `JUDGE_LLM_MODEL`——换评测模型续跑会写入旧模型目录。
- evaluate 自动选择最新**转换** run（按 convert 写入时间），可能与 adapter 选到的最新源 run 不一致；交叉核对旧源 run 时用 `--source-run-id`。
- 一键模式多数据集 + `indexing` 会逐数据集重算同一张图；这种情况改用 `indexing` 子命令补跑。
- 旧 `generation`/`retrieval`/`all` 子命令已删除；`--metrics` 支持的取值见「指标与路由」。
- 退出码：`0` 成功；`2` 无数据集；`3` JudgeError（配置/转换/索引失败）；`130` 中断；`10` 未预期异常。
