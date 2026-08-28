#!/usr/bin/env python
"""
GraphRAG v2.0.0 检索器 —— 只做 local search 检索，产出 SAG 格式的检索结果 JSON。

设计：GraphRAG 只当检索器，QA/评估复用 sag-benchmark 的 run_qa_benchmark.py，
保证 QA prompt、LLM、EM/F1、gold 与 SAG / HippoRAG 逐字节相同，口径零漂移。

每道题执行 local_search 的 `engine.context_builder.build_context(...)`（只取 context，
不生成答案），将 `res.context_chunks` 整段图上下文作为单条 retrieved_docs 写出。

（注：思考模式通过 patch 层 src/language_model/providers/fnllm/utils.py 关闭，
由环境变量 GRAPHRAG_DISABLE_THINKING=1 控制，本脚本不需单独设置。）

用法:
  export GRAPHRAG_COST_FILE=<project_root>/caches/<dataset>/cost.json
  export GRAPHRAG_COST_PHASE=query
  python -m graphrag_benchmark.retrieval \
      --root ./caches/musique \
      --questions ./caches/musique/questions.jsonl \
      --out ./outputs/musique/response/graphrag_musique_result.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from pipeline.evaluation.utils.sample_identity import (
    index_unique_records,
    validate_identity_coverage,
    validate_question_identity,
)

import pandas as pd

from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.load_config import load_config
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import get_embedding_store, load_search_prompt

from graphrag_benchmark.config import apply_env


def _fmt_dur(seconds: float) -> str:
    """把秒格式化成 08m12s / 1h23m45s（与 reproduce/progress.py 口径一致）。

    本文件位于 graphrag_benchmark 包内，不 import reproduce/progress.py（跨包且
    reproduce 无 __init__.py），这里本地复制一份 4 行实现。
    """
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def _read_parquets(output_dir: Path) -> dict:
    def rd(name: str, optional: bool = False):
        p = output_dir / f"{name}.parquet"
        if not p.exists():
            if optional:
                return None
            raise FileNotFoundError(f"缺少 index 产物: {p}（先跑 graphrag index）")
        return pd.read_parquet(p)

    return {
        "entities": rd("entities"),
        "communities": rd("communities"),
        "community_reports": rd("community_reports"),
        "text_units": rd("text_units"),
        "relationships": rd("relationships"),
        "covariates": rd("covariates", optional=True),
    }


def _build_search_engine(config, dfs, community_level: int):
    """复刻官方 local_search_streaming 的组装，但只用于取 context（不生成答案）。"""
    vector_store_args = {}
    for index, store in config.vector_store.items():
        vector_store_args[index] = store.model_dump()
    description_embedding_store = get_embedding_store(
        config_args=vector_store_args,
        embedding_name=entity_description_embedding,
    )
    entities_ = read_indexer_entities(dfs["entities"], dfs["communities"], community_level)
    covariates_ = (
        read_indexer_covariates(dfs["covariates"]) if dfs["covariates"] is not None else []
    )
    prompt = load_search_prompt(config.root_dir, config.local_search.prompt)

    # 修正 text_units.document_ids 类型（csv 输入时为 int，需转 str）
    tu_df = dfs["text_units"].copy()
    if "document_ids" in tu_df.columns:

        def fix_ids(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)) or hasattr(x, "__iter__"):
                return [str(i) for i in x]
            return [str(x)]

        tu_df["document_ids"] = tu_df["document_ids"].apply(fix_ids)

    return get_local_search_engine(
        config=config,
        reports=read_indexer_reports(dfs["community_reports"], dfs["communities"], community_level),
        text_units=read_indexer_text_units(tu_df),
        entities=entities_,
        relationships=read_indexer_relationships(dfs["relationships"]),
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,
        response_type="Multiple Paragraphs",  # 不生成答案，此值无实际作用
        system_prompt=prompt,
    )


def _resolve_output_dir(config, root: Path) -> Path:
    """config.output.base_dir 可能是相对路径，相对 root 解析。"""
    output_dir = Path(config.output.base_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return output_dir


class Retriever:
    """GraphRAG local search retriever.

    Builds the search engine once from the index at ``root`` and answers
    context-retrieval queries against it.  Only ``build_context`` is called —
    no answer generation LLM is invoked.
    """

    def __init__(self, root: Path, community_level: int = 2) -> None:
        # 回填 settings.yaml 的 ${VAR}，必须在 load_config 之前（幂等）。
        apply_env()
        config = load_config(root.resolve(), None, {})
        output_dir = _resolve_output_dir(config, root)
        dfs = _read_parquets(output_dir)
        self._engine = _build_search_engine(config, dfs, community_level)

    def retrieve(self, query: str) -> str:
        """Return the full graph context for ``query`` as a single string."""
        result = self._engine.context_builder.build_context(
            query=query,
            **self._engine.context_builder_params,
        )
        cc = result.context_chunks
        return cc if isinstance(cc, str) else "\n\n".join(str(x) for x in cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--questions", required=True, help="jsonl，每行含 question / (可选)id")
    ap.add_argument("--out", required=True, help="输出 SAG 格式 search_results.json")
    ap.add_argument("--community-level", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    retriever = Retriever(root, args.community_level)

    with open(args.questions, encoding="utf-8") as stream:
        questions = [json.loads(line) for line in stream if line.strip()]
    question_by_id = index_unique_records(questions, "id", "GraphRAG questions")
    if args.limit:
        questions = questions[: args.limit]
        question_by_id = index_unique_records(questions, "id", "GraphRAG questions")
    print(f"[info] 共 {len(questions)} 题")
    if not questions:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f"[done] 0 条 → {args.out}")
        return

    results = []
    empty = 0

    t_retrieve_start = time.perf_counter()
    for i, q in enumerate(questions):
        question = q.get("question") or q.get("query")
        full_ctx = ""
        try:
            full_ctx = retriever.retrieve(question)
        except Exception as e:  # 单题失败不中断
            print(f"  [{i + 1}] ERR: {e!r}")
        if not full_ctx.strip():
            empty += 1
        sample_id = questions[i]["id"]
        validate_question_identity(
            questions[i].get("question"), question, sample_id, "GraphRAG retrieval"
        )
        results.append(
            {
                "question_index": i,
                "dataset_sample_id": sample_id,
                "question": question,
                "retrieved_docs": [full_ctx] if full_ctx.strip() else [],
            }
        )
        if (i + 1) % 20 == 0 or (i + 1) == len(questions):
            done = i + 1
            elapsed = time.perf_counter() - t_retrieve_start
            eta = elapsed * (len(questions) - done) / done if done else None
            eta_s = f"  预计剩余 {_fmt_dur(eta)}" if eta is not None else ""
            print(
                f"  检索进度 {done}/{len(questions)}  (空检索 {empty})  "
                f"已用 {_fmt_dur(elapsed)}{eta_s}",
                flush=True,
            )

    index_unique_records(results, "dataset_sample_id", "GraphRAG retrieval results")
    validate_identity_coverage(
        set(question_by_id),
        {row["dataset_sample_id"] for row in results},
        "GraphRAG retrieval results",
    )
    t_retrieve_elapsed = time.perf_counter() - t_retrieve_start
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(results)} 条 → {args.out}  (空检索 {empty} 条)")
    print(f"[retrieve done]  {len(questions)} 题全部检索完，用时 {_fmt_dur(t_retrieve_elapsed)}")
    print(
        f"[retrieve time] {t_retrieve_elapsed:.1f}s ({t_retrieve_elapsed / len(questions):.1f}s/题)"
    )
    try:
        from graphrag import cost_meter

        cost_meter.record_elapsed(t_retrieve_elapsed)
    except Exception:
        pass


if __name__ == "__main__":
    main()
