"""Step 3 — 一次检索产出所有评估原料（合并了原 Step_3b）。

每题**检索一次**，同时拿到：
  - ctx：喂 LLM 的检索上下文（按 token 截断，保持 Hyper-RAG 原行为）
  - retrieved_docs：去重 chunk 列表（全量，不截断；仅作检索记录，不参与指标）

LLM 只用统一的简洁无 few-shot Thought:/Answer: 提示词生成**短答案**
pred_answer（供 EM/F1）。源项目原生长回答 result 不再生成（不维护 native result）。

一个结果记录含全字段，评估（EM/F1、Recall@k、打分、选择）都从它读，不重复检索。
不改 hyperrag 内核、不改 PROMPTS。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# 直接指向 utils 目录而非仓库根：只加载 sample_identity.py 单文件，
# 避免 pipeline 包 __init__ 引入本项目精简 venv 中不存在的重型依赖。
EVALUATION_UTILS = REPOSITORY_ROOT / "pipeline" / "evaluation" / "utils"
if str(EVALUATION_UTILS) not in sys.path:
    sys.path.insert(0, str(EVALUATION_UTILS))

import argparse
import datetime
import json
import re
import string
import time
from collections import Counter

from hyperrag import HyperRAG, QueryParam
from hyperrag import operate as opp
from hyperrag.prompt import PROMPTS
from hyperrag.utils import EmbeddingFunc, always_get_an_event_loop
from hyperrag_config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_FUNC_MAX_ASYNC,
    EMBEDDING_MAX_TOKEN_SIZE,
    EMBEDDING_MODEL_NAME,
    LLM_FUNC_MAX_ASYNC,
    embedding_func,
    llm_model_func,
)
from sample_identity import (
    index_unique_records,
    validate_question_identity,
)
from tqdm import tqdm

# 召回候选池大小（与 QueryParam.top_k 默认 60 一致）
RETRIEVE_POOL = QueryParam().top_k


# ============================================================================
#  QA prompt：与仓库统一 QA benchmark 保持一致
# ============================================================================
QA_SYSTEM_PROMPT = (
    "You are an advanced reading comprehension assistant. Read the provided Wikipedia "
    "passages and answer the question at the end. Think briefly, then output your final "
    'answer on a new line prefixed with "Answer: ".'
)


# ============================================================================
#  Token 估算（内核 openai_complete_if_cache 只 return content、丢弃 response.usage，
#  拿不到真实 usage。这里用 tiktoken 对「QA 提示词 + 答案生成输出」做估算，标注为估算值）
# ============================================================================
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import tiktoken

            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKENIZER = None
    return _TOKENIZER


def estimate_tokens(text: str) -> int:
    enc = _get_tokenizer()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text or ""))


def parse_pred(resp: str) -> str:
    """按 ``Answer:``（大小写不敏感）取最后一段作为最终短答案。

    取最后一段而非第一段：模型常在思考过程里复述 "Answer:"，最终答案总在末尾。
    """
    if not resp:
        return ""
    marker = "answer:"
    idx = resp.lower().rfind(marker)
    if idx == -1:
        return resp.strip()
    pred = resp[idx + len(marker) :].strip()
    if pred.lower() in {"<empty>", "empty", ""}:
        return ""
    return pred


# ============================================================================
#  EM / F1 内嵌评估（SQuAD/MRQA 口径：多 gold 取 max，再对题求均）
# ============================================================================
def _normalize_answer(answer: str) -> str:
    """lower -> 去标点 -> 去 a/an/the -> 合并空白。"""
    if answer is None:
        return ""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(answer)))))


def _compute_em(pred: str, gold: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0


def _compute_f1(pred: str, gold: str) -> float:
    gold_tokens = _normalize_answer(gold).split()
    pred_tokens = _normalize_answer(pred).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def compute_emf1(records: list[dict]) -> dict:
    """对成功记录计算 EM / F1（多 gold 取 max，宏平均）。返回 {ExactMatch, F1, num_examples}。"""
    total_em, total_f1 = 0.0, 0.0
    for rec in records:
        pred = rec.get("pred_answer", "") or ""
        golds = list(rec.get("gold_answers", [])) or [""]
        total_em += max(_compute_em(pred, g) for g in golds)
        total_f1 += max(_compute_f1(pred, g) for g in golds)
    n = len(records) or 1
    return {
        "ExactMatch": round(total_em / n, 4),
        "F1": round(total_f1 / n, 4),
        "num_examples": len(records),
    }


# ============================================================================
#  检索：一次同时拿到 ctx（截断）+ docs（全量去重）
# ============================================================================
async def _extract_keywords(query: str, global_config: dict):
    """复刻 operate.hyper_query 的关键词抽取（照搬 L1077-1105，含解析容错）。"""
    kw_prompt = PROMPTS["keywords_extraction"].format(query=query)
    result = await global_config["llm_model_func"](kw_prompt)
    try:
        data = json.loads(result)
        return (
            ", ".join(data.get("low_level_keywords", [])),
            ", ".join(data.get("high_level_keywords", [])),
        )
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "").replace("user", "").replace("model", "").strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            data = json.loads(result)
            return (
                ", ".join(data.get("low_level_keywords", [])),
                ", ".join(data.get("high_level_keywords", [])),
            )
        except (json.JSONDecodeError, IndexError):
            return "", ""


async def retrieve_naive(rag, question: str):
    """naive：chunks_vdb 向量召回（0 LLM）。返回 (ctx截断串, docs全量去重)。"""
    results = await rag.chunks_vdb.query(question, top_k=RETRIEVE_POOL)
    if not results:
        return ("", [])
    chunk_ids = [r["id"] for r in results]
    chunks = await rag.text_chunks.get_by_ids(chunk_ids)
    contents = [c["content"] for c in chunks if c]  # 召回顺序，id 天然唯一
    # ctx：按 token 截断（与 naive_query 一致，≤ max_token_for_text_unit）
    qp = QueryParam()
    maybe = opp.truncate_list_by_token_size(
        [{"content": c} for c in contents],
        key=lambda x: x["content"],
        max_token_size=qp.max_token_for_text_unit,
    )
    ctx = "--New Chunk--\n".join([m["content"] for m in maybe])
    # docs：完整召回池（不截断），仅作检索记录，不参与任何指标
    docs = contents
    return (ctx, docs)


async def retrieve_hyper(rag, question: str, global_config: dict):
    """hyper：关键词抽取(1 LLM) + 实体/关系召回。返回 (ctx截断串, docs全量去重)。"""
    entity_keywords, relation_keywords = await _extract_keywords(question, global_config)

    ectx = rctx = None
    if entity_keywords:
        ectx = await opp._build_entity_query_context(
            entity_keywords,
            rag.chunk_entity_relation_hypergraph,
            rag.entities_vdb,
            rag.text_chunks,
            QueryParam(),
        )
    if relation_keywords:
        rctx = await opp._build_relation_query_context(
            relation_keywords,
            rag.chunk_entity_relation_hypergraph,
            rag.entities_vdb,  # operate.hyper_query 这里传的就是 entities_vdb
            rag.relationships_vdb,
            rag.text_chunks,
            QueryParam(),
        )

    # ctx：combine_contexts 拼成（已按 token 截断，保持原行为）
    ctx = opp.combine_contexts(
        (rctx or {}).get("context", ""),
        (ectx or {}).get("context", ""),
    )

    # docs：从两个 context 的 text_units 合并去重，按首次命中顺序（不截断）
    text_units = []
    if ectx:
        text_units.extend(ectx.get("text_units", []))
    if rctx:
        text_units.extend(rctx.get("text_units", []))
    seen, docs = set(), []
    for t in text_units:
        c = t.get("content")
        if c and c not in seen:
            seen.add(c)
            docs.append(c)
    return (ctx, docs)


async def retrieve(rag, question: str, mode: str, global_config: dict):
    """一次检索，返回 (ctx, docs)。"""
    if mode == "naive":
        return await retrieve_naive(rag, question)
    # hyper / hyper-lite 都走 hyper 检索
    return await retrieve_hyper(rag, question, global_config)


# ============================================================================
#  LLM 生成（简洁无 few-shot Thought:/Answer:）
# ============================================================================


async def answer_one(query_text: str, mode: str, rag, global_config) -> dict:
    # ---- 检索一次 ----
    ctx, docs = await retrieve(rag, query_text, mode, global_config)
    if not isinstance(ctx, str) or not ctx.strip():
        ctx = "(no retrieved context)"

    # ---- 统一简洁 QA prompt -> 短答案 pred_answer（唯一路径）----
    user2 = f"{ctx}\n\nQuestion: {query_text}\nThought: "
    try:
        resp2 = await llm_model_func(user2, system_prompt=QA_SYSTEM_PROMPT, history_messages=[])
        pred = parse_pred(resp2)
    except Exception as e:
        resp2, pred = f"[path2 error] {e}", ""

    # Token 估算（QA 阶段：提示词=system+user，输出=resp2）
    prompt_est = estimate_tokens(QA_SYSTEM_PROMPT) + estimate_tokens(user2)
    completion_est = estimate_tokens(resp2)

    return {
        "pred_answer": pred,
        "retrieved_docs": docs,
        "context": ctx,
        "raw_path2": resp2,
        "qa_prompt_tokens_est": prompt_est,
        "qa_completion_tokens_est": completion_est,
    }


def resolve_run_dir(
    data_name: str,
    mode: str,
    run_id: str | None,
    resume_run_id: str | None,
    no_resume: bool,
) -> tuple[Path, str]:
    """解析本次 QA 的输出目录 outputs/{ds}/response/{run_id}/。

    图（caches/）和响应（outputs/）分开存放：同一个图可以跑多次 QA，
    每次一个 run_id 目录，靠 manifest.json 记录 mode/检索参数便于事后区分。
    默认续跑最近一个同 mode 的时间戳 run；--no_resume 或无历史时新建。
    新建用原子 mkdir 抢占，防同一秒并发进程撞出同 run_id。
    """
    response_root = Path("outputs") / data_name / "response"
    response_root.mkdir(parents=True, exist_ok=True)

    if resume_run_id:
        return response_root / resume_run_id, resume_run_id
    if run_id:
        return response_root / run_id, run_id

    # 续跑候选只认时间戳 run（自定义名 --run-id 需显式 --resume-run-id，
    # 避免字典序把字母名误判为「最新」）。
    ts_pat = re.compile(r"\d{8}_\d{6}")
    existing = sorted(
        p.name
        for p in response_root.iterdir()
        if p.is_dir()
        and ts_pat.fullmatch(p.name)
        and (p / f"{mode}_{data_name}_result.json").is_file()
    )
    if existing and not no_resume:
        return response_root / existing[-1], existing[-1]

    # 新建 run：mkdir(exist_ok=False) 抢占目录，撞了就等下一秒再试。
    for _ in range(120):
        cand_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cand = response_root / cand_id
        try:
            cand.mkdir(parents=True, exist_ok=False)
            return cand, cand_id
        except FileExistsError:
            time.sleep(0.05)
    raise RuntimeError("无法创建唯一的 run 目录（同一秒碰撞过久）")


def _is_successful_record(record: dict) -> bool:
    """一条已存在的记录是否「成功」（无需重跑）。"""
    return bool((record.get("pred_answer", "") or "").strip())


def load_questions(data_name: str, questions_file: str | None = None) -> list[dict]:
    work_dir = Path("caches") / data_name
    if questions_file:
        path = Path(questions_file)
    else:
        sel = work_dir / "questions" / f"{data_name}_questions_selected.json"
        path = sel if sel.exists() else work_dir / "questions" / f"{data_name}_questions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 "
            f"`python reproduce/Step_0_load_dataset.py --data_name {data_name}`"
        )
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    index_unique_records(questions, "id", "HyperRAG questions")
    return questions


async def run(
    questions,
    rag,
    mode,
    global_config,
    output_file,
    error_file,
    existing_records=None,
    concurrency=8,
):
    import asyncio

    existing_records = existing_records or {}
    total = len(questions)
    n_ok = n_err = n_empty_pred = n_empty_ctx = 0
    n_carried = len(existing_records)
    sem = asyncio.Semaphore(concurrency)
    all_records = list(existing_records.values())  # 供 EM/F1 统计（含续跑接管 + 新跑）
    t0 = time.time()

    async def process_one(item):
        """处理单题（带并发信号量），返回 (record_or_None, err_or_None)。"""
        async with sem:
            qid = item.get("id", "")
            query_text = item["question"]
            try:
                out = await answer_one(query_text, mode, rag, global_config)
                record = {
                    "id": qid,
                    "query": query_text,
                    "pred_answer": out["pred_answer"],
                    "gold_answers": item.get("gold_answers", []),
                    "gold_ref": item.get("gold_ref", ""),
                    "retrieved_docs": out["retrieved_docs"],
                    "context": out["context"],
                    "raw_path2": out["raw_path2"],
                    "qa_prompt_tokens_est": out.get("qa_prompt_tokens_est", 0),
                    "qa_completion_tokens_est": out.get("qa_completion_tokens_est", 0),
                }
                return (record, None)
            except Exception as e:
                return (None, {"id": qid, "query": query_text, "error": str(e)})

    # 分批并发：每批 concurrency 个，跑完立即落盘（崩溃保留已完成批次）
    # 续跑：已有成功记录按原序先写，只追加新结果。
    BATCH = concurrency
    with (
        open(output_file, "w", encoding="utf-8") as result_file,
        open(error_file, "w", encoding="utf-8") as err_file,
    ):
        result_file.write("[\n")
        first = True
        carried_list = sorted(existing_records.values(), key=lambda r: r.get("_order", 0))
        for rec in carried_list:
            if not first:
                result_file.write(",\n")
            json.dump(
                {k: v for k, v in rec.items() if k not in {"_idx", "_order"}},
                result_file,
                ensure_ascii=False,
                indent=2,
            )
            first = False
        pbar = tqdm(total=total, desc="Processing queries", unit="query")
        pbar.update(n_carried)
        for start in range(0, total, BATCH):
            batch = questions[start : start + BATCH]
            results = await asyncio.gather(*[process_one(it) for it in batch])
            for record, err in results:
                if record is not None:
                    if not first:
                        result_file.write(",\n")
                    json.dump(record, result_file, ensure_ascii=False, indent=2)
                    first = False
                    all_records.append(record)
                    n_ok += 1
                    if not record["pred_answer"]:
                        n_empty_pred += 1
                    if not record.get("context", "").strip():
                        n_empty_ctx += 1
                else:
                    json.dump(err, err_file, ensure_ascii=False, indent=2)
                    err_file.write("\n")
                    n_err += 1
                pbar.update(1)
            result_file.flush()  # 每批落盘，崩溃不丢
        pbar.close()
        result_file.write("\n]")

    qa_seconds = time.time() - t0
    # 汇总统计
    print(
        f"[stats] 总数={total} 续跑接管={n_carried} 成功={n_ok} 失败(跳过)={n_err}"
        f" 其中 pred_answer 为空={n_empty_pred} 检索上下文为空={n_empty_ctx}"
    )
    if n_err:
        print(f"[stats] 失败详情见 {error_file}")
    # 内嵌 EM/F1：对全部成功记录（续跑接管 + 本次新跑）求值
    return {
        "qa_seconds": round(qa_seconds, 3),
        "metrics": compute_emf1(all_records),
        "total": total,
        "n_ok": n_ok,
        "n_err": n_err,
        "n_carried": n_carried,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyper-RAG 一次检索 + 两路出答案 + 导出检索 chunk")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, default="naive", choices=["naive", "hyper", "hyper-lite"]
    )
    parser.add_argument("--questions_file", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=8, help="同时处理的问题数（并发）")
    parser.add_argument("--no_resume", action="store_true", help="不续跑，新建运行目录")
    parser.add_argument("--run-id", default=None, help="指定输出运行目录名（默认时间戳）")
    parser.add_argument(
        "--resume-run-id", default=None, help="从 outputs/<ds>/response/<run_id> 继续运行"
    )
    args = parser.parse_args()
    if args.run_id and args.resume_run_id:
        parser.error("--run-id 与 --resume-run-id 不能同时使用")

    data_name = args.data_name
    mode = args.mode
    work_dir = Path("caches") / data_name

    questions = load_questions(data_name, args.questions_file)

    rag = HyperRAG(
        working_dir=str(work_dir),
        embedding_func_max_async=EMBEDDING_FUNC_MAX_ASYNC,
        llm_model_max_async=LLM_FUNC_MAX_ASYNC,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
            tokenize_url=EMBEDDING_BASE_URL,
            tokenize_model=EMBEDDING_MODEL_NAME,
            tokenize_api_key=EMBEDDING_API_KEY,
        ),
    )
    global_config = {"llm_model_func": llm_model_func}

    # 最终结果统一输出到 outputs/，中间产物留在 caches/
    out_dir, run_id = resolve_run_dir(
        data_name, mode, args.run_id, args.resume_run_id, args.no_resume
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode}_{data_name}_result.json"
    err_path = out_dir / f"{mode}_{data_name}_errors.json"
    # ---- 断点续跑：加载已有成功结果，只重跑失败的 ----
    existing_records: dict[str, dict] = {}  # qid -> record
    if not args.no_resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            previous = json.load(f)
        previous_by_id = index_unique_records(previous, "id", "HyperRAG existing results")
        questions_by_id = index_unique_records(questions, "id", "HyperRAG questions")
        question_order = {sample_id: i for i, sample_id in enumerate(questions_by_id)}
        for sample_id, rec in previous_by_id.items():
            question = questions_by_id.get(sample_id)
            if question is None:
                raise ValueError(f"HyperRAG existing result has unknown sample ID {sample_id!r}")
            rec["_order"] = question_order[sample_id]
            validate_question_identity(
                question["question"], rec.get("query"), sample_id, "HyperRAG resume"
            )
            if _is_successful_record(rec):
                existing_records[sample_id] = rec
        print(
            f"[resume] 已有 {len(previous)} 条结果，其中 {len(existing_records)} 条成功，"
            f"需重跑 {len(previous) - len(existing_records)} 条"
        )

    pending = [q for q in questions if q["id"] not in existing_records]

    if not pending:
        print("[resume] 全部已完成，无需重跑。")
        return

    loop = always_get_an_event_loop()
    summary = loop.run_until_complete(
        run(
            pending,
            rag,
            mode,
            global_config,
            out_path,
            err_path,
            existing_records,
            concurrency=args.concurrency,
        )
    )

    # Token 估算汇总（QA 阶段）
    with open(out_path, encoding="utf-8") as f:
        final_records = json.load(f)
    total_prompt_est = sum(r.get("qa_prompt_tokens_est", 0) for r in final_records)
    total_completion_est = sum(r.get("qa_completion_tokens_est", 0) for r in final_records)
    llm_usage_est = {
        "estimated": True,
        "note": "内核 openai_complete_if_cache 只 return content 丢弃 usage，此为 tiktoken 估算",
        "qa_prompt_tokens_est": total_prompt_est,
        "qa_completion_tokens_est": total_completion_est,
        "qa_total_tokens_est": total_prompt_est + total_completion_est,
        "qa_calls": summary["n_ok"],
    }

    # manifest 记录「这次响应用的是哪个图 + 什么检索参数」——
    # 图和响应分目录后，这是唯一能把两者关联起来的凭据。
    # hyperrag 内核把图存为 hypergraph_chunk_entity_relation.hgdb（storage.py）。
    graph_file = work_dir / "hypergraph_chunk_entity_relation.hgdb"
    manifest = {
        "dataset": data_name,
        "run_id": run_id,
        "mode": mode,
        # 真正的检索参数：召回池大小取 Hyper-RAG 内核 QueryParam 默认值。
        "top_k": RETRIEVE_POOL,
        "max_token_for_text_unit": QueryParam().max_token_for_text_unit,
        "concurrency": args.concurrency,
        "index_dir": str(work_dir.resolve()),
        "index_built_at": (
            datetime.datetime.fromtimestamp(graph_file.stat().st_mtime).isoformat()
            if graph_file.exists()
            else None
        ),
        "num_questions": len(questions),
        "result_file": str(out_path),
        "error_file": str(err_path),
        "metrics": summary["metrics"],
        "runtime_metrics": {"qa_seconds": summary["qa_seconds"]},
        "llm_usage_est": llm_usage_est,
        "finished_at": datetime.datetime.now().isoformat(),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── 结果摘要（检索指标由共享 Judge 计算）────────────────────────────
    m = summary["metrics"]
    print("==> [摘要] 指标")
    print(f"  [QA]     ExactMatch={m['ExactMatch']}  F1={m['F1']}  (n={m['num_examples']})")
    print(f"  [Time]   qa_seconds={summary['qa_seconds']}")
    print(
        f"  [Tokens] est qa_total={llm_usage_est['qa_total_tokens_est']} "
        f"(prompt={total_prompt_est}, completion={total_completion_est}, 估算)"
    )
    print(f"[response] {data_name} ({mode}): 完成 -> {out_path}")


if __name__ == "__main__":
    main()
