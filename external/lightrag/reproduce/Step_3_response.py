"""Step 3 — LightRAG hybrid QA：检索一次 + SAG2 风格 Thought:/Answer: 出答案。

每题调用一次 LLM（SAG2 风格），context 是纯文本直接送 LLM。
LightRAG context 中的 Sources 块会解析为 retrieved_docs。

结果记录 schema：
  {id, query, pred_answer, gold_answers, gold_ref, retrieved_docs, context, raw_path2}

支持断点续跑：如果输出文件已存在，自动加载已有结果，只重新处理之前失败的题。
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
import asyncio
import datetime
import json
import re
import string
import time
from collections import Counter

from lightrag_config import build_rag_kwargs, llm_model_func
from sample_identity import (
    index_unique_records,
    validate_question_identity,
)
from tqdm import tqdm

# lightrag_config 必须先于 lightrag 内核 import：内核 import 时 load_dotenv(override=True)
# 会覆盖命令行注入的 env（--env-file），先读 config 才能保住注入值。
# E402 由仓库根 ruff 配置全局忽略；isort: skip 保持该行在 config 之后
from lightrag import LightRAG, QueryParam  # isort: skip


# ============================================================================
#  QA prompt：与仓库统一 QA benchmark 保持一致
# ============================================================================
QA_SYSTEM_PROMPT = (
    "You are an advanced reading comprehension assistant. Read the provided Wikipedia "
    "passages and answer the question at the end. Think briefly, then output your final "
    'answer on a new line prefixed with "Answer: ".'
)


def resolve_run_dir(
    data_name: str,
    run_id: str | None,
    resume_run_id: str | None,
    no_resume: bool,
) -> tuple[Path, str]:
    """解析本次 QA 的输出目录 outputs/{ds}/response/{run_id}/。

    图（caches/）和响应（outputs/）分开存放：同一个图可以跑多次 QA，
    每次一个 run_id 目录，靠 manifest.json 记录检索参数便于事后区分。
    默认续跑最近一个时间戳 run；--no_resume 或无历史时新建时间戳 run。
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
        and (p / f"hybrid_{data_name}_result.json").is_file()
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
        # 兜底：约 4 字符/token
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


def _is_successful_record(record: dict) -> bool:
    pred = record.get("pred_answer", "")
    return bool((pred or "").strip())


def load_questions(data_name: str, questions_file: str | None = None) -> list[dict]:
    work_dir = Path("caches") / data_name
    if questions_file:
        path = Path(questions_file)
    else:
        path = work_dir / "questions" / f"{data_name}_questions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 "
            f"`python reproduce/Step_0_load_dataset.py --data_name {data_name}`"
        )
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    index_unique_records(questions, "id", "LightRAG questions")
    return questions


def extract_sources_docs(ctx: str) -> list[str]:
    """Parse LightRAG Sources into retrieved document chunks."""
    if not isinstance(ctx, str) or "-----Sources-----" not in ctx:
        return []
    sources = ctx.split("-----Sources-----", 1)[1]
    if "-----" in sources:
        sources = sources.split("-----", 1)[0]
    docs: list[str] = []
    for line in sources.splitlines():
        line = line.strip().strip(chr(96))
        if not line:
            continue
        if line.lower().startswith("id,") and "content" in line.lower():
            continue
        content = line.split(",", 1)[1] if "," in line else line
        content = content.strip().strip('"')
        if content:
            docs.append(content)
    return docs


async def answer_one(query_text: str, rag: LightRAG) -> dict:
    try:
        ctx = await rag.aquery(
            query_text,
            QueryParam(
                mode="hybrid",
                top_k=5,
                max_token_for_text_unit=4000,
                max_token_for_global_context=4000,
                max_token_for_local_context=4000,
                only_need_context=True,
            ),
        )
    except Exception as e:
        ctx = f"(retrieval error) {e}"
    if not isinstance(ctx, str) or not ctx.strip():
        ctx = "(no retrieved context)"
    docs = extract_sources_docs(ctx)

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


async def run(
    questions,
    rag,
    output_file,
    errors_file,
    existing_records,
    concurrency=8,
    all_questions=None,
):
    total = len(questions)
    n_ok = n_err = n_empty_pred = n_empty_ctx = 0
    n_carried = len(existing_records)
    sem = asyncio.Semaphore(concurrency)
    all_records = list(existing_records.values())  # 供 EM/F1 统计（含续跑接管 + 新跑）
    t0 = time.time()

    async def process_one(item):
        async with sem:
            qid = item.get("id", "")
            query_text = item["question"]
            try:
                out = await answer_one(query_text, rag)
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

    BATCH = concurrency
    with (
        open(output_file, "w", encoding="utf-8") as result_file,
        open(errors_file, "w", encoding="utf-8") as err_file,
    ):
        result_file.write("[\n")
        first = True

        question_order = {item["id"]: i for i, item in enumerate(all_questions or questions)}
        carried_list = sorted(existing_records.values(), key=lambda r: question_order[r["id"]])
        for rec in carried_list:
            if not first:
                result_file.write(",\n")
            json.dump(rec, result_file, ensure_ascii=False, indent=2)
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
            result_file.flush()
        pbar.close()
        result_file.write("\n]")

    qa_seconds = time.time() - t0
    print(
        f"[stats] 总数={total} 续跑接管={n_carried} 新成功={n_ok} 新失败={n_err}"
        f" 其中 pred_answer 为空={n_empty_pred} 检索上下文为空={n_empty_ctx}"
    )
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
    parser = argparse.ArgumentParser(
        description="LightRAG hybrid QA：检索一次 + 两路出答案 + 导出检索 chunk（支持断点续跑）"
    )
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--questions_file", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--run-id", default=None, help="指定输出运行目录名（默认时间戳）")
    parser.add_argument(
        "--resume-run-id", default=None, help="从 outputs/<ds>/response/<run_id> 继续运行"
    )
    args = parser.parse_args()
    if args.run_id and args.resume_run_id:
        parser.error("--run-id 与 --resume-run-id 不能同时使用")

    data_name = args.data_name
    work_dir = Path("caches") / data_name

    all_questions = load_questions(data_name, args.questions_file)

    # 最终结果统一输出到 outputs/，中间产物留在 caches/
    out_dir, run_id = resolve_run_dir(data_name, args.run_id, args.resume_run_id, args.no_resume)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hybrid_{data_name}_result.json"
    err_path = out_dir / f"hybrid_{data_name}_errors.json"

    existing_records: dict[str, dict] = {}
    if not args.no_resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            previous = json.load(f)
        previous_by_id = index_unique_records(previous, "id", "LightRAG existing results")
        questions_by_id = index_unique_records(all_questions, "id", "LightRAG questions")
        for sample_id, rec in previous_by_id.items():
            question = questions_by_id.get(sample_id)
            if question is None:
                raise ValueError(f"LightRAG existing result has unknown sample ID {sample_id!r}")
            validate_question_identity(
                question["question"], rec.get("query"), sample_id, "LightRAG resume"
            )
            if _is_successful_record(rec):
                existing_records[sample_id] = rec
        print(
            f"[resume] 已有 {len(previous)} 条结果，其中 {len(existing_records)} 条成功，"
            f"需重跑 {len(previous) - len(existing_records)} 条"
        )

    pending = [q for q in all_questions if q["id"] not in existing_records]

    if not pending:
        print("[resume] 全部已完成，无需重跑。")
        return

    rag = LightRAG(working_dir=str(work_dir), **build_rag_kwargs())

    loop = asyncio.get_event_loop()
    summary = loop.run_until_complete(
        run(
            pending,
            rag,
            out_path,
            err_path,
            existing_records,
            concurrency=args.concurrency,
            all_questions=all_questions,
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
    graph_file = work_dir / "graph_chunk_entity_relation.graphml"
    manifest = {
        "dataset": data_name,
        "run_id": run_id,
        "mode": "hybrid",
        # 真正的检索参数（QueryParam），与 answer_one 里的显式取值保持一致。
        "top_k": 5,
        "max_token_for_text_unit": 4000,
        "max_token_for_global_context": 4000,
        "max_token_for_local_context": 4000,
        "concurrency": args.concurrency,
        "index_dir": str(work_dir.resolve()),
        "index_built_at": (
            datetime.datetime.fromtimestamp(graph_file.stat().st_mtime).isoformat()
            if graph_file.exists()
            else None
        ),
        "num_questions": len(all_questions),
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
    print(f"[response] {data_name} (hybrid): 完成 -> {out_path}")


if __name__ == "__main__":
    main()
