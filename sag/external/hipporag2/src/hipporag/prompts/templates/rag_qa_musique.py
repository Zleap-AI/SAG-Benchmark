# 对齐 sag-benchmark run_qa_benchmark.py 的 QA prompt：无 one-shot 示例，简短作答，
# 减少 prompt/completion token 开销，便于与 SAG 的 QA 结果直接比较。

rag_qa_system = (
    "You are an advanced reading comprehension assistant. Read the provided Wikipedia "
    "passages and answer the question at the end. Think briefly, then output your final "
    'answer on a new line prefixed with "Answer: ".'
)

prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": "${prompt_user}"}
]
