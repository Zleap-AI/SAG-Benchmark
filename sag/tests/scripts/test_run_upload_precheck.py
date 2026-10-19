"""run_upload.py 的 ES 索引预检调用位置守卫。

不 import 脚本模块（会拉起重型 pipeline 依赖），用 AST 解析源码，断言
assert_indices_ready 的调用位于 set_active_embedding_dim 之后、PipelineEngine 构造之前。
防止后人删调用或挪错顺序（挪到 set_active_embedding_dim 之前会用错维度，
挪到 PipelineEngine 之后则首次写入已发生、预检形同虚设）。
"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUN_UPLOAD = PROJECT_ROOT / "scripts" / "run_upload.py"


def _line_of_first_call(source: str, func_name: str) -> int | None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == func_name:
                return node.lineno
    return None


def test_run_upload_prechecks_indices_between_dim_setup_and_engine():
    source = RUN_UPLOAD.read_text(encoding="utf-8")

    set_dim_line = _line_of_first_call(source, "set_active_embedding_dim")
    precheck_line = _line_of_first_call(source, "assert_indices_ready")
    engine_line = _line_of_first_call(source, "PipelineEngine")

    assert set_dim_line is not None, "set_active_embedding_dim 调用不存在"
    assert precheck_line is not None, "assert_indices_ready 预检调用缺失"
    assert engine_line is not None, "PipelineEngine 构造不存在"

    assert set_dim_line < precheck_line < engine_line, (
        f"预检顺序错误：set_active_embedding_dim={set_dim_line}, "
        f"assert_indices_ready={precheck_line}, PipelineEngine={engine_line}"
    )
