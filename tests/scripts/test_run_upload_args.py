"""run_upload.py 的 --dataset choices 参数测试。

不 import 脚本模块（run_upload.main() 会拉起重型 pipeline 依赖），
直接用 AST 解析源码中的 add_argument choices 列表。
"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUN_UPLOAD = PROJECT_ROOT / "scripts" / "run_upload.py"

NARRATIVEQA_DATASET = "narrativeqa"


def _find_dataset_choices(script_path: Path) -> list[str] | None:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if func_name != "add_argument":
            continue
        positional = [a.value if isinstance(a, ast.Constant) else None for a in node.args]
        if "--dataset" not in positional:
            continue
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, ast.List):
                return [elem.value for elem in kw.value.elts]
    return None


def test_run_upload_dataset_choices_include_narrativeqa():
    choices = _find_dataset_choices(RUN_UPLOAD)

    assert choices is not None
    assert NARRATIVEQA_DATASET in choices
