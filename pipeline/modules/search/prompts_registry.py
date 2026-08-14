"""
把检索提示词注册进 MLflow Prompt Registry —— 合并为 2 组 chat 格式 prompt。

2 组（name = multi_search.<group>）：
    multi_search.ner          system + oneshot(user/assistant) + Question 模板
    multi_search.rerank       system + 3组 few-shot(user/assistant) + 模板

用法：
    # 默认：从 MLflow 现有分条 prompt 的最新版拉内容拼（保留 UI 上的改动）
    python -m pipeline.modules.search.prompts_registry --from mlflow --uri http://localhost:5050

    # 应急/离线：从代码常量 _FALLBACKS 拼
    python -m pipeline.modules.search.prompts_registry --from const --uri http://localhost:5050

行为：
- 每组按 GROUP_SPECS（见 prompts.py）的有序 (role, 源短名) 序列拼成 chat 消息列表；
- register_prompt(name, template=list[dict])（同名再注册自动 +1 版本，版本不可变）；
- 不对 @latest set alias（latest 是保留选择器，运行时用 @latest 拉最新版）；
- 幂等：重复运行新增版本。

注意：本脚本不参与运行时；运行时加载见 prompts.py 的 PromptProvider。
"""

from __future__ import annotations

import argparse
import os
import sys

from pipeline.modules.search.prompts import (
    _FALLBACKS,
    GROUP_NAMES,
    GROUP_SPECS,
    PROMPT_NAMES,
    build_chat_from_parts,
    full_prompt_name,
)


def _delete_legacy(mlflow) -> int:
    """删除旧的分条 prompt（multi_search.<short>）。

    对每条：先删所有别名（若有），再逐个删版本，最后删 prompt 条目本身。
    仅删 PROMPT_NAMES 里的分条名，绝不碰 GROUP_NAMES 的 2 组 chat prompt。
    返回失败条数。
    """
    from mlflow import MlflowClient

    # 双保险：确认待删名与保留的 2 组名无交集
    group_full = {full_prompt_name(g) for g in GROUP_NAMES}

    client = MlflowClient()
    failures = 0
    for short in PROMPT_NAMES:
        name = full_prompt_name(short)
        if name in group_full:
            print(f"  ⚠ 跳过（属于保留的 chat 组）: {name}")
            continue
        try:
            # 直接删整条 prompt（含所有版本/别名）
            client.delete_prompt(name)
            print(f"  ✓ 已删除 {name}")
        except Exception as exc:  # noqa: BLE001
            # 某些版本/别名可能需要先单独清理；退化为逐版本删后再删条目
            try:
                _force_delete_prompt(client, name)
                print(f"  ✓ 已删除 {name}（逐版本）")
            except Exception as exc2:  # noqa: BLE001
                failures += 1
                print(f"  ✗ 删除失败 {name}: {exc2 or exc}", file=sys.stderr)
    return failures


def _force_delete_prompt(client, name: str) -> None:
    """兜底：先删别名+所有版本，再删 prompt 条目。"""
    # 删常见别名（忽略不存在）
    for alias in ("production", "staging"):
        try:
            client.delete_prompt_alias(name, alias)
        except Exception:  # noqa: BLE001
            pass
    # 逐版本删（版本号从 1 起，遇不存在即停）
    v = 1
    while True:
        try:
            client.delete_prompt_version(name, v)
            v += 1
        except Exception:  # noqa: BLE001
            break
    client.delete_prompt(name)



def _collect_parts_from_mlflow(mlflow, alias: str) -> dict[str, str]:
    """从 MLflow 现有分条 prompt（multi_search.<short>@alias）拉取每条最新内容。

    返回 {短名: content 文本}。缺失的短名会缺项，拼装时抛 KeyError 暴露问题。
    """
    parts: dict[str, str] = {}
    # 收集 2 组用到的全部源短名（去重）
    shorts = {short for spec in GROUP_SPECS.values() for _role, short in spec}
    for short in sorted(shorts):
        uri = f"prompts:/{full_prompt_name(short)}@{alias}"
        p = mlflow.genai.load_prompt(uri)
        # 分条 prompt 的 .template 是 str
        parts[short] = p.template
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="注册 2 组 chat 格式检索提示词到 MLflow Prompt Registry"
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("MLFLOW_PROMPT_URI", "http://localhost:5050"),
        help="MLflow tracking URI（可通过 MLFLOW_PROMPT_URI 环境变量设置）",
    )
    parser.add_argument(
        "--from",
        dest="source",
        choices=["mlflow", "const"],
        default="mlflow",
        help="内容来源：mlflow=从现有分条 prompt 最新版拉取拼装（默认，保留 UI 改动）；"
        "const=从代码常量 _FALLBACKS 拼（应急/离线）",
    )
    parser.add_argument(
        "--source-alias",
        default="latest",
        help="--from mlflow 时，拉取源分条 prompt 用的选择器（默认: latest）",
    )
    parser.add_argument(
        "--commit-message",
        default="merge into chat-format group",
        help="本次注册的 commit message",
    )
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="删除旧的分条 prompt（multi_search.<short>）。"
        "可单独使用（只删不注册）；与注册并用时先注册 2 组再删旧的。",
    )
    parser.add_argument(
        "--only-delete-legacy",
        action="store_true",
        help="只删旧分条 prompt，跳过注册 2 组。",
    )
    args = parser.parse_args()

    import mlflow

    mlflow.set_tracking_uri(args.uri)

    # 只删不注册
    if args.only_delete_legacy:
        print(f"[删除] tracking_uri={args.uri}, 删除旧 {len(PROMPT_NAMES)} 条分条 prompt...")
        fails = _delete_legacy(mlflow)
        if fails:
            print(f"\n[删除] 完成，但 {fails} 条失败。", file=sys.stderr)
            return 1
        print("\n[删除] 旧分条 prompt 全部删除完成。")
        return 0

    print(
        f"[注册] tracking_uri={args.uri}, from={args.source}, "
        f"共 {len(GROUP_NAMES)} 组: {GROUP_NAMES}"
    )

    # 准备各短名内容
    if args.source == "mlflow":
        print(f"[注册] 从 MLflow 现有分条 prompt 拉取内容（@{args.source_alias}）...")
        try:
            parts = _collect_parts_from_mlflow(mlflow, args.source_alias)
        except Exception as exc:  # noqa: BLE001
            print(f"[注册] 拉取现有分条 prompt 失败: {exc}", file=sys.stderr)
            return 1
    else:
        print("[注册] 从代码常量 _FALLBACKS 拼装内容...")
        parts = dict(_FALLBACKS)

    failures: list[str] = []
    for group in GROUP_NAMES:
        name = full_prompt_name(group)
        try:
            chat_template = build_chat_from_parts(GROUP_SPECS[group], parts)
            pv = mlflow.genai.register_prompt(
                name=name,
                template=chat_template,
                commit_message=args.commit_message,
                tags={"module": "search.multi", "group": group, "format": "chat"},
            )
            print(f"  ✓ {name:<28} v{pv.version} （{len(chat_template)} 条消息）")
        except Exception as exc:  # noqa: BLE001 — 逐组汇报，不中断整体
            failures.append(name)
            print(f"  ✗ {name:<28} 失败: {exc}", file=sys.stderr)

    if failures:
        print(f"\n[注册] 完成，但 {len(failures)} 组失败: {failures}", file=sys.stderr)
        return 1
    print(f"\n[注册] 全部 {len(GROUP_NAMES)} 组成功。运行时用 @latest 自动拉最新版。")

    # 注册成功后，可选删除旧分条
    if args.delete_legacy:
        print(f"\n[删除] 删除旧 {len(PROMPT_NAMES)} 条分条 prompt...")
        fails = _delete_legacy(mlflow)
        if fails:
            print(f"[删除] 完成，但 {fails} 条失败。", file=sys.stderr)
            return 1
        print("[删除] 旧分条 prompt 全部删除完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
