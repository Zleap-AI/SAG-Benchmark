"""
GraphRAG reproduce/ 公共工具模块。

所有 Step 脚本共享的：根目录定位、.env 加载、VIRTUAL_ENV 劫持防护、
带时间戳打印、subprocess 流式执行、文件锁（防重入）、graphrag 包覆盖同步。
"""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv as _load_dotenv

# ── 项目根目录 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"
DATASETS = [
    "musique",
    "2wikimultihopqa",
    "hotpotqa",
    "test_hotpotqa",
    "sample",
    "narrativeqa",
]

# ── 目录约定（对齐 hyperrag：caches=中间产物 / outputs=最终结果）──────
# caches/<ds>/ 既是 graphrag 引擎工作区（settings.yaml + input/output/cache），
# 也承载 Step_0 的产物汇聚层（contexts/ + questions/）。
CACHE_ROOT = PROJECT_ROOT / "caches"
# outputs/<ds>/ 只放最终结果：检索 response/ + QA qa/。
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# reproduce/ 下受版本控制的 settings.yaml 模板，Step_0 原样复制到 caches/<ds>/
SETTINGS_TEMPLATE = Path(__file__).resolve().parent / "settings.yaml"


def workspace_dir(ds: str, ensure: bool = False) -> Path:
    """caches/<ds>/ —— graphrag 引擎工作区兼 Step_0 产物汇聚层。

    ensure=True 时自动建 contexts/ 和 questions/ 子目录。
    """
    d = CACHE_ROOT / ds
    if ensure:
        for sub in ("contexts", "questions"):
            (d / sub).mkdir(parents=True, exist_ok=True)
    return d


# ── 带时间戳打印（无缓冲，nohup 下也立刻可见；对齐 hyperrag 的 _log）──
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── .env 加载（读取仓库根 .env，不覆盖已在 shell 中 export 的变量）──
def load_env(env_path: str | Path | None = None) -> None:
    """加载环境变量，默认使用仓库根 .env。"""
    resolved = Path(env_path) if env_path else DEFAULT_ENV_FILE
    # 若通过 SAG_ENV_FILE 指定，则相对于 REPOSITORY_ROOT 解析
    if env_path is None:
        sag_env = os.environ.get("SAG_ENV_FILE")
        if sag_env:
            candidate = (REPOSITORY_ROOT / sag_env).resolve()
            if candidate.is_file():
                resolved = candidate
    log(f"[env] 加载 {resolved}")
    if resolved.exists():
        _load_dotenv(str(resolved), override=False)
    else:
        log(f"[env] ⚠ {resolved} 不存在，请先创建")


# ── VIRTUAL_ENV 劫持防护 ─────────────────────────────────────
def pin_venv() -> None:
    """把本项目的 .venv 钉到 VIRTUAL_ENV + PATH 头部，防其他项目劫持."""
    venv = PROJECT_ROOT / ".venv"
    if venv.is_dir():
        os.environ["VIRTUAL_ENV"] = str(venv)
        bin_dir = str(venv / "bin")
        path = os.environ.get("PATH", "")
        if bin_dir not in path:
            os.environ["PATH"] = f"{bin_dir}:{path}"


# ── subprocess 流式执行（逐行 tee 到 stdout）───────────────────
def run_streamed(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict | None = None,
    timeout: int | None = None,
) -> int:
    """运行命令，逐行 tee stdout+stderr，返回 returncode."""
    log(f"[cmd] {' '.join(str(x) for x in cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=env or os.environ,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:  # type: ignore[union-attr]
        print(line.rstrip("\n"), flush=True)
    proc.wait(timeout=timeout)
    return proc.returncode


# ── 文件锁（防重入，等价原 run_gr.sh 的 run.lock + trap 清理）────
def _pid_alive(pid: int) -> bool:
    """进程是否存活（signal 0 只做权限/存在性检查，不真的发信号）。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # 存在但不属于当前用户
        return True
    return True


@contextmanager
def file_lock(lock_path: Path, who: str = ""):
    """排他锁，已被活进程持有则 SystemExit；退出时本进程创建的锁才清理。

    持锁进程被 kill -9 时 finally 不会执行，锁会残留。这里按锁里记录的 PID
    判断存活：进程已不在就视为陈旧锁，接管并继续。
    """
    lock_fd = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                holder = int(lock_path.read_text().strip())
            except (OSError, ValueError):
                holder = -1
            if holder > 0 and _pid_alive(holder):
                log(f"[lock] 另一个进程正在跑（{lock_path} 存在，PID={holder}），退出")
                sys.exit(1)
            log(f"[lock] 发现陈旧锁（PID={holder} 已不存在），接管 {lock_path}")
            lock_path.unlink(missing_ok=True)
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        log(f"[lock] 获取锁 {lock_path}  ({who})")
        yield
    finally:
        if lock_fd is not None:  # 只有本进程成功创建了锁才清理
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)
            log(f"[lock] 释放锁 {lock_path}")


# ── 工具：读/写 JSON ──────────────────────────────────────────
def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── graphrag pip 包覆盖同步（cost 埋点等本地修改）────────────────
def ensure_overrides() -> None:
    """把本地 src/ 覆盖文件同步进 .venv site-packages（幂等，uv sync 后自动补齐）。

    [PATCH LAYER] 本函数是本项目唯一允许改动 site-packages 的地方。上游
    graphrag==2.0.0 不提供 cost 计数与 enable_thinking 关闭的扩展点，因此只能
    以文件替换方式注入。若上游将来提供等效扩展点，整个 src/ 可删除。

    用 .patch_manifest 记录已同步的相对路径，每次运行时回滚孤儿文件（src/ 里
    已删除、但 site-packages 里旧副本仍残留的），避免「只追加不回滚」悄悄影响运行。

    回滚只**删除**孤儿文件，不恢复被覆盖的上游原件；若上游同名文件仍需存在
    （即从 src/ 移除某 patch），须重跑 `uv sync` 重建。
    """
    import shutil

    venv = PROJECT_ROOT / ".venv"
    if not venv.is_dir():
        return
    # 兼容 lib/ 与 lib64/（RHEL 系 python 布局，lib64 常是 lib 的符号链接，resolve 去重）
    sites = sorted(
        {p.resolve() for d in ("lib", "lib64") for p in (venv / d).glob("python*/site-packages")}
    )
    if not sites:
        return
    site = sites[0]
    local = PROJECT_ROOT / "src"  # 覆盖补丁层（对齐 hipporag2 的 src/ 布局）
    manifest = venv / ".graphrag_patch_manifest"

    src_rels = (
        {src.relative_to(local) for src in sorted(local.rglob("*.py"))} if local.is_dir() else set()
    )

    # 1) 读上次 manifest，回滚孤儿文件（已同步过、但 src/ 里已不存在的）
    prev_rels: set[str] = set()
    if manifest.exists():
        try:
            prev_rels = set(manifest.read_text(encoding="utf-8").splitlines())
        except OSError:
            prev_rels = set()
    for rel_str in sorted(prev_rels - {str(r) for r in src_rels}):
        orphan = site / "graphrag" / rel_str
        if orphan.is_file():
            orphan.unlink()
            log(f"[override] 回滚孤儿 {rel_str}")

    # 2) 复制当前 src/ 文件（现有逻辑不变）
    if local.is_dir():
        for src in sorted(local.rglob("*.py")):
            rel = src.relative_to(local)
            dst = site / "graphrag" / rel
            if not dst.exists() or src.read_bytes() != dst.read_bytes():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                log(f"[override] 同步 {rel}")

    # 3) 写新 manifest
    manifest.write_text("\n".join(sorted(str(r) for r in src_rels)), encoding="utf-8")
