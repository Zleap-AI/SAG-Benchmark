"""Build and load the local FTS5 ranking extension once per source/platform."""

from __future__ import annotations

import hashlib
import os
import platform
import shlex
import sqlite3
import subprocess
import sys
import sysconfig
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def extension_path() -> Path:
    source = Path(__file__).with_name("native")
    digest = hashlib.sha256()
    for name in ("sag_rank.c", "sqlite3.h", "sqlite3ext.h"):
        digest.update((source / name).read_bytes())
    digest.update(f"{sys.platform}-{platform.machine()}".encode())
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "sag2"
        / "native"
        / digest.hexdigest()
    )
    target = cache / ("sag_rank.dll" if sys.platform == "win32" else "sag_rank.so")
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / target.name
            if sys.platform == "win32":
                command = [
                    os.environ.get("CC", "cl"),
                    "/LD",
                    "/O2",
                    f"/I{source}",
                    str(source / "sag_rank.c"),
                    "/link",
                    f"/OUT:{output}",
                ]
            else:
                compiler = shlex.split(
                    os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
                )
                flags = (
                    ["-bundle", "-undefined", "dynamic_lookup"]
                    if sys.platform == "darwin"
                    else ["-shared", "-fPIC"]
                )
                command = [
                    *compiler,
                    "-O3",
                    *flags,
                    f"-I{source}",
                    str(source / "sag_rank.c"),
                    "-o",
                    str(output),
                    "-lm",
                ]
            subprocess.run(command, check=True, cwd=temporary)
            try:
                os.link(output, target)
            except FileExistsError:
                # Another process published the same immutable build first.
                pass
    return target


def load_rank_extension(connection: sqlite3.Connection) -> None:
    path = extension_path()
    connection.enable_load_extension(True)
    try:
        connection.load_extension(str(path))
    finally:
        connection.enable_load_extension(False)
