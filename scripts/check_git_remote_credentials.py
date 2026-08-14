#!/usr/bin/env python3
"""Fail when an HTTP(S) Git remote URL contains userinfo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


def has_http_userinfo(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.username or parsed.password)


def safe_remote_label(remote: str, url: str) -> str:
    parsed = urlsplit(url)
    return (
        f"remote={remote} scheme={parsed.scheme or 'unknown'} "
        f"host={parsed.hostname or 'unknown'}"
    )


def list_remote_urls(repo_root: Path) -> list[tuple[str, str]]:
    remotes_result = subprocess.run(
        ["git", "remote"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    results: list[tuple[str, str]] = []
    for remote in remotes_result.stdout.splitlines():
        remote = remote.strip()
        if not remote:
            continue
        urls_result = subprocess.run(
            ["git", "remote", "get-url", "--all", remote],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        results.extend((remote, url.strip()) for url in urls_result.stdout.splitlines())
    return results


def audit_remote_urls(repo_root: Path) -> list[str]:
    return [
        safe_remote_label(remote, url)
        for remote, url in list_remote_urls(repo_root)
        if has_http_userinfo(url)
    ]


def main() -> int:
    try:
        violations = audit_remote_urls(Path.cwd())
    except (OSError, subprocess.CalledProcessError):
        print("Unable to inspect Git remotes.", file=sys.stderr)
        return 2

    if violations:
        print("Credential-like HTTP userinfo found in Git remote URL(s):", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        print("Use a clean URL plus SSH or a credential helper.", file=sys.stderr)
        return 1
    print("Git remote credential check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
