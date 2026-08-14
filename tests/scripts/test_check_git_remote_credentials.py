from scripts.check_git_remote_credentials import has_http_userinfo, safe_remote_label


def test_rejects_http_userinfo() -> None:
    assert has_http_userinfo("http://user:fixture-secret@example.test/repo.git")
    assert has_http_userinfo("https://fixture-secret@example.test/repo.git")


def test_accepts_clean_http_and_ssh_urls() -> None:
    assert not has_http_userinfo("https://example.test/repo.git")
    assert not has_http_userinfo("git@example.test:group/repo.git")


def test_safe_label_never_contains_credentials() -> None:
    label = safe_remote_label(
        "origin",
        "https://user:fixture-secret@example.test/repo.git",
    )
    assert label == "remote=origin scheme=https host=example.test"
    assert "user" not in label
    assert "fixture-secret" not in label
