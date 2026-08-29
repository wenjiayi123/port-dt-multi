#!/usr/bin/env python3
"""Fail closed when publishable files contain personal-material or secret traces."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 12_000_000
PRIVATE_FILENAME_RE = re.compile(
    r"(?:录制台词|演示台词|演示脚本|demo[_ -]?script|recording[_ -]?script|"
    r"个人简历|个人自传|求职材料|面试台词|采访稿|答辩稿|身份证|护照)",
    re.IGNORECASE,
)
ROOT_MEDIA_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".png.import", ".jpg.import", ".jpeg.import", ".webp.import", ".gif.import",
)
LOCAL_ACCOUNT_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:/Users/[^/\s]+|/home/[^/\s]+|[A-Z]:\\Users\\[^\\\s]+)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@(?:[A-Z][A-Z0-9-]*\.)+[A-Z]{2,}\b"
)
LABELED_PHONE_RE = re.compile(
    r"(?i)(?:\b(?:phone|mobile|tel)\b|电话|手机|手机号)[^\n\r0-9]{0,24}(?:\+?\d[\d ()-]{7,}\d)"
)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
TOKEN_RE = re.compile(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")


def _publishable_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / raw.decode("utf-8", errors="surrogateescape")
        for raw in completed.stdout.split(b"\0")
        if raw
    ]


def _text(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
        return None
    raw = path.read_bytes()
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    findings: list[tuple[str, int, str]] = []
    scanner = Path(__file__).resolve()
    for path in _publishable_files():
        relative = path.relative_to(ROOT).as_posix()
        lowered = relative.casefold()
        if PRIVATE_FILENAME_RE.search(relative):
            findings.append((relative, 0, "personal-material filename"))
        if "/" not in relative and lowered.endswith(ROOT_MEDIA_SUFFIXES):
            findings.append((relative, 0, "root-level personal-media risk"))
        content = _text(path)
        if content is None or path.resolve() == scanner:
            continue
        is_guard = (
            "release_check" in lowered
            or "/release/check." in lowered
            or relative == "backend/app/tests/test_dashboard_api.py"
        )
        checks = (
            ("local account path", LOCAL_ACCOUNT_RE),
            ("email address", EMAIL_RE),
            ("labeled phone number", LABELED_PHONE_RE),
            ("private key", PRIVATE_KEY_RE),
            ("access token", TOKEN_RE),
        )
        for line_number, line in enumerate(content.splitlines(), start=1):
            for label, pattern in checks:
                if is_guard and label in {"local account path", "private key", "access token"}:
                    continue
                if pattern.search(line):
                    findings.append((relative, line_number, label))

    if findings:
        print(f"PUBLIC_PRIVACY_SCAN:FAIL:{len(findings)}")
        for relative, line_number, label in findings:
            location = f"{relative}:{line_number}" if line_number else relative
            print(f"- {location}: {label}")
        return 1
    print("PUBLIC_PRIVACY_SCAN:PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
