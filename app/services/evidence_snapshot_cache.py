"""Process-local evidence reuse bound to the files the verified build reads.

This is not a TTL or an alternate admission path. A changed, replaced, added,
or deleted input causes the original builder (including its SHA gates) to run.
The lock coalesces concurrent cold requests; a changing build is never cached.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import stat as stat_mode
import threading
from pathlib import Path
from typing import Any, Callable, Iterable


def _within_repo(path: Path, repo: Path) -> Path:
    if not path.resolve().is_relative_to(repo.resolve()):
        raise RuntimeError(f"Evidence dependency escapes repository: {path}")
    return path


def file_snapshot(paths: Iterable[Path], *, root: Path | None = None) -> tuple:
    records: dict[str, tuple] = {}

    def visit(path: Path, ancestors: frozenset[tuple[int, int]] = frozenset()) -> None:
        name = str(path.absolute())
        if name in records:
            return
        try:
            stat = path.lstat()
        except FileNotFoundError:
            records[name] = ("missing",)
            return
        if stat_mode.S_ISLNK(stat.st_mode):
            if root is not None:
                _within_repo(path, root)
            records[name + ":symlink"] = (stat.st_dev, stat.st_ino, stat.st_size,
                                           stat.st_mtime_ns, stat.st_ctime_ns)
            try:
                stat = path.stat()
            except FileNotFoundError:
                records[name] = ("missing-target",)
                return
        if stat_mode.S_ISDIR(stat.st_mode):
            identity = (stat.st_dev, stat.st_ino)
            records[name] = ("directory", *identity)
            if identity in ancestors:
                raise RuntimeError(f"Evidence dependency directory cycle: {path}")
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.name != "__pycache__" and not entry.name.endswith((".pyc", ".pyo")):
                        visit(Path(entry.path), ancestors | {identity})
            return
        # ctime catches same-size rewrites even when mtime is restored. Inode
        # and device also distinguish atomic replacement and symlink targets.
        records[name] = (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size,
                         stat.st_mtime_ns, stat.st_ctime_ns)

    for path in paths:
        if root is not None:
            _within_repo(Path(path), root)
        visit(Path(path))
    return tuple(sorted(records.items()))


class FileBoundEvidenceCache:
    """One immutable snapshot per service; failures and mixed builds aren't reused."""

    def __init__(self, root: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._root = root
        self._signature: tuple | None = None
        self._payload: Any = None

    def get(self, dependencies: Callable[[], Iterable[Path]], build: Callable[[], Any]) -> Any:
        with self._lock:
            signature = file_snapshot(dependencies(), root=self._root)
            if signature != self._signature:
                # Discard prior admission before doing work which might fail.
                self._signature, self._payload = None, None
                payload = build()
                if signature != file_snapshot(dependencies(), root=self._root):
                    raise RuntimeError("Evidence inputs changed during verification; retry with a stable snapshot")
                self._payload = payload
                self._signature = signature
            return copy.deepcopy(self._payload)


def _path_references(value: Any, repo: Path, reports: list[Path] | None = None) -> Iterable[Path]:
    """Discover explicit paths in the current config/pointer/report, without trusting them."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and item and (key == "path" or key.endswith(("_path", "_dir"))):
                if "://" not in item:
                    path = _within_repo(repo / item, repo)
                    yield path
                    if reports is not None and (key == "report_path" or key.endswith("_report_path")):
                        reports.append(path)
            elif "/" in key and (item is None or (isinstance(item, str) and len(item) == 64
                                                    and all(char in "0123456789abcdef" for char in item.lower()))):
                # V8 manifests use {repository_path: sha256}, including absent
                # historical pointers. Their keys are dependencies as well.
                yield _within_repo(repo / key, repo)
            elif isinstance(item, (dict, list)):
                yield from _path_references(item, repo, reports)
    elif isinstance(value, list):
        for item in value:
            yield from _path_references(item, repo, reports)


def energy_evidence_dependencies(service: Any) -> list[Path]:
    """Cover raw inputs, selected artifacts, diagnostics, mappings, and source files.

    Directory enumeration is metadata-only and detects new/deleted dependencies.
    Explicit report/config references also cover runs outside the usual folders.
    JSON parsing here only discovers files; the builder still verifies contents.
    """
    repo, module = service.repo_root, service.v3_evidence.name
    paths = [service.root, service.v3_evidence, repo / "config",
             repo / "data/rl/datasets", repo / "evidence/v3/value_improvement_v32.json",
             repo / f"app/services/{module}_evidence.py", repo / "app/services/evidence_snapshot_cache.py",
             repo / "app/services/training_process_evidence.py",
             repo / "app/services/value_improvement.py"]
    config_path = repo / "config" / f"{module}_v3.json"
    pointer = service.v3_evidence / "latest.json"
    discover = [config_path, pointer]
    if module in {"shore_bess", "bess_energy"}:
        paths.extend([repo / "evidence/public_models/legacy_v3_v6_20260912",
                      repo / "evidence/v7/public_models", repo / "app/services/rl_training",
                      repo / "app/services/rl_model/shore_bess/v8_public_artifacts.py"])
    if module == "shore_bess":
        paths.extend([repo / "evidence/v8/shore_bess", repo / "scripts",
                      repo / "app/services/shore_bess_v8_evidence.py"])
        discover.extend(repo / "evidence/v8/shore_bess" / name for name in
                        ("latest.json", "offline_champion.json", "diagnostics_latest.json"))
    seen: set[Path] = set()
    while discover:
        path = _within_repo(discover.pop(0), repo)
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
        try:
            # Check containment before reading JSON, including config/pointer
            # symlinks. Invalid evidence is rejected by the ordinary builder.
            value = json.loads(path.read_text(encoding="utf-8"))
            reports: list[Path] = []
            paths.extend(_path_references(value, repo, reports))
            for report in reports:
                discover.append(report)
                # Do not recursively scan an arbitrary report's entire parent.
                # These siblings are the persisted process/replay dependencies.
                paths.extend(report.parent.glob("seed_*"))
                paths.extend(report.parent / name for name in
                             ("source", "config.json", "manifest.json", "checkpoint_reward_replay.json"))
        except (OSError, ValueError, TypeError):
            pass
    return paths


def csv_record_count(path: Path) -> int:
    """Count records, not physical lines; match DictReader's blank-line handling."""
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        return sum(1 for row in rows if row)


def verify_report_dataset(repo: Path, report: dict[str, Any]) -> None:
    """Bind refreshed runtime inference to the data cited by the sealed report."""
    dataset = report.get("dataset") or {}
    files = list(dataset.get("files") or [])
    if dataset.get("dataset_id") and dataset.get("sha256"):
        files.append({"path": f"data/rl/datasets/{dataset['dataset_id']}.csv", "sha256": dataset["sha256"]})
    for item in files:
        path = _within_repo(repo / str(item.get("path") or ""), repo)
        expected = item.get("sha256")
        if not expected or not path.is_file():
            raise ValueError(f"Formal dataset artifact missing or unanchored: {item.get('path')}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Formal dataset SHA-256 mismatch: {item.get('path')}")
