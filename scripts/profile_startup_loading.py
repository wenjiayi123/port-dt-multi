#!/usr/bin/env python3
"""Bounded HTTP startup benchmark; issues only read-only GET requests.

Cold means a fresh Python process, not a flushed operating-system file cache.
Optional macOS sandbox-exec denies writes anywhere under --repo. Artifacts must
be outside that checkout. The same fixed GET set compares two source revisions.
For before/after comparisons, use a fresh output directory each time and keep
this order: provenance -> energy -> homepage -> provenance-summary. The output's
Matplotlib cache is shared across phases; do not prewarm it externally. A later
phase is process-cold but may reuse the font cache created by the earlier phase.
"""
from __future__ import annotations

import argparse
import asyncio
import cProfile
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pstats
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ENERGY = [
    f"/api/v3/modules/{name}/evidence"
    for name in ("yard-lighting", "hvac", "shore-bess", "bess-energy", "yard-crane")
]
HOME_CORE = [
    "/api/actuators/capabilities", "/api/assets", "/api/esg/summary",
    "/api/compliance/catalog", "/api/app_center/overview", "/api/multiport/summary",
    "/api/exec_cockpit/summary", "/api/v3/runtime/coverage",
    "/api/v3/runtime/status", "/api/v3/monitoring/evidence",
]
HOME = HOME_CORE + ["/api/system/provenance"] + ENERGY
SOURCE_ROOTS = ("app/", "scripts/", "tests/", "config/", "configs/", ".github/")
SOURCE_SUFFIXES = {".py", ".js", ".cjs", ".mjs", ".ts", ".tsx", ".jsx", ".html", ".css",
                   ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".sql", ".sh"}
ROOT_CONFIG_NAMES = {"pyproject.toml", "package.json", "package-lock.json", "Dockerfile", "Makefile",
                     "docker-compose.yml", "docker-compose.yaml", ".dockerignore", ".gitignore"}
PHASE_NAMES = ("provenance", "provenance-summary", "energy", "homepage", "provenance-profile")


def command(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def save(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_inventory(repo: Path) -> dict:
    """Include untracked implementation additions, excluding runtime/evidence trees."""
    paths = set(command(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], repo).split("\0"))
    hashes, external = {}, []
    for rel in sorted(paths):
        path = Path(rel)
        selected = (rel.startswith(SOURCE_ROOTS) and path.suffix.lower() in SOURCE_SUFFIXES
                    or rel in ROOT_CONFIG_NAMES
                    or path.parent == Path(".") and path.name.startswith("requirements") and path.suffix == ".txt")
        if not rel or not selected:
            continue
        full = repo / rel
        if not full.resolve().is_relative_to(repo):
            external.append(rel)
        elif full.is_file():
            hashes[rel] = hashlib.sha256(full.read_bytes()).hexdigest()
    return {
        "files_sha256": hashes, "external_symlinks_excluded": external,
        "inventory_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "selection": {
            "git_command": "git ls-files -z --cached --others --exclude-standard",
            "roots": list(SOURCE_ROOTS), "suffixes": sorted(SOURCE_SUFFIXES),
            "root_config_names": sorted(ROOT_CONFIG_NAMES), "root_requirements": "requirements*.txt",
            "excluded": "Ignored paths and files outside the declared source/config/script roots, including data/, evidence/, backend/, .codex_artifacts/; external symlinks are not followed",
        },
    }


def font_cache_state(output: Path) -> dict:
    directory = output / "matplotlib"
    files = {}
    if directory.is_dir():
        for path in sorted(directory.glob("fontlist*.json")):
            if path.is_file():
                stat = path.stat()
                files[path.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"directory": str(directory), "directory_exists": directory.is_dir(), "fontlist_files": files}


def checkout_state(repo: Path) -> dict:
    tracked = command(["git", "ls-files", "-z"], repo).split("\0")
    files = {}
    for rel in tracked:
        p = repo / rel
        if rel and p.is_file():
            stat = p.stat()
            files[rel] = [stat.st_size, stat.st_mtime_ns]
    manifests = [
        repo / "evidence/public_models/legacy_v3_v6_20260912/manifest.json",
        repo / "evidence/v8/shore_bess/public_models/manifest.json",
    ]
    return {
        "head": command(["git", "rev-parse", "HEAD"], repo),
        "git_status": command(["git", "status", "--porcelain"], repo),
        "tracked_file_size_mtime": files,
        "source_inventory": source_inventory(repo),
        "legacy_raw_example_exists": (repo / "data/rl/runs/rl-20260813T064228701Z/model.zip").exists(),
        "manifest_hashes": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in manifests if p.is_file()},
    }


def fetch(base: str, path: str, origin: float, timeout: float = 180) -> dict:
    started = time.perf_counter()
    row = {"path": path, "started_offset_seconds": started - origin}
    try:
        req = Request(base + path, headers={"Accept": "application/json", "Accept-Encoding": "identity"})
        try:
            response = urlopen(req, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            content = response.read()
            row.update(status=response.status, response_bytes=len(content),
                       body_sha256=hashlib.sha256(content).hexdigest(),
                       content_type=response.headers.get("Content-Type"))
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    row["top_level_keys"] = sorted(obj)
                    row["semantic_flags"] = {k: obj[k] for k in (
                        "status", "available", "admitted", "simulation_mode", "live_data_verified",
                        "dispatch_allowed", "production_authority") if k in obj}
            except (ValueError, UnicodeDecodeError):
                pass
    except Exception as error:
        row.update(status=None, error=f"{type(error).__name__}: {error}", response_bytes=0)
    row["seconds"] = time.perf_counter() - started
    return row


def sandbox_command(args: list[str], repo: Path, enabled: bool) -> list[str]:
    if not enabled:
        return args
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("--read-only-sandbox requires macOS sandbox-exec")
    policy = '(version 1)(allow default)(deny file-write* (subpath ' + json.dumps(str(repo)) + '))'
    return ["/usr/bin/sandbox-exec", "-p", policy] + args


class LocalServer:
    def __init__(self, repo: Path, output: Path, label: str, read_only_sandbox: bool):
        self.repo, self.output, self.label = repo, output, label
        self.read_only_sandbox = read_only_sandbox

    def __enter__(self):
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            self.port = reserve.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                   MPLCONFIGDIR=str(self.output / "matplotlib"))
        args = [sys.executable, "-B", "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1",
                "--port", str(self.port), "--log-level", "info"]
        args = sandbox_command(args, self.repo, self.read_only_sandbox)
        self.log_path = self.output / f"{self.label}_server.log"
        self.log = self.log_path.open("wb")
        self.origin = time.perf_counter()
        self.process = subprocess.Popen(args, cwd=self.repo, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            while time.perf_counter() - self.origin < 180:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Server exited {self.process.returncode}; see {self.log_path}")
                health = fetch(self.base, "/health/live", self.origin, timeout=.5)
                if health["status"] == 200:
                    self.startup = {"process_to_health_seconds": time.perf_counter() - self.origin,
                                    "first_health": health, "port": self.port,
                                    "write_protection": "macOS sandbox denies file-write* under measured repo" if self.read_only_sandbox else "not enabled"}
                    return self
                time.sleep(.1)
            raise TimeoutError("Server health did not become available within 180 seconds")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_args):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()


def measure(server: LocalServer, paths: list[str], concurrency: int) -> dict:
    begin = time.perf_counter()
    done = threading.Event()
    health_rows = []

    def health_probe():
        # Let the workload arrive first; one outstanding probe maximum.
        while not done.wait(.5):
            health_rows.append(fetch(server.base, "/health/live", begin))

    thread = threading.Thread(target=health_probe, daemon=True)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda path: fetch(server.base, path, begin), paths))
    finally:
        done.set()
        thread.join(timeout=190)
    return {"wall_seconds": time.perf_counter() - begin, "concurrency": concurrency,
            "requests": results, "concurrent_health": health_rows,
            "max_health_seconds": max((r["seconds"] for r in health_rows), default=None)}


def metadata(repo: Path, phase: str) -> dict:
    packages = {}
    for name in ("fastapi", "uvicorn", "numpy", "torch", "stable-baselines3", "sb3-contrib", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": "port-startup-http-performance.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo), "phase": phase, "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "hardware": command(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else platform.processor(),
        "logical_cpus": os.cpu_count(), "memory_bytes": int(command(["sysctl", "-n", "hw.memsize"])) if sys.platform == "darwin" else None,
        "packages": packages, "load_average_at_start": os.getloadavg(),
        "request_protocol": {
            "method": "GET", "accept_encoding": "identity", "timeout_seconds": 180,
            "cold": "Fresh server Python process; OS filesystem cache is not purged",
            "warm": "Same GET set on the same process immediately after its cold pass",
            "homepage": "Fixed 16-request subset observed in baseline homepage source; not all browser requests",
            "health_sampling": "one outstanding /health/live request, 0.5s between probes",
            "exclusions": "No training, formal strategy evaluation, SSE, or mutating HTTP requests",
            "energy_paths": ENERGY, "homepage_paths": HOME,
            "homepage_paths_sha256": hashlib.sha256(json.dumps(HOME, separators=(",", ":")).encode()).hexdigest(),
            "summary_boundary": "provenance-summary is a reduced loading scope, never an equivalent full-evidence speed comparison",
        },
    }


def profile_worker(repo: Path, output: Path) -> None:
    """Instrumentation is external; timed HTTP passes never install these wrappers."""
    sys.path.insert(0, str(repo))
    import app.server as server

    rows = []
    names = ["_site_twin_calibration", "_site_shadow_acceptance", "_site_execution_acceptance",
             "_port_call_collaboration", "_maritime_interoperability", "_forecast_uncertainty",
             "_business_benefit_attribution", "_end_to_end_coordination", "_production_continuity",
             "_operating_model_governance"]
    targets = [("runtime_status", server.di.strategy_runtime, "status"),
               ("rl_capabilities", server.TRAINING_MANAGER, "capabilities")]
    targets += [(name, getattr(server, name), "readiness") for name in names]
    for name, obj, method in targets:
        original = getattr(obj, method)

        def wrapped(*args, _original=original, _name=name, **kwargs):
            profiler = cProfile.Profile()
            started = time.perf_counter()
            try:
                return profiler.runcall(_original, *args, **kwargs)
            finally:
                row = {"component": _name, "seconds_with_cprofile_overhead": time.perf_counter() - started}
                rows.append(row)
                profiler.dump_stats(str(output / f"profile_{_name}.prof"))
                with (output / f"profile_{_name}.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(40)
                save(output / "provenance_component_partial.json", rows)
                print(json.dumps(row), flush=True)

        setattr(obj, method, wrapped)
    started = time.perf_counter()
    response = asyncio.run(server.system_provenance())
    save(output / "provenance_components.json", {
        "wall_seconds_with_cprofile_overhead": time.perf_counter() - started,
        "response_bytes": len(response.body), "components": rows,
        "boundary": "Direct async handler in fresh process, cProfile overhead included; separate from HTTP latency benchmark",
    })


def finalize_report(repo: Path, report_path: Path, report: dict, before: dict) -> None:
    after = checkout_state(repo)
    report["checkout_after"] = after
    report["checkout_unchanged"] = before == after
    report["load_average_at_end"] = os.getloadavg()
    report["font_cache_after_phase"] = font_cache_state(report_path.parent)
    log_path = report_path.with_name(f"{report_path.stem}_server.log")
    report["font_cache_rebuild_logged"] = (
        "Matplotlib is building the font cache" in log_path.read_text(errors="replace")
        if log_path.is_file() else None
    )
    save(report_path, report)
    if not report["checkout_unchanged"]:
        raise RuntimeError("Measured checkout state changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASE_NAMES, required=True)
    parser.add_argument("--read-only-sandbox", action="store_true", help="macOS-only write protection for immutable baseline checkout")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if args.worker:
        profile_worker(repo, output)
        return
    if output == repo or repo in output.parents:
        raise ValueError("Output must be outside the measured repository")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{args.phase}.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = metadata(repo, args.phase)
    report["font_cache_before_phase"] = font_cache_state(output)
    report["phase_sequence_in_output_before_start"] = [
        path.stem for path in sorted(
            (output / f"{name}.json" for name in PHASE_NAMES if (output / f"{name}.json").is_file()),
            key=lambda path: path.stat().st_mtime_ns,
        )
    ]
    report["font_cache_protocol"] = "Shared output/matplotlib; new output for each revision; same phase order; no external font prewarm"
    before = checkout_state(repo)
    report["checkout_before"] = before
    report["profiler_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(output / f"{args.phase}_started.json", report)
    try:
        if args.phase == "provenance-profile":
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(output / "matplotlib"))
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--repo", str(repo), "--output", str(output),
                   "--phase", args.phase, "--worker"]
            subprocess.run(sandbox_command(cmd, repo, args.read_only_sandbox), cwd=repo, env=env, check=True, timeout=240)
            report["profile"] = json.loads((output / "provenance_components.json").read_text())
            return
        with LocalServer(repo, output, args.phase, args.read_only_sandbox) as server:
            report["startup"] = server.startup
            print(json.dumps({"phase": args.phase, "startup": server.startup}), flush=True)
            paths = {"provenance": ["/api/system/provenance"],
                     "provenance-summary": ["/api/system/provenance?detail=summary"],
                     "energy": ENERGY, "homepage": HOME}[args.phase]
            concurrency = len(paths) if args.phase == "homepage" else 1
            for label in ("cold", "warm"):
                report[label] = measure(server, paths, concurrency)
                save(output / f"{args.phase}_{label}_partial.json", report)
                print(json.dumps({"phase": args.phase, "pass": label,
                                  "wall_seconds": report[label]["wall_seconds"],
                                  "max_health_seconds": report[label]["max_health_seconds"],
                                  "requests": [{k: row.get(k) for k in ("path", "status", "seconds", "response_bytes", "error")}
                                               for row in report[label]["requests"]]}), flush=True)
    finally:
        # This assertion is inside finally so the profile branch's early return
        # cannot report success after an input/source change.
        finalize_report(repo, report_path, report, before)


if __name__ == "__main__":
    main()
