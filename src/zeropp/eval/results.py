"""Write a result table as parquet with a JSON sidecar.

The sidecar records the model version, the configuration and its hash, the git commit, whether the working
tree was modified, a hash of the source files (src/, scripts/, configs/) and the time of writing.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

_SOURCE_TREE_DIRS = ("src", "scripts", "configs")


def _git_sha() -> str:
    """HEAD commit hash (read from .git if the git binary is not available)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        head = REPO_ROOT / ".git" / "HEAD"
        if not head.exists():
            return "unknown"
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            ref_path = REPO_ROOT / ".git" / ref[5:]
            if ref_path.exists():
                return ref_path.read_text().strip()
            packed = REPO_ROOT / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(ref[5:]):
                        return line.split()[0]
            return "unknown"
        return ref


def _git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes, None if git is not available."""
    try:
        porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return len(porcelain.strip()) > 0


def _source_tree_sha256() -> str:
    hasher = hashlib.sha256()
    files = []
    for dirname in _SOURCE_TREE_DIRS:
        root = REPO_ROOT / dirname
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix in (".pyc", ".pyo"):
                continue
            if path.name == ".DS_Store":
                continue
            files.append(path)
    for path in sorted(files, key=lambda p: p.relative_to(REPO_ROOT).as_posix()):
        relpath = path.relative_to(REPO_ROOT).as_posix()
        hasher.update(relpath.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def write_result(
    df: pd.DataFrame, *, name: str, model_version: str, config: dict, out_dir: str = "results"
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path / f"{name}.parquet", index=False)

    meta = {
        "model_version": model_version,
        "config": config,
        "config_hash": _config_hash(config),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "source_tree_sha256": _source_tree_sha256(),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
