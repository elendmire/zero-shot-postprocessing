"""Persist result DataFrames with CLAUDE.md's non-negotiable provenance rule 3:
every results/ file carries git SHA, model version, and config hash. Every
later script in this project must call write_result() instead of calling
df.to_parquet() directly.

Final-review fix round (item 2): this project's mandatory workflow is
sync-then-run (rsync the working tree to `altay`, run there, no remote git
commits until later) -- so `git_sha` alone names the nearest PRIOR commit, not
necessarily the exact code that produced a given result file (proven stale at
least once: a shipped result's git_sha named a commit that predates the code
that produced it). Two fields added to close that gap:
  - `git_dirty`: True if `git status --porcelain` at write time was non-empty
    (i.e. the working tree differed from `git_sha`'s commit in ANY tracked
    file, anywhere in the repo -- a coarse, repo-wide signal).
  - `source_tree_sha256`: a sha256 digest over every file under src/,
    scripts/, configs/ (sorted by path, content-addressed) -- this identifies
    the EXACT source tree that ran, independent of whether it was ever
    committed. Two runs with the same source_tree_sha256 ran identical code,
    regardless of what git_sha/git_dirty say."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

# Directories whose contents define "the exact code that ran" for this
# project's results -- see source_tree_sha256's docstring above.
_SOURCE_TREE_DIRS = ("src", "scripts", "configs")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _git_dirty() -> bool:
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode()
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
