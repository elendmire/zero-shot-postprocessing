#!/usr/bin/env python3
"""Reference integrity checks for the ZeroPP manuscript.

Checks, in order of severity:
  1. every refs.bib entry carries a DOI or an arXiv id           (ERROR)
  2. no duplicate DOIs across entries                            (ERROR)
  3. no near-duplicate titles (same work under two keys)         (WARN)
  4. every \\cite{key} in the manuscript exists in refs.bib      (ERROR)
  5. every refs.bib entry is cited at least once                 (WARN)
  6. every DOI in selected_dois.tsv made it into refs.bib        (WARN)

Exit code 1 if any ERROR fires, so it can gate CI.

Usage:
    python scripts/check_references.py
    python scripts/check_references.py --manuscript paper/main.tex
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

BIB_PATH = Path("docs/references/refs.bib")
TSV_PATH = Path("docs/references/selected_dois.tsv")
DEFAULT_MANUSCRIPT_GLOBS = ("paper/**/*.tex", "paper/**/*.md")

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
CITE_RE = re.compile(r"\\cite(?:alp|alt|author|year)?[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}")
TITLE_SIM_THRESHOLD = 0.92


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split text on `sep`, but only where brace depth is 0, so a comma inside
    a field's {...} value (e.g. an author list) never splits that field into
    two pieces. Depth-aware, not full BibTeX-grammar-aware -- the generator
    only ever emits simple, single-level-braced field values."""
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _parse_fields(body: str) -> dict:
    """Parse a BibTeX entry body (everything after `@type{key,`) into a
    lowercased field-name -> value dict. Depth-aware comma-splitting (see
    _split_top_level) makes this correct for BOTH the multi-line, one-field-
    per-line style (arXiv/DataCite output) and the compact single-line,
    every-field-on-one-line style (Crossref's actual doi.org output) --
    a single-line-anchored regex (the previous approach) only handles the
    former and silently mis-parses the latter into one giant bogus field,
    which is what most real fetched entries in this project actually are."""
    fields = {}
    for chunk in _split_top_level(body):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip().lower()
        if not key:
            continue
        value = value.strip()
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1]
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        fields[key] = value.strip()
    return fields


def parse_bib(text: str) -> list[dict]:
    """Split a .bib file into entries. Deliberately simple: we control the
    generator, so we do not need a full BibTeX grammar."""
    entries = []
    matches = list(ENTRY_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        fields = _parse_fields(body)
        entries.append(
            {
                "type": m.group(1).lower(),
                "key": m.group(2),
                "fields": fields,
                "raw": body,
            }
        )
    return entries


def normalise_title(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return " ".join(t.split())


def has_identifier(entry: dict) -> bool:
    f = entry["fields"]
    if f.get("doi"):
        return True
    for key in ("eprint", "archiveprefix", "url", "note"):
        if "arxiv" in f.get(key, "").lower():
            return True
    return False


def collect_manuscript_text(paths: list[Path]) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in paths)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", action="append", default=None)
    ap.add_argument("--bib", default=str(BIB_PATH))
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    bib_path = Path(args.bib)
    if not bib_path.exists():
        print(f"ERROR: {bib_path} does not exist. Run scripts/build_refs.sh first.")
        return 1

    entries = parse_bib(bib_path.read_text(encoding="utf-8"))
    print(f"parsed {len(entries)} entries from {bib_path}")

    # 1. identifier present
    for e in entries:
        if not has_identifier(e):
            errors.append(f"no DOI or arXiv id: {e['key']}")

    # 2. duplicate DOIs
    dois = [e["fields"]["doi"].lower() for e in entries if e["fields"].get("doi")]
    for doi, n in Counter(dois).items():
        if n > 1:
            keys = [e["key"] for e in entries if e["fields"].get("doi", "").lower() == doi]
            errors.append(f"duplicate DOI {doi} under keys: {', '.join(keys)}")

    # 3. near-duplicate titles
    titled = [(e["key"], normalise_title(e["fields"].get("title", ""))) for e in entries]
    titled = [(k, t) for k, t in titled if t]
    for i in range(len(titled)):
        for j in range(i + 1, len(titled)):
            ratio = SequenceMatcher(None, titled[i][1], titled[j][1]).ratio()
            if ratio >= TITLE_SIM_THRESHOLD:
                warnings.append(
                    f"near-duplicate titles ({ratio:.2f}): {titled[i][0]} / {titled[j][0]}"
                )

    # 4 + 5. cite-key sync
    if args.manuscript:
        ms_paths = [Path(p) for p in args.manuscript]
    else:
        ms_paths = []
        for pattern in DEFAULT_MANUSCRIPT_GLOBS:
            ms_paths.extend(Path(".").glob(pattern))
    ms_paths = [p for p in ms_paths if p.exists()]

    bib_keys = {e["key"] for e in entries}

    if not ms_paths:
        warnings.append("no manuscript files found; skipped cite-key sync checks")
    else:
        text = collect_manuscript_text(ms_paths)
        cited: set[str] = set()
        for group in CITE_RE.findall(text):
            cited.update(k.strip() for k in group.split(",") if k.strip())

        for key in sorted(cited - bib_keys):
            errors.append(f"cited but not in refs.bib: {key}")
        for key in sorted(bib_keys - cited):
            warnings.append(f"in refs.bib but never cited: {key}")
        print(f"scanned {len(ms_paths)} manuscript file(s), {len(cited)} distinct cite keys")

    # 6. selection coverage
    if TSV_PATH.exists():
        with TSV_PATH.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        bib_dois = {d.lower() for d in dois}
        for row in rows:
            doi = (row.get("doi") or "").strip().lower()
            if doi and doi not in bib_dois:
                warnings.append(
                    f"selected but missing from refs.bib: [{row['block']}] {row['bibkey']} ({doi})"
                )

    for w in warnings:
        print(f"WARN   {w}")
    for e in errors:
        print(f"ERROR  {e}")

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
