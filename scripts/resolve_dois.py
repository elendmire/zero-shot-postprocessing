#!/usr/bin/env python3
"""Resolve DOIs from titles via the Crossref API for gap_fills.tsv.

Entries whose `doi` column is RESOLVE get looked up by title. The script does
NOT auto-accept: it prints the top candidate with a title-similarity score and
writes a review file. A human confirms before anything reaches selected_dois.tsv.

Rationale: a wrong DOI is worse than a missing one. Fuzzy title matching finds
the right paper most of the time and the wrong paper some of the time, so the
confirmation step is not optional.

Usage:
    python scripts/resolve_dois.py
    python scripts/resolve_dois.py --threshold 0.85
Output:
    docs/references/gap_fills_resolved.tsv   (review this by hand)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

IN_PATH = Path("docs/references/gap_fills.tsv")
OUT_PATH = Path("docs/references/gap_fills_resolved.tsv")
API = "https://api.crossref.org/works"
# Crossref asks for a contact address in the User-Agent; polite pool is faster.
UA = "ZeroPP-refs/1.0 (mailto:avcio20@itu.edu.tr)"


def normalise(t: str) -> str:
    import re

    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return " ".join(t.split())


def query_crossref(title: str, rows: int = 5) -> list[dict]:
    params = urllib.parse.urlencode({"query.bibliographic": title, "rows": rows})
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as fh:
        payload = json.load(fh)
    return payload.get("message", {}).get("items", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--infile", default=str(IN_PATH))
    args = ap.parse_args()

    src = Path(args.infile)
    if not src.exists():
        print(f"missing {src}", file=sys.stderr)
        return 1

    with src.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    out_rows = []
    for row in rows:
        doi = (row.get("doi") or "").strip()
        title = (row.get("title_for_crossref") or "").strip()
        bibkey = row.get("bibkey", "")

        if doi and doi != "RESOLVE":
            out_rows.append({**row, "resolved_doi": doi, "match_score": "", "verdict": "PREEXISTING"})
            continue

        if not title:
            out_rows.append({**row, "resolved_doi": "", "match_score": "", "verdict": "NO_TITLE"})
            continue

        try:
            items = query_crossref(title)
        except Exception as exc:  # network, rate limit, malformed payload
            print(f"  ERROR  {bibkey}: {exc}", file=sys.stderr)
            out_rows.append({**row, "resolved_doi": "", "match_score": "", "verdict": "LOOKUP_FAILED"})
            time.sleep(2)
            continue

        best, best_score = None, 0.0
        for it in items:
            cand_title = (it.get("title") or [""])[0]
            score = SequenceMatcher(None, normalise(title), normalise(cand_title)).ratio()
            if score > best_score:
                best, best_score = it, score

        if best is None:
            verdict, resolved = "NO_RESULTS", ""
        elif best_score >= args.threshold:
            verdict, resolved = "CONFIRM", best.get("DOI", "")
        else:
            verdict, resolved = "LOW_MATCH_REVIEW", best.get("DOI", "")

        cand_title = (best.get("title") or [""])[0] if best else ""
        cand_year = ""
        if best:
            parts = best.get("issued", {}).get("date-parts", [[None]])
            cand_year = str(parts[0][0]) if parts and parts[0] else ""

        print(f"  {verdict:17s} {best_score:.2f}  {bibkey}")
        print(f"      want: {title[:90]}")
        print(f"      got : {cand_title[:90]}  ({cand_year})")

        out_rows.append(
            {
                **row,
                "resolved_doi": resolved,
                "match_score": f"{best_score:.3f}",
                "verdict": verdict,
                "crossref_title": cand_title,
                "crossref_year": cand_year,
            }
        )
        time.sleep(1)

    fieldnames = list(rows[0].keys()) + [
        "resolved_doi",
        "match_score",
        "verdict",
        "crossref_title",
        "crossref_year",
    ]
    with OUT_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    n_confirm = sum(1 for r in out_rows if r.get("verdict") == "CONFIRM")
    n_review = sum(1 for r in out_rows if r.get("verdict") in {"LOW_MATCH_REVIEW", "NO_RESULTS", "LOOKUP_FAILED"})
    print(f"\nwrote {OUT_PATH}")
    print(f"{n_confirm} above threshold, {n_review} need manual review")
    print("REVIEW EVERY ROW BY HAND before copying into selected_dois.tsv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
