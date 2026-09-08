#!/usr/bin/env bash
# Build refs.bib from the curated DOI list by fetching canonical BibTeX
# from doi.org content negotiation. Never writes an entry from memory.
#
# Usage:
#   ./scripts/build_refs.sh                      # all blocks
#   ./scripts/build_refs.sh B8 B8b               # selected blocks only
#
# Output: docs/references/refs.bib  (rebuilt from scratch each run)
#         docs/references/failed_dois.txt  (anything doi.org would not resolve)

set -uo pipefail

TSV="docs/references/selected_dois.tsv"
OUT="docs/references/refs.bib"
FAILED="docs/references/failed_dois.txt"

[ -f "$TSV" ] || { echo "missing $TSV" >&2; exit 1; }

: > "$OUT"
: > "$FAILED"

# Captured before the read loop below: the function is always called as
# `wanted_block "$block" "$@"`, so inside the function $# is at least 1 even
# when the script itself received zero filter args — checking $# inside the
# function can never detect "no filter" correctly. n_filter_args records the
# script's own original argument count instead.
n_filter_args=$#

wanted_block() {
  local b="$1"; shift
  [ "$n_filter_args" -eq 0 ] && return 0
  for w in "$@"; do [ "$b" = "$w" ] && return 0; done
  return 1
}

n_ok=0
n_fail=0

# skip header
tail -n +2 "$TSV" | while IFS=$'\t' read -r block bibkey doi label priority; do
  [ -z "${doi:-}" ] && continue
  wanted_block "$block" "$@" || continue

  response=$(curl -sL --max-time 30 -w '\nHTTP_STATUS:%{http_code}' \
        -H "Accept: application/x-bibtex; charset=utf-8" \
        "https://doi.org/${doi}")
  http_code=$(printf '%s' "$response" | grep -o 'HTTP_STATUS:[0-9]*$' | cut -d: -f2)
  bib=$(printf '%s' "$response" | sed '$ s/HTTP_STATUS:[0-9]*$//')

  # A DOI that fails to resolve on doi.org's end often still returns HTTP 200
  # with an HTML "DOI Not Found" error page rather than an HTTP error code --
  # and that page can contain a line starting with "@" (e.g. a CSS @media
  # rule), which used to fool a naive "does any line start with @" check into
  # silently accepting the HTML as a valid bib entry. Verified on this exact
  # dataset: this is what happened to a truncated legacy-AMS DOI, and the fix
  # was tightened further after that first fix still relied on only two of
  # three independent signals -- now ALL THREE must hold, not just one or two:
  # (a) the first non-blank character of the body is "@"; (b) the body
  # contains an actual doi/DOI field (an HTML error page could in principle
  # mention the string "doi" without being a bib entry, and a genuine bib
  # entry always carries one); (c) the body contains no HTML markers anywhere.
  # Any single one of these failing is treated as a failed fetch.
  is_valid=1
  [ "$http_code" = "200" ] || is_valid=0
  [ -n "$bib" ] || is_valid=0

  starts_with_at=0
  if [ "$is_valid" -eq 1 ]; then
    first_line=$(printf '%s' "$bib" | grep -m1 '[^[:space:]]' | sed 's/^[[:space:]]*//')
    case "$first_line" in
      @*) starts_with_at=1 ;;
    esac
  fi

  has_doi_field=0
  if [ "$is_valid" -eq 1 ] && printf '%s' "$bib" | grep -qiE '(^|[,{[:space:]])doi[[:space:]]*=' ; then
    has_doi_field=1
  fi

  has_html_marker=0
  if [ "$is_valid" -eq 1 ] && printf '%s' "$bib" | grep -qi '<!doctype\|<html'; then
    has_html_marker=1
  fi

  if [ "$starts_with_at" -ne 1 ] || [ "$has_doi_field" -ne 1 ] || [ "$has_html_marker" -ne 0 ]; then
    is_valid=0
  fi

  if [ "$is_valid" -ne 1 ]; then
    echo "FAILED  $block  $bibkey  $doi  -- $label (http=${http_code:-?})" | tee -a "$FAILED" >&2
    n_fail=$((n_fail+1))
    sleep 1
    continue
  fi

  # Replace the publisher-assigned key with our normalised bibkey so that
  # \cite{} keys stay stable and human-readable across rebuilds.
  # (Uses awk, not sed: GNU sed's `0,/regex/` range address and empty-pattern
  # `s//repl/` reuse are not portable to BSD/macOS sed, which silently no-ops
  # instead of erroring — this was verified to break the substitution for
  # every single entry on macOS.)
  normalised=$(printf '%s' "$bib" | awk -v newkey="$bibkey" '
    NR==1 {
      match($0, /@[A-Za-z]+\{/)
      if (RSTART > 0) {
        prefix = substr($0, 1, RSTART+RLENGTH-1)
        rest = substr($0, RSTART+RLENGTH)
        sub(/^[^,]*,/, newkey ",", rest)
        print prefix rest
        next
      }
    }
    { print }
  ')

  # doi.org's Crossref-sourced bibtex leaves raw HTML entities in field
  # values (e.g. journal names with "&" render as literal "&amp;", not a
  # real "&"). BibTeX passes these through uninterpreted into the .bbl,
  # where an unescaped "&" is a fatal LaTeX "Misplaced alignment tab"
  # error. Confirmed on this exact dataset: 3 entries (JBES x2, Communications
  # Earth & Environment) carried "&amp;" straight from doi.org. Decode the
  # handful of entities Crossref actually emits, escaping "&" for LaTeX.
  # Some Crossref titles also carry literal HTML formatting tags around a
  # software-package name (e.g. "<b>scoringRules</b>") -- confirmed on this
  # exact dataset for two entries (allen_2023_weighted_scoringrules,
  # jordan_2017_scoringrules). Decode entities first (above), then strip any
  # well-formed open/close tag; done on a separate line from the entity
  # decode above so a tag stripped here can never re-expose an entity that
  # was inside it.
  decoded=$(printf '%s' "$normalised" | sed \
    -e 's/&amp;/\\\&/g' \
    -e 's/&lt;/</g' \
    -e 's/&gt;/>/g' \
    -e "s/&#39;/'/g" \
    -e 's/&quot;/"/g' \
    -e 's/<\/\{0,1\}[a-zA-Z][a-zA-Z0-9]*[[:space:]]*\/\{0,1\}>//g')

  {
    printf '%% [%s] %s\n' "$block" "$label"
    printf '%s\n\n' "$decoded"
  } >> "$OUT"

  echo "ok      $block  $bibkey"
  n_ok=$((n_ok+1))
  sleep 1   # be polite to doi.org
done

echo
echo "wrote $OUT"
grep -cE '^\s*@' "$OUT" 2>/dev/null | xargs -I{} echo "entries: {}"
if [ -s "$FAILED" ]; then
  echo "FAILURES recorded in $FAILED -- resolve these manually, do not invent them"
fi
