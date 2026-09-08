"""Regression tests for scripts/build_refs.sh's HTML-entity/tag decode step.

build_refs.sh is a bash script with no Python entry point (its main loop makes
real network calls to doi.org), so it cannot be imported or driven end-to-end
here the way check_references.py is in test_references.py. Instead, this test
extracts the exact `sed` pipeline build_refs.sh applies to decode doi.org's
Crossref BibTeX (the source of two real bugs found in the 04_results.tex
review: un-decoded "&amp;" breaking LaTeX compilation, and literal "<b>...</b>"
tags surviving into the compiled References list) and runs that pipeline via
subprocess against fixed input.

The sed expressions are duplicated here rather than parsed out of the script,
so test_decode_step_still_present_verbatim_in_build_refs_sh fails loudly if
build_refs.sh's decode step is edited without this test being updated to
match -- silent drift would defeat the point of a regression test.
"""
import subprocess
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_refs.sh"

_DECODE_SED_EXPRS = [
    r"s/&amp;/\\\&/g",
    r"s/&lt;/</g",
    r"s/&gt;/>/g",
    r"s/&#39;/'/g",
    r's/&quot;/"/g',
    r"s/<\/\{0,1\}[a-zA-Z][a-zA-Z0-9]*[[:space:]]*\/\{0,1\}>//g",
]


def _script_text() -> str:
    return _SCRIPT_PATH.read_text(encoding="utf-8")


def test_decode_step_still_present_verbatim_in_build_refs_sh():
    text = _script_text()
    for expr in _DECODE_SED_EXPRS:
        assert expr in text, (
            f"build_refs.sh's decode step changed (missing: {expr!r}) -- update "
            "_DECODE_SED_EXPRS in this test to match the new step, then re-verify "
            "the behavioural tests below still pass against the new pipeline."
        )


def _run_decode(raw: str) -> str:
    cmd = ["sed"]
    for expr in _DECODE_SED_EXPRS:
        cmd += ["-e", expr]
    result = subprocess.run(cmd, input=raw, capture_output=True, text=True, check=True)
    return result.stdout


def test_html_ampersand_entity_is_decoded_to_latex_escaped_ampersand():
    # Real bug: doi.org's Crossref BibTeX left "&amp;" un-decoded in 3 journal
    # fields (e.g. "Journal of Business &amp; Economic Statistics"). BibTeX
    # passed it straight into main.bbl as a literal, fatal LaTeX "Misplaced
    # alignment tab character &" error.
    raw = "journal={Journal of Business &amp; Economic Statistics}"
    out = _run_decode(raw)
    assert out.strip() == r"journal={Journal of Business \& Economic Statistics}"
    assert "amp;" not in out


def test_html_bold_tags_are_stripped():
    # Real bug: doi.org's Crossref BibTeX also left literal HTML tags in a
    # title field ("Weighted <b>scoringRules</b>: ..."), which rendered
    # verbatim -- tags included -- in the compiled References list.
    raw = "title={Weighted <b>scoringRules</b>: Emphasizing Particular Outcomes}"
    out = _run_decode(raw)
    assert out.strip() == "title={Weighted scoringRules: Emphasizing Particular Outcomes}"
    assert "<b>" not in out and "</b>" not in out


def test_decode_is_a_no_op_on_plain_text():
    raw = "title={A Perfectly Normal Title With No Entities Or Tags}"
    out = _run_decode(raw)
    assert out.strip() == raw
