"""Unit tests for scripts/check_references.py's reference-integrity checks.

check_references.py lives under scripts/, not src/zeropp/, so it is loaded here via
importlib.util.spec_from_file_location, the same mechanism this project's other
scripts/ tests use (see test_data_size_sweep.py). Only top-level imports/definitions
execute on load; main() only runs when the module is invoked as a script, so importing
it here is side-effect-free.

Each test builds a minimal synthetic refs.bib (and, where relevant, a synthetic
manuscript file) in tmp_path and drives check_references.main() through sys.argv,
exactly as it would be invoked from the command line — this exercises the real
argument parsing and the real end-to-end main() control flow, not just the
individual helper functions in isolation.
"""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_references.py"
_SPEC = importlib.util.spec_from_file_location("check_references_module", _SCRIPT_PATH)
check_references = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_references)


def _write_bib(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "refs.bib"
    p.write_text(content, encoding="utf-8")
    return p


def _run(monkeypatch, bib_path: Path, manuscript_path: Path | None):
    """Invoke check_references.main() via sys.argv, always passing an explicit
    --manuscript so the cite-key-sync checks never depend on this repo's real
    layout (e.g. whether a paper/ directory happens to exist yet)."""
    argv = ["check_references.py", "--bib", str(bib_path)]
    if manuscript_path is not None:
        argv += ["--manuscript", str(manuscript_path)]
    monkeypatch.setattr("sys.argv", argv)
    return check_references.main()


def test_entry_without_doi_or_arxiv_is_caught_as_error(tmp_path, monkeypatch, capsys):
    bib = _write_bib(
        tmp_path,
        """
@article{orphan_2020_noid,
  title = {A Paper With No Identifier At All},
}
""",
    )
    missing_manuscript = tmp_path / "does_not_exist.tex"
    exit_code = _run(monkeypatch, bib, missing_manuscript)
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "ERROR  no DOI or arXiv id: orphan_2020_noid" in out


def test_duplicate_doi_is_caught_as_error_case_insensitively(tmp_path, monkeypatch, capsys):
    bib = _write_bib(
        tmp_path,
        """
@article{key_one,
  title = {Paper One About Something},
  doi = {10.1234/SAME},
}

@article{key_two,
  title = {Paper Two, A Completely Different Topic},
  doi = {10.1234/same},
}
""",
    )
    missing_manuscript = tmp_path / "does_not_exist.tex"
    exit_code = _run(monkeypatch, bib, missing_manuscript)
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "duplicate DOI 10.1234/same" in out
    assert "key_one" in out and "key_two" in out


def test_near_duplicate_titles_is_a_warning_not_an_error(tmp_path, monkeypatch, capsys):
    # Same work under two keys with a one-word title variation (singular vs.
    # plural) -- similarity independently verified at ~0.99, comfortably over
    # the module's 0.92 threshold. Different DOIs so finding 2 (duplicate DOI)
    # does not also fire and mask what this test is checking.
    bib = _write_bib(
        tmp_path,
        """
@article{author_2020_paperone,
  title = {Improving Ensemble Forecast Calibration With Neural Networks},
  doi = {10.1234/aaaa},
}

@article{author_2020_papertwo,
  title = {Improving Ensemble Forecast Calibration with Neural Network},
  doi = {10.1234/bbbb},
}
""",
    )
    missing_manuscript = tmp_path / "does_not_exist.tex"
    exit_code = _run(monkeypatch, bib, missing_manuscript)
    out = capsys.readouterr().out
    assert exit_code == 0, "a near-duplicate title must not be an ERROR"
    assert "WARN   near-duplicate titles" in out
    assert "ERROR" not in out


def test_cite_key_sync_is_bidirectional(tmp_path, monkeypatch, capsys):
    bib = _write_bib(
        tmp_path,
        """
@article{smith_2020_cited,
  title = {A Paper That Gets Cited},
  doi = {10.1234/smith2020},
}

@article{jones_2021_never_cited,
  title = {A Paper That Never Gets Cited},
  doi = {10.1234/jones2021},
}
""",
    )
    manuscript = tmp_path / "main.tex"
    manuscript.write_text(
        r"As shown by \citep{smith_2020_cited} and also \cite{ghost_key_not_in_bib}.",
        encoding="utf-8",
    )
    exit_code = _run(monkeypatch, bib, manuscript)
    out = capsys.readouterr().out
    # Direction 1: cited in the manuscript but absent from refs.bib -> ERROR.
    assert exit_code == 1
    assert "ERROR  cited but not in refs.bib: ghost_key_not_in_bib" in out
    # Direction 2: present in refs.bib but never cited -> WARN, not ERROR.
    assert "WARN   in refs.bib but never cited: jones_2021_never_cited" in out
    # The key that IS cited and IS in refs.bib must not appear in either list.
    assert "smith_2020_cited" not in [
        line.split(": ", 1)[-1] for line in out.splitlines() if "not in refs.bib" in line or "never cited" in line
    ]


def test_compact_single_line_crossref_style_entry_parses_correctly(tmp_path, monkeypatch, capsys):
    # This is the ACTUAL shape doi.org's content negotiation returns for a
    # Crossref-registered DOI: every field on one line, comma-separated, not
    # one-field-per-line. A regression guard: a naive per-line field regex
    # parses this into one giant bogus "title" field and silently loses the
    # doi field entirely, which is exactly the bug found when this test suite
    # was first run against the real, build_refs.sh-generated refs.bib.
    bib = _write_bib(
        tmp_path,
        ' @article{demaeyer_2023_euppbench, title={The EUPPBench postprocessing'
        ' benchmark dataset v1.0}, volume={15}, ISSN={1866-3516},'
        ' url={http://dx.doi.org/10.5194/essd-15-2635-2023},'
        ' DOI={10.5194/essd-15-2635-2023}, number={6}, journal={Earth System'
        ' Science Data}, publisher={Copernicus GmbH}, author={Demaeyer, Jonathan'
        ' and Lerch, Sebastian}, year={2023}, month=June, pages={2635–2653} }\n',
    )
    missing_manuscript = tmp_path / "does_not_exist.tex"
    exit_code = _run(monkeypatch, bib, missing_manuscript)
    out = capsys.readouterr().out
    assert exit_code == 0, f"a real, DOI-bearing entry must not be an ERROR; got:\n{out}"
    assert "no DOI or arXiv id: demaeyer_2023_euppbench" not in out


def test_citealp_and_citealt_are_recognized_as_citations(tmp_path, monkeypatch, capsys):
    # Regression guard for a real bug found in the 04_results.tex review: CITE_RE
    # only matched \cite, \citet, \citep (and their starred forms), not natbib's
    # \citealp / \citealt -- so a real, present citation like
    # \citealp{luger_2004_exact_tests} in 03_methods.tex was silently invisible to
    # the cite-key-sync check and falsely reported as "never cited" (direction 2
    # of test_cite_key_sync_is_bidirectional above), which in turn corrupted the
    # D4 citation-pruning audit that reads this same "never cited" list.
    bib = _write_bib(
        tmp_path,
        """
@article{alp_2020_cited,
  title = {A Paper Cited Only Via Citealp},
  doi = {10.1234/alp2020},
}

@article{alt_2021_cited,
  title = {A Paper Cited Only Via Citealt},
  doi = {10.1234/alt2021},
}
""",
    )
    manuscript = tmp_path / "main.tex"
    manuscript.write_text(
        r"Following the logic of exact tests \citealp{alp_2020_cited}, and also "
        r"\citealt{alt_2021_cited}.",
        encoding="utf-8",
    )
    exit_code = _run(monkeypatch, bib, manuscript)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "never cited: alp_2020_cited" not in out
    assert "never cited: alt_2021_cited" not in out


def test_empty_manuscript_gives_warning_not_error(tmp_path, monkeypatch, capsys):
    bib = _write_bib(
        tmp_path,
        """
@article{fine_2020_entry,
  title = {A Perfectly Fine Entry},
  doi = {10.1234/fine2020},
}
""",
    )
    missing_manuscript = tmp_path / "does_not_exist.tex"
    exit_code = _run(monkeypatch, bib, missing_manuscript)
    out = capsys.readouterr().out
    assert exit_code == 0, "no manuscript found must not be an ERROR"
    assert "WARN   no manuscript files found; skipped cite-key sync checks" in out
    assert "ERROR" not in out
