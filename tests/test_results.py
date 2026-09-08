import json
import subprocess

import pandas as pd
import pytest

from zeropp.eval.results import write_result


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


def test_write_result_creates_parquet_and_json(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="emos-v1", config={"k": 1}, out_dir=str(tmp_path))
    assert (tmp_path / "test_result.parquet").exists()
    assert (tmp_path / "test_result.json").exists()


def test_write_result_parquet_content_matches(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="emos-v1", config={"k": 1}, out_dir=str(tmp_path))
    loaded = pd.read_parquet(tmp_path / "test_result.parquet")
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_write_result_json_has_required_provenance_fields(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="emos-v1", config={"k": 1}, out_dir=str(tmp_path))
    meta = json.loads((tmp_path / "test_result.json").read_text())
    assert meta["model_version"] == "emos-v1"
    assert meta["config"] == {"k": 1}
    assert "config_hash" in meta
    assert "git_sha" in meta
    assert "written_at" in meta


def test_write_result_git_sha_matches_real_head(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="emos-v1", config={}, out_dir=str(tmp_path))
    meta = json.loads((tmp_path / "test_result.json").read_text())
    real_head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    assert meta["git_sha"] == real_head


def test_write_result_config_hash_is_deterministic(tmp_path, sample_df):
    write_result(sample_df, name="r1", model_version="v1", config={"a": 1, "b": 2}, out_dir=str(tmp_path))
    write_result(sample_df, name="r2", model_version="v1", config={"b": 2, "a": 1}, out_dir=str(tmp_path))
    meta1 = json.loads((tmp_path / "r1.json").read_text())
    meta2 = json.loads((tmp_path / "r2.json").read_text())
    assert meta1["config_hash"] == meta2["config_hash"]  # key order must not matter


def test_write_result_creates_out_dir_if_missing(tmp_path, sample_df):
    out_dir = tmp_path / "nested" / "results"
    write_result(sample_df, name="r", model_version="v1", config={}, out_dir=str(out_dir))
    assert (out_dir / "r.parquet").exists()


def test_write_result_json_has_git_dirty_bool(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="v1", config={}, out_dir=str(tmp_path))
    meta = json.loads((tmp_path / "test_result.json").read_text())
    assert "git_dirty" in meta
    assert isinstance(meta["git_dirty"], bool)
    # Ground truth: a real, independent call to `git status --porcelain` right
    # now must agree with what write_result recorded (allowing for the fact that
    # write_result's own tmp_path artifacts are outside the repo and don't affect
    # this) -- not just "is a bool", but the RIGHT bool.
    porcelain = subprocess.check_output(["git", "status", "--porcelain"]).decode()
    assert meta["git_dirty"] == (len(porcelain.strip()) > 0)


def test_write_result_json_has_source_tree_sha256(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="v1", config={}, out_dir=str(tmp_path))
    meta = json.loads((tmp_path / "test_result.json").read_text())
    assert "source_tree_sha256" in meta
    assert isinstance(meta["source_tree_sha256"], str)
    assert len(meta["source_tree_sha256"]) == 64  # hex-encoded sha256 digest


def test_write_result_source_tree_sha256_changes_when_a_source_file_changes(tmp_path, sample_df):
    # Real repo files under src/ are hashed as-is (no source_tree_dirs override
    # exists to redirect this at a fixture), so this test edits a throwaway file
    # inside src/zeropp/ itself, restoring it afterward -- the point is proving
    # the digest is actually content-sensitive, not merely present.
    from zeropp.eval.results import REPO_ROOT

    probe_path = REPO_ROOT / "src" / "zeropp" / "_provenance_test_probe.py"
    assert not probe_path.exists(), "leftover probe file from a previous failed test run"
    try:
        write_result(sample_df, name="before", model_version="v1", config={}, out_dir=str(tmp_path))
        meta_before = json.loads((tmp_path / "before.json").read_text())

        probe_path.write_text("# provenance test probe\n")
        write_result(sample_df, name="after", model_version="v1", config={}, out_dir=str(tmp_path))
        meta_after = json.loads((tmp_path / "after.json").read_text())

        assert meta_before["source_tree_sha256"] != meta_after["source_tree_sha256"]
    finally:
        probe_path.unlink(missing_ok=True)
