import errno
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from icml_2026_repro import cli

UNREPLICATED_MODEL_CLAIMS = {
    "claim_4": {
        "status": "not replicated",
        "reason": "Named model outputs and paid API budget were not available.",
    },
    "claim_5": {
        "status": "not replicated",
        "reason": "Named model outputs and paid API budget were not available.",
    },
    "claim_6": {
        "status": "not replicated",
        "reason": "Named model outputs and paid API budget were not available.",
    },
}
LEGACY_ARTIFACT_NAMES = ("claim_1_audit.json", "claim_2_trace.json")
CANONICAL_SPACE_URL = (
    "https://huggingface.co/spaces/wrice/"
    "repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets"
)


def patch_evidence_builders(monkeypatch, claim_1_builder, claim_2_builder, claim_3_builder):
    monkeypatch.setattr(cli, "build_benchmark_evidence", claim_1_builder)
    monkeypatch.setattr(cli, "build_predictability_evidence", claim_2_builder)
    monkeypatch.setattr(cli, "build_future_adaptation_evidence", claim_3_builder)


def test_build_bundle_writes_portable_evidence_files(tmp_path, monkeypatch):
    claim_1 = {
        "observed": {"operations": 3, "trajectories": 2},
        "verdict": "reproduced",
    }
    claim_2 = {
        "observed": {"weighted_coverage_pct": 68.04},
        "verdict": "reproduced_from_released_outputs",
    }
    claim_3 = {
        "release_sweep": {"summary": {"target_preserved_cases": 52}},
        "verdict": "reproduced",
    }
    claim_3_output_dirs = []
    patch_evidence_builders(
        monkeypatch,
        lambda: claim_1,
        lambda: claim_2,
        lambda output_dir: claim_3_output_dirs.append(output_dir) or claim_3,
    )
    result = cli.build_bundle(tmp_path)

    assert result == tmp_path
    expected_files = {
        "claim_1_benchmark.json",
        "claim_2_predictability.json",
        "claim_3_future_adaptation.json",
        "claims_4_6_status.json",
        "environment.json",
        "README.md",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    assert json.loads((tmp_path / "claim_1_benchmark.json").read_text(encoding="utf-8")) == claim_1
    assert json.loads((tmp_path / "claim_2_predictability.json").read_text(encoding="utf-8")) == (
        claim_2
    )
    assert (
        json.loads((tmp_path / "claim_3_future_adaptation.json").read_text(encoding="utf-8"))
        == claim_3
    )
    assert json.loads((tmp_path / "claims_4_6_status.json").read_text(encoding="utf-8")) == (
        UNREPLICATED_MODEL_CLAIMS
    )
    assert (
        json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))["schema"]
        == "icml-2026-repro-bundle/v2"
    )
    assert len(claim_3_output_dirs) == 1
    assert claim_3_output_dirs[0] != tmp_path
    assert str(tmp_path) not in "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.iterdir()
    )
    bundle_readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    for required_text in (
        "52",
        "11,907",
        "35-821",
        "229",
        "164",
        "68.04%",
        "50/52",
        "52/52",
        "one residual fixture",
        "Claims 4-6 were not replicated",
        "released oracle outputs",
        "zero paid API calls",
        CANONICAL_SPACE_URL,
    ):
        assert required_text in bundle_readme


def test_build_bundle_upgrades_regular_legacy_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    for artifact_name in LEGACY_ARTIFACT_NAMES:
        (output_dir / artifact_name).write_text("legacy\n", encoding="utf-8")
    patch_evidence_builders(
        monkeypatch,
        lambda: {"verdict": "reproduced"},
        lambda: {"verdict": "reproduced_from_released_outputs"},
        lambda output_dir: {"verdict": "reproduced"},
    )

    cli.build_bundle(output_dir)

    assert {path.name for path in output_dir.iterdir()} == set(cli.BUNDLE_ARTIFACT_NAMES)


def test_build_bundle_rejects_unexpected_entry_before_evidence_work(tmp_path, monkeypatch):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    patch_evidence_builders(
        monkeypatch,
        lambda: pytest.fail("claim 1 evidence builder was called"),
        lambda: pytest.fail("claim 2 evidence builder was called"),
        lambda output_dir: pytest.fail("claim 3 evidence builder was called"),
    )

    with pytest.raises(RuntimeError, match="unexpected output directory entry: sentinel.txt"):
        cli.build_bundle(output_dir)

    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert {path.name for path in output_dir.iterdir()} == {"sentinel.txt"}


@pytest.mark.parametrize("artifact_name", cli.BUNDLE_ARTIFACT_NAMES)
def test_build_bundle_rejects_current_artifact_directory_before_evidence_work(
    tmp_path, monkeypatch, artifact_name
):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    artifact_path = output_dir / artifact_name
    artifact_path.mkdir()
    patch_evidence_builders(
        monkeypatch,
        lambda: pytest.fail("claim 1 evidence builder was called"),
        lambda: pytest.fail("claim 2 evidence builder was called"),
        lambda output_dir: pytest.fail("claim 3 evidence builder was called"),
    )

    with pytest.raises(RuntimeError, match=f"{artifact_name} must be a regular file"):
        cli.build_bundle(output_dir)

    assert artifact_path.is_dir()


@pytest.mark.parametrize("artifact_name", LEGACY_ARTIFACT_NAMES)
@pytest.mark.parametrize("entry_type", ["symlink", "directory"])
def test_build_bundle_rejects_non_regular_legacy_artifact_before_evidence_work(
    tmp_path, monkeypatch, artifact_name, entry_type
):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    external_file = tmp_path / "external.txt"
    external_file.write_text("unchanged\n", encoding="utf-8")
    legacy_path = output_dir / artifact_name
    if entry_type == "symlink":
        legacy_path.symlink_to(external_file)
    else:
        legacy_path.mkdir()
    patch_evidence_builders(
        monkeypatch,
        lambda: pytest.fail("claim 1 evidence builder was called"),
        lambda: pytest.fail("claim 2 evidence builder was called"),
        lambda output_dir: pytest.fail("claim 3 evidence builder was called"),
    )

    with pytest.raises(RuntimeError, match=f"{artifact_name} must be a regular file"):
        cli.build_bundle(output_dir)

    assert external_file.read_text(encoding="utf-8") == "unchanged\n"
    if entry_type == "symlink":
        assert legacy_path.is_symlink()
    else:
        assert legacy_path.is_dir()


def test_build_bundle_preserves_legacy_artifacts_when_new_artifact_write_fails(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    for artifact_name in LEGACY_ARTIFACT_NAMES:
        (output_dir / artifact_name).write_text("legacy\n", encoding="utf-8")
    patch_evidence_builders(
        monkeypatch,
        lambda: {"verdict": "reproduced"},
        lambda: {"verdict": "reproduced_from_released_outputs"},
        lambda output_dir: {"verdict": "reproduced"},
    )

    def fail_write(file_descriptor, contents):
        del file_descriptor, contents
        raise OSError(errno.EIO, "simulated write failure")

    monkeypatch.setattr(cli.os, "write", fail_write)

    with pytest.raises(RuntimeError, match="could not write claim_1_benchmark.json"):
        cli.build_bundle(output_dir)

    assert {path.name for path in output_dir.iterdir()} == set(LEGACY_ARTIFACT_NAMES)
    for artifact_name in LEGACY_ARTIFACT_NAMES:
        assert (output_dir / artifact_name).read_text(encoding="utf-8") == "legacy\n"


def test_build_bundle_accepts_raw_macos_temporary_directory_alias(monkeypatch):
    if cli.platform.system() != "Darwin":
        pytest.skip("macOS root alias regression")

    with TemporaryDirectory(prefix="nape-repro-alias-") as temporary_directory:
        raw_temporary_path = Path(temporary_directory)
        root_component = raw_temporary_path.parts[1]
        if root_component not in {"var", "tmp"} or not (Path("/") / root_component).is_symlink():
            pytest.skip("temporary directory does not use a standard macOS root alias")
        output_dir = raw_temporary_path / "bundle"
        patch_evidence_builders(
            monkeypatch,
            lambda: {"verdict": "reproduced"},
            lambda: {"verdict": "reproduced_from_released_outputs"},
            lambda output_dir: {"verdict": "reproduced"},
        )

        result = cli.build_bundle(output_dir)

        assert result == output_dir
        assert {path.name for path in output_dir.iterdir()} == set(cli.BUNDLE_ARTIFACT_NAMES)


def test_build_bundle_rejects_symlinked_output_directory_before_evidence_work(
    tmp_path, monkeypatch
):
    external_directory = tmp_path / "external"
    external_directory.mkdir()
    sentinel = external_directory / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    output_dir = tmp_path / "bundle"
    output_dir.symlink_to(external_directory, target_is_directory=True)
    patch_evidence_builders(
        monkeypatch,
        lambda: pytest.fail("claim 1 evidence builder was called"),
        lambda: pytest.fail("claim 2 evidence builder was called"),
        lambda output_dir: pytest.fail("claim 3 evidence builder was called"),
    )

    with pytest.raises(RuntimeError, match="output directory must not be a symlink"):
        cli.build_bundle(output_dir)

    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert {path.name for path in external_directory.iterdir()} == {"sentinel.txt"}


def test_build_bundle_rejects_symlinked_output_ancestor_before_evidence_work(tmp_path, monkeypatch):
    external_directory = tmp_path / "external"
    external_directory.mkdir()
    sentinel = external_directory / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    linked_ancestor = tmp_path / "link"
    linked_ancestor.symlink_to(external_directory, target_is_directory=True)
    patch_evidence_builders(
        monkeypatch,
        lambda: pytest.fail("claim 1 evidence builder was called"),
        lambda: pytest.fail("claim 2 evidence builder was called"),
        lambda output_dir: pytest.fail("claim 3 evidence builder was called"),
    )

    with pytest.raises(RuntimeError, match="output directory.*symlink"):
        cli.build_bundle(linked_ancestor / "bundle")

    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert {path.name for path in external_directory.iterdir()} == {"sentinel.txt"}


@pytest.mark.parametrize("alias_name", ["var", "tmp"])
def test_build_bundle_rejects_user_symlink_named_like_macos_root_alias(
    tmp_path, monkeypatch, alias_name
):
    external_directory = tmp_path / "external"
    external_directory.mkdir()
    sentinel = external_directory / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    user_alias = tmp_path / alias_name
    user_alias.symlink_to(external_directory, target_is_directory=True)
    patch_evidence_builders(
        monkeypatch,
        lambda: pytest.fail("claim 1 evidence builder was called"),
        lambda: pytest.fail("claim 2 evidence builder was called"),
        lambda output_dir: pytest.fail("claim 3 evidence builder was called"),
    )

    with pytest.raises(RuntimeError, match="output directory.*symlink"):
        cli.build_bundle(user_alias / "bundle")

    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert {path.name for path in external_directory.iterdir()} == {"sentinel.txt"}


@pytest.mark.parametrize("artifact_name", cli.BUNDLE_ARTIFACT_NAMES)
def test_build_bundle_rejects_symlinked_artifact_before_evidence_work(
    tmp_path, monkeypatch, artifact_name
):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    external_file = tmp_path / "external.txt"
    external_file.write_text("unchanged\n", encoding="utf-8")
    (output_dir / artifact_name).symlink_to(external_file)
    patch_evidence_builders(
        monkeypatch,
        lambda: pytest.fail("claim 1 evidence builder was called"),
        lambda: pytest.fail("claim 2 evidence builder was called"),
        lambda output_dir: pytest.fail("claim 3 evidence builder was called"),
    )

    with pytest.raises(RuntimeError, match=f"{artifact_name} must be a regular file"):
        cli.build_bundle(output_dir)

    assert external_file.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.parametrize("artifact_name", cli.BUNDLE_ARTIFACT_NAMES)
def test_build_bundle_replaces_hard_link_without_changing_external_inode(
    tmp_path, monkeypatch, artifact_name
):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    external_file = tmp_path / "external.txt"
    external_file.write_text("unchanged\n", encoding="utf-8")
    os.link(external_file, output_dir / artifact_name)
    claim_1 = {"verdict": "reproduced"}
    claim_2 = {"verdict": "reproduced_from_released_outputs"}
    claim_3 = {"verdict": "reproduced"}
    patch_evidence_builders(
        monkeypatch,
        lambda: claim_1,
        lambda: claim_2,
        lambda output_dir: claim_3,
    )

    cli.build_bundle(output_dir)

    assert external_file.read_text(encoding="utf-8") == "unchanged\n"
    assert not external_file.samefile(output_dir / artifact_name)
    artifact_text = (output_dir / artifact_name).read_text(encoding="utf-8")
    if artifact_name == "claim_1_benchmark.json":
        assert json.loads(artifact_text) == claim_1
    elif artifact_name == "claim_2_predictability.json":
        assert json.loads(artifact_text) == claim_2
    elif artifact_name == "claim_3_future_adaptation.json":
        assert json.loads(artifact_text) == claim_3
    elif artifact_name == "claims_4_6_status.json":
        assert json.loads(artifact_text) == UNREPLICATED_MODEL_CLAIMS
    elif artifact_name == "environment.json":
        assert json.loads(artifact_text)["schema"] == "icml-2026-repro-bundle/v2"
    else:
        assert artifact_text.startswith("# NAPE Reproduction Evidence\n")
    assert {path.name for path in output_dir.iterdir()} == set(cli.BUNDLE_ARTIFACT_NAMES)


def test_build_bundle_atomically_replaces_artifact_symlink_after_preflight(tmp_path, monkeypatch):
    output_dir = tmp_path / "bundle"
    external_file = tmp_path / "external.txt"
    external_file.write_text("unchanged\n", encoding="utf-8")
    claim_1 = {"verdict": "reproduced"}

    def build_claim_1_with_symlink_swap():
        (output_dir / "claim_1_benchmark.json").symlink_to(external_file)
        return claim_1

    patch_evidence_builders(
        monkeypatch,
        build_claim_1_with_symlink_swap,
        lambda: {"verdict": "reproduced_from_released_outputs"},
        lambda output_dir: {"verdict": "reproduced"},
    )

    result = cli.build_bundle(output_dir)

    assert result == output_dir
    assert external_file.read_text(encoding="utf-8") == "unchanged\n"
    assert not (output_dir / "claim_1_benchmark.json").is_symlink()
    assert (
        json.loads((output_dir / "claim_1_benchmark.json").read_text(encoding="utf-8")) == claim_1
    )
    assert {path.name for path in output_dir.iterdir()} == set(cli.BUNDLE_ARTIFACT_NAMES)


@pytest.mark.parametrize("failing_operation", ["write", "replace"])
def test_build_bundle_removes_staging_file_after_artifact_failure(
    tmp_path, monkeypatch, failing_operation
):
    output_dir = tmp_path / "bundle"
    patch_evidence_builders(
        monkeypatch,
        lambda: {"verdict": "reproduced"},
        lambda: {"verdict": "reproduced_from_released_outputs"},
        lambda output_dir: {"verdict": "reproduced"},
    )
    if failing_operation == "write":

        def fail_write(file_descriptor, contents):
            del file_descriptor, contents
            raise OSError(errno.EIO, "simulated write failure")

        monkeypatch.setattr(cli.os, "write", fail_write)
    else:

        def fail_replace(source, destination, **kwargs):
            del source, destination, kwargs
            raise OSError(errno.EIO, "simulated replace failure")

        monkeypatch.setattr(cli.os, "replace", fail_replace)

    with pytest.raises(
        RuntimeError,
        match=f"could not {failing_operation} claim_1_benchmark.json",
    ):
        cli.build_bundle(output_dir)

    assert list(output_dir.iterdir()) == []
