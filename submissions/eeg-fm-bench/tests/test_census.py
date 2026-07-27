from pathlib import Path

from eeg_fm_bench_repro.census import run_census_audit


def _write_fixture_snapshot(snapshot: Path) -> None:
    wrapper = snapshot / "data" / "processor" / "wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        """
DATASET_SELECTOR = {
    "MotorDataset": MotorDataset,
    "SleepDataset": SleepDataset,
}
""",
        encoding="utf-8",
    )

    dataset_dir = snapshot / "data" / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "motor.py").write_text(
        "class MotorDataset:\n    pass\n", encoding="utf-8"
    )
    (dataset_dir / "sleep.py").write_text(
        "class SleepDataset:\n    pass\n", encoding="utf-8"
    )

    conf_dir = snapshot / "assets" / "conf"
    conf_dir.mkdir(parents=True)
    (conf_dir / "motor.yaml").write_text(
        "dataset: MotorDataset\nparadigm: motor_imagery\n", encoding="utf-8"
    )
    (conf_dir / "sleep.yaml").write_text(
        "dataset: SleepDataset\nparadigm: sleep_staging\n", encoding="utf-8"
    )


def test_census_record_contains_computed_mapping_and_labeled_paper_context(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_fixture_snapshot(snapshot)

    record = run_census_audit(snapshot)

    assert record["claim_id"] == "fourteen-dataset-ten-paradigm-curation"
    assert record["kind"] == "structural_audit"
    assert record["computed"]["dataset_count"] == 2
    assert record["computed"]["paradigm_count"] == 2
    assert record["computed"]["dataset_paradigms"] == {
        "MotorDataset": ["motor_imagery"],
        "SleepDataset": ["sleep_staging"],
    }
    assert record["paper_context"]["label"] == "paper_reported_not_reproduced"
    assert record["paper_context"]["dataset_count"] == 14
    assert record["paper_context"]["paradigm_count"] == 10
    assert len(record["paper_context"]["datasets"]) == 14
    assert len(record["paper_context"]["paradigms"]) == 10


def test_census_output_is_deterministic(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_fixture_snapshot(snapshot)

    assert run_census_audit(snapshot) == run_census_audit(snapshot)


def test_release_task_type_not_paper_context_drives_computed_paradigm(
    tmp_path: Path,
) -> None:
    """Catches copying the paper-context paradigm into computed evidence."""
    snapshot = tmp_path / "snapshot"
    wrapper = snapshot / "data" / "processor" / "wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        'DATASET_SELECTOR = {"adftd": AdftdBuilder}\n',
        encoding="utf-8",
    )
    dataset_dir = snapshot / "data" / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "adftd.py").write_text(
        """
class AdftdConfig:
    task_type: DatasetTaskType = DatasetTaskType.RELEASE_DEFINED_TASK

class AdftdBuilder:
    pass
""",
        encoding="utf-8",
    )
    (snapshot / "assets" / "conf").mkdir(parents=True)

    record = run_census_audit(snapshot)

    assert record["computed"]["dataset_paradigms"] == {
        "ADFTD": ["release_defined_task"]
    }
    assert record["computed"]["mapping_source"] == "released_dataset_task_type"
