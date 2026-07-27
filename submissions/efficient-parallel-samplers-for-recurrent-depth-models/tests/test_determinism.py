import hashlib
from pathlib import Path
import tempfile
import shutil
from recurrent_sampler_repro.evidence import run_pipeline


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def hash_directory_files(directory: Path) -> dict[str, str]:
    file_hashes = {}
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            rel_path = str(file_path.relative_to(directory))
            file_hashes[rel_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return file_hashes


def test_evidence_generation_determinism():
    project_root = get_project_root()

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        dir1 = Path(tmp1)
        dir2 = Path(tmp2)

        # Copy vendor files to temp directories
        shutil.copytree(project_root / "vendor", dir1 / "vendor")
        shutil.copytree(project_root / "vendor", dir2 / "vendor")

        # Run pipeline on both
        run_pipeline(dir1)
        run_pipeline(dir2)

        # Compare outputs byte for byte
        hashes1 = hash_directory_files(dir1)
        hashes2 = hash_directory_files(dir2)

        assert hashes1 == hashes2, "Evidence generation outputs are not byte-for-byte deterministic!"
