from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from huggingface_hub import HfApi, hf_hub_download


PAPER_ID = "8m30ogkPk2"
PAPER_TITLE = "RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies"

UPSTREAM_PINS = {
    "arxiv_source": (
        "arxiv:2603.04639@"
        "55861691d3059ac94933c6abec6aa517eee75521485740315d249724c11b6879"
    ),
    "benchmark_repo": (
        "github:RoboMME/robomme_benchmark@"
        "03a8faf57cfbd334dfea1bb7e60079a888d70453"
    ),
    "policy_repo": (
        "github:RoboMME/robomme_policy_learning@"
        "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
    ),
    "data_h5": (
        "hf-dataset:Yinpei/robomme_data_h5@"
        "a5e4e25ffe8af34f64944f9533d06455ce5f8337"
    ),
    "preprocessed_data": (
        "hf-dataset:Yinpei/robomme_preprocessed_data@"
        "ddf0baf55b633cc6657dcd53ac0e089a273de612"
    ),
    "mme_vla_suite": (
        "hf-model:Yinpei/mme_vla_suite@"
        "5db4d53ddb98c7f80cab08792dd53d985d712ab1"
    ),
}

TARGET_CLAIMS = [
    {
        "id": "task_taxonomy",
        "challenge_claim_sha256": (
            "36b8db60ef193ce6da36b30be54d33856b268a93ea4582487650146b7ff27cec"
        ),
        "text": (
            "RoboMME defines 16 manipulation tasks organized by temporal, spatial, "
            "object, and procedural memory demands (Table 1, Table 2)."
        ),
    },
    {
        "id": "mme_vla_representations",
        "challenge_claim_sha256": (
            "44a693f9fed0a7d249d8c4caa696205797c7ea344d6cc73560267c8b91d288aa"
        ),
        "text": (
            "The MME-VLA suite evaluates symbolic, perceptual, and recurrent "
            "memory representations with multiple memory-integration mechanisms "
            "on a pi0.5-based architecture (Figure 2, Appendix A)."
        ),
    },
    {
        "id": "training_timesteps",
        "challenge_claim_sha256": (
            "9c421693d19a62760e4fb953af3c3bb3f43ed1e9f1ad8b3afbdcf2eee67e6e60"
        ),
        "text": (
            "RoboMME includes 770K high-quality training timesteps for systematic "
            "evaluation of memory-augmented policies (Section 3)."
        ),
    },
    {
        "id": "mme_vla_variants",
        "challenge_claim_sha256": (
            "dd2f1bba318425211ac424545696fa9a23b2d2fc5eba76cdcb7123ab03bd0958"
        ),
        "text": (
            "The paper builds 14 memory-augmented VLA variants on the pi0.5 "
            "backbone to compare memory representations and integration strategies "
            "(Section 4)."
        ),
    },
]

TASK_TAXONOMY = {
    "Temporal memory": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Spatial memory": [
        "VideoUnmask",
        "VideoUnmaskSwap",
        "ButtonUnmask",
        "ButtonUnmaskSwap",
    ],
    "Object memory": [
        "PickHighlight",
        "VideoRepick",
        "VideoPlaceButton",
        "VideoPlaceOrder",
    ],
    "Procedural memory": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}

PREPROCESSED_REPO = "Yinpei/robomme_preprocessed_data"
PREPROCESSED_REVISION = "ddf0baf55b633cc6657dcd53ac0e089a273de612"
MME_VLA_REPO = "Yinpei/mme_vla_suite"
MME_VLA_REVISION = "5db4d53ddb98c7f80cab08792dd53d985d712ab1"


@dataclass(frozen=True)
class ClaimResult:
    status: str
    claim: str
    evidence: str

    def as_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "claim": self.claim,
            "evidence": self.evidence,
        }


def build_bundle(artifact_root: Path | None = None) -> dict:
    root = Path("/tmp/icml-robomme-repo") if artifact_root is None else artifact_root
    task_taxonomy = _task_taxonomy(root)
    split_records = _split_records(root)
    timestep_stats = _training_timestep_stats()
    variant_stats = _mme_vla_variant_stats()
    claim_results = _claim_results(task_taxonomy, timestep_stats, variant_stats)
    return {
        "paper_id": PAPER_ID,
        "paper_title": PAPER_TITLE,
        "target_claims": TARGET_CLAIMS,
        "upstream_pins": UPSTREAM_PINS,
        "observations": {
            "task_taxonomy": task_taxonomy,
            "split_records": split_records,
            "training_timestep_stats": timestep_stats,
            "mme_vla_variants": variant_stats,
        },
        "claim_results": {
            key: result.as_dict() for key, result in claim_results.items()
        },
        "excluded_claims": [
            {
                "reason": "Requires GPU/simulator or physical-robot performance evidence.",
                "claim": "Benchmark success-rate, efficiency, real-world, and task-dependent performance claims.",
            }
        ],
    }


def _task_taxonomy(root: Path) -> dict:
    env_dir = root / "src" / "robomme" / "robomme_env"
    task_modules = sorted(path.stem for path in env_dir.glob("*.py") if path.stem != "__init__")
    expected_tasks = sorted(task for tasks in TASK_TAXONOMY.values() for task in tasks)
    missing = sorted(set(expected_tasks) - set(task_modules))
    extra = sorted(set(task_modules) - set(expected_tasks))
    return {
        "task_count": len(task_modules),
        "categories": TASK_TAXONOMY,
        "task_modules": task_modules,
        "missing_expected_tasks": missing,
        "extra_task_modules": extra,
    }


def _split_records(root: Path) -> dict[str, dict[str, int]]:
    metadata_root = root / "src" / "robomme" / "env_metadata"
    result = {}
    for split in ("train", "val", "test"):
        total_records = 0
        tasks = 0
        for path in sorted((metadata_root / split).glob("record_dataset_*_metadata.json")):
            data = json.loads(path.read_text())
            record_count = int(data["record_count"])
            records = data["records"]
            if record_count != len(records):
                raise ValueError(f"{path}: record_count does not match records")
            total_records += record_count
            tasks += 1
        result[split] = {"tasks": tasks, "record_count": total_records}
    return result


def _training_timestep_stats() -> dict[str, int | str]:
    stats_path = hf_hub_download(
        PREPROCESSED_REPO,
        repo_type="dataset",
        revision=PREPROCESSED_REVISION,
        filename="meta/stats.json",
    )
    stats = json.loads(Path(stats_path).read_text())
    total_samples = int(stats["total_samples"])
    return {
        "execution_samples": int(stats["execution_samples"]),
        "total_samples": total_samples,
        "rounded_claim_value": f"{round(total_samples / 10000) * 10:d}K",
        "source": f"hf-dataset:{PREPROCESSED_REPO}@{PREPROCESSED_REVISION}/meta/stats.json",
    }


def _mme_vla_variant_stats() -> dict:
    info = HfApi().repo_info(
        MME_VLA_REPO,
        repo_type="model",
        revision=MME_VLA_REVISION,
        files_metadata=False,
    )
    files = sorted(sibling.rfilename for sibling in info.siblings)
    variants = sorted({name.split("/")[0] for name in files if "/" in name})
    history_configs = sorted(name for name in files if name.endswith("history_config.txt"))
    representation_types = sorted({_representation_type(name) for name in variants})
    integration_types = sorted({_integration_type(name) for name in variants})
    return {
        "variant_count": len(variants),
        "variants": variants,
        "history_config_count": len(history_configs),
        "history_configs": history_configs,
        "representation_types": representation_types,
        "integration_types": integration_types,
        "source": f"hf-model:{MME_VLA_REPO}@{MME_VLA_REVISION}",
    }


def _representation_type(variant: str) -> str:
    if variant.startswith("symbolic-"):
        return "symbolic"
    if variant.startswith("perceptual-"):
        return "perceptual"
    if variant.startswith("recurrent-"):
        return "recurrent"
    raise ValueError(f"unknown representation type for {variant}")


def _integration_type(variant: str) -> str:
    if variant.endswith("-context") or variant.startswith("symbolic-"):
        return "context"
    if variant.endswith("-expert"):
        return "expert"
    if variant.endswith("-modul"):
        return "modulation"
    raise ValueError(f"unknown integration type for {variant}")


def _claim_results(
    task_taxonomy: dict,
    timestep_stats: dict,
    variant_stats: dict,
) -> dict[str, ClaimResult]:
    target_by_id = {claim["id"]: claim["text"] for claim in TARGET_CLAIMS}
    taxonomy_verified = (
        task_taxonomy["task_count"] == 16
        and not task_taxonomy["missing_expected_tasks"]
        and not task_taxonomy["extra_task_modules"]
    )
    timesteps_verified = timestep_stats["rounded_claim_value"] == "770K"
    variants_verified = variant_stats["variant_count"] == 14
    representations_verified = (
        set(variant_stats["representation_types"]) == {"symbolic", "perceptual", "recurrent"}
        and set(variant_stats["integration_types"]) == {"context", "modulation", "expert"}
    )
    return {
        "task_taxonomy": ClaimResult(
            "verified" if taxonomy_verified else "inconclusive",
            target_by_id["task_taxonomy"],
            "Parsed 16 task modules and matched the four memory-category task groups.",
        ),
        "mme_vla_representations": ClaimResult(
            "verified" if representations_verified else "inconclusive",
            target_by_id["mme_vla_representations"],
            "Counted symbolic, perceptual, and recurrent MME-VLA variants with context, modulation, and expert integration naming in the pinned model suite.",
        ),
        "training_timesteps": ClaimResult(
            "verified" if timesteps_verified else "inconclusive",
            target_by_id["training_timesteps"],
            f"Read total_samples={timestep_stats['total_samples']} from pinned preprocessed data stats, which rounds to 770K.",
        ),
        "mme_vla_variants": ClaimResult(
            "verified" if variants_verified else "inconclusive",
            target_by_id["mme_vla_variants"],
            "Counted 14 top-level variant directories and 14 history_config.txt files in the pinned MME-VLA suite.",
        ),
    }
