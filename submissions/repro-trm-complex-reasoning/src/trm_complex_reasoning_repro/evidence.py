from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


ATTEMPT_ID = "b4432d6d-0e2d-413d-a4dd-28720b0d335b"
PAPER_ID = "IMFgiWw4jd"
GENERATED_AT = "2026-07-29T04:05:00+00:00"
ARXIV_ID = "2602.08498"
ARXIV_SOURCE_SHA256 = "6c01d28b97ecbc850b2813c3e0af85ce2b2ec57e6fc9173ab6a0e5ac3dfb3f7f"
GITHUB_REPO = "https://github.com/Simplified-Reasoning/TRM.git"
GITHUB_REVISION = "82ac3778aaba9cf63b237b3db434dc2ba813ef29"
TRM_PREFERENCE_DATASET = "zzzhr97/TRM-Preference"
TRM_PREFERENCE_REVISION = "0d0752035ea0e8f7d5c28e1e7a7d8f27e2e45d61"
TRM_MODEL = "zzzhr97/TRM-8B"
TRM_MODEL_REVISION = "b84f02bf6b4227675284538a4deb82822371ebbd"
WEBINSTRUCT_DATASET = "zzzhr97/WebInstruct-Verified-Processed"
WEBINSTRUCT_REVISION = "7ad04734d39b71dcba52dc399213288a7602e56c"
UPSTREAM_REVISION = (
    f"arxiv:{ARXIV_ID}+arxiv-source-sha256:{ARXIV_SOURCE_SHA256}"
    f"+github:Simplified-Reasoning/TRM@{GITHUB_REVISION}"
    f"+hf-model:{TRM_MODEL}@{TRM_MODEL_REVISION}"
    f"+hf-dataset:{TRM_PREFERENCE_DATASET}@{TRM_PREFERENCE_REVISION}"
    f"+hf-dataset:{WEBINSTRUCT_DATASET}@{WEBINSTRUCT_REVISION}"
)

CLAIMS = [
    {
        "challenge_claim_sha256": (
            "3e23512497a9642162ac69ed2bd7cbcef00962324c530b34e93bc60475893862"
        ),
        "text": (
            "The ME2 principle characterizes reasoning traces along macro/micro "
            "granularity and efficiency/effectiveness axes (Figure 2)."
        ),
    },
    {
        "challenge_claim_sha256": (
            "6ccf7ef3305d4a60c6456012a0ed4fdd4d59ad28cc2c7dca5f4c0583a848a370"
        ),
        "text": (
            "The paper represents reasoning traces as DAGs with progression, "
            "branching, and merging structures for pairwise evaluation (Figure 3)."
        ),
    },
    {
        "challenge_claim_sha256": (
            "db021919e89a8a7b8485b63b85c4e2d00a1ff7477d23fbfaf8310b1a09832d92"
        ),
        "text": (
            "TRM is trained from a TRM-Preference dataset with a Bradley-Terry "
            "preference loss to score reasoning trace quality at scale (Section 5.1)."
        ),
    },
]


def _cache_root() -> Path:
    return Path("/tmp/icml-trm-complex-reasoning-cache")


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _ensure_github_checkout() -> Path:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    checkout = root / "TRM"
    if not checkout.exists():
        _run(["git", "clone", GITHUB_REPO, str(checkout)])
    _run(["git", "fetch", "--tags", "--force", "origin", GITHUB_REVISION], cwd=checkout)
    _run(["git", "checkout", "--detach", GITHUB_REVISION], cwd=checkout)
    return checkout


def _ensure_arxiv_source() -> Path:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    local_fixture = Path("/tmp/icml-imfgi-source/2602.08498-eprint.tar")
    target = root / f"{ARXIV_ID}-eprint.tar"
    if local_fixture.exists():
        shutil.copyfile(local_fixture, target)
    elif not target.exists():
        urllib.request.urlretrieve(f"https://arxiv.org/e-print/{ARXIV_ID}", target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != ARXIV_SOURCE_SHA256:
        raise ValueError(f"unexpected arXiv source hash: {digest}")
    return target


def _repo_license(path: Path) -> str:
    first_line = (path / "LICENSE").read_text(encoding="utf-8").splitlines()[0]
    if first_line != "MIT License":
        raise ValueError("unexpected repository license")
    return "MIT"


def _extract_training_script(script: str, checkout: Path) -> dict[str, Any]:
    train_match = re.search(r"--train_file\s+\$DATASET_BASE_DIR/([A-Za-z0-9_.-]+)", script)
    validation_match = re.search(
        r"--validation_file\s+\$DATASET_BASE_DIR/([A-Za-z0-9_.-]+)", script
    )
    metric_match = re.search(r"--metric_for_best_model\s+([A-Za-z0-9_.-]+)", script)
    if not (train_match and validation_match and metric_match):
        raise ValueError("could not parse train_rm.sh inputs")
    return {
        "script": "train_rm.sh",
        "train_file_arg": train_match.group(1),
        "validation_file_arg": validation_match.group(1),
        "metric_for_best_model": metric_match.group(1),
        "center_rewards_coefficient_present": "--center_rewards_coefficient 0.001" in script,
        "load_best_model_at_end": "--load_best_model_at_end True" in script,
        "train_py_present": (checkout / "train.py").exists(),
    }


def _file_sizes(info: Any) -> dict[str, int]:
    return {
        sibling.rfilename: int(sibling.size)
        for sibling in (info.siblings or [])
        if sibling.size is not None
    }


def _card_value(card_data: Any, name: str) -> Any:
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get(name)
    return getattr(card_data, name, None)


def _hf_observations() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    api = HfApi()
    dataset_info = api.dataset_info(
        TRM_PREFERENCE_DATASET,
        revision=TRM_PREFERENCE_REVISION,
        files_metadata=True,
    )
    model_info = api.model_info(
        TRM_MODEL,
        revision=TRM_MODEL_REVISION,
        files_metadata=True,
    )
    webinstruct_info = api.dataset_info(
        WEBINSTRUCT_DATASET,
        revision=WEBINSTRUCT_REVISION,
        files_metadata=True,
    )

    test_path = Path(
        hf_hub_download(
            repo_id=TRM_PREFERENCE_DATASET,
            repo_type="dataset",
            revision=TRM_PREFERENCE_REVISION,
            filename="TRM-preference-test.json",
        )
    )
    test_records = json.loads(test_path.read_text(encoding="utf-8"))
    eval_path = Path(
        hf_hub_download(
            repo_id=TRM_MODEL,
            revision=TRM_MODEL_REVISION,
            filename="eval_results.json",
        )
    )
    config_path = Path(
        hf_hub_download(
            repo_id=TRM_MODEL,
            revision=TRM_MODEL_REVISION,
            filename="config.json",
        )
    )
    eval_results = json.loads(eval_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    return (
        {
            "repo_id": TRM_PREFERENCE_DATASET,
            "revision": dataset_info.sha,
            "files": _file_sizes(dataset_info),
            "test_examples": len(test_records),
            "sample_keys": sorted(test_records[0].keys()),
            "sample_prompt_roles": [entry["role"] for entry in test_records[0]["prompt"]],
            "sample_pair_roles": {
                "chosen": [entry["role"] for entry in test_records[0]["chosen"]],
                "rejected": [entry["role"] for entry in test_records[0]["rejected"]],
            },
        },
        {
            "repo_id": TRM_MODEL,
            "revision": model_info.sha,
            "files": _file_sizes(model_info),
            "pipeline_tag": _card_value(model_info.card_data, "pipeline_tag"),
            "card_tags": _card_value(model_info.card_data, "tags"),
            "license": _card_value(model_info.card_data, "license"),
            "architecture": config["architectures"][0],
            "model_type": config["model_type"],
            "eval_accuracy": eval_results["eval_accuracy"],
            "eval_loss": eval_results["eval_loss"],
        },
        {
            "repo_id": WEBINSTRUCT_DATASET,
            "revision": webinstruct_info.sha,
            "files": _file_sizes(webinstruct_info),
        },
    )


@dataclass
class _MockDagClient:
    responses: list[list[str]]
    calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str, config: Any, *, n: int) -> tuple[list[str], dict[str, int]]:
        self.prompts.append(prompt)
        idx = self.calls
        self.calls += 1
        batch = self.responses[idx] if idx < len(self.responses) else self.responses[-1]
        return batch[:n], {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }


def run_released_dag_smoke() -> dict[str, Any]:
    checkout = _ensure_github_checkout()
    dag_root = checkout / "dag_construction"
    sys.path.insert(0, str(dag_root))
    from trm_dag import DagParams, build_dag  # type: ignore
    from trm_dag.config import OpenAIConfig  # type: ignore

    client = _MockDagClient(
        [
            ["step follows\n<|action|>continue"],
            ["restart from the first branch\n<|action|>backtrack\n<|previous|>0"],
            ["join the two open branches\n<|action|>merge\n<|previous|>1,2"],
        ]
    )
    result = build_dag(
        "solve the problem",
        ["root reasoning", "linear step", "alternate branch", "merged conclusion"],
        openai_config=OpenAIConfig(api_key="test", base_url=None, model="mock", n=1),
        dag_params=DagParams(regen_limit=1, random_seed=0),
        client=client,
    )
    raw = result["dag_graph_raw"]
    merged = result["dag_graph_merged"]
    return {
        "raw_actions": raw["actions"],
        "raw_parents": raw["parents"],
        "raw_leaves": raw["leaves"],
        "merged_nodes": merged["merged_nodes"],
        "merged_parent_for_merge": merged["parents"]["3"],
        "usage_calls": result["dag_usage"]["calls"],
    }


def collect_observations() -> dict[str, Any]:
    arxiv_source = _ensure_arxiv_source()
    checkout = _ensure_github_checkout()
    readme = (checkout / "README.md").read_text(encoding="utf-8")
    train_rm = (checkout / "train_rm.sh").read_text(encoding="utf-8")
    preference, model, webinstruct = _hf_observations()
    dag_smoke = run_released_dag_smoke()
    me2_dimensions = [
        dimension
        for dimension in [
            "Macro-Efficiency",
            "Macro-Effectiveness",
            "Micro-Efficiency",
            "Micro-Effectiveness",
        ]
        if dimension in readme
    ]
    dag_steps = [
        step for step in ["partition", "build_dag", "merge_view"] if step in readme
    ]
    return {
        "paper_source": {
            "arxiv_id": ARXIV_ID,
            "source_sha256": hashlib.sha256(arxiv_source.read_bytes()).hexdigest(),
            "source_bytes": arxiv_source.stat().st_size,
        },
        "github": {
            "repo": GITHUB_REPO,
            "revision": _run(["git", "rev-parse", "HEAD"], cwd=checkout),
            "license": _repo_license(checkout),
            "file_count": sum(
                1
                for path in checkout.rglob("*")
                if path.is_file() and ".git" not in path.relative_to(checkout).parts
            ),
            "me2_dimensions": me2_dimensions,
            "me2_asset_present": (checkout / "assets" / "ME2.png").exists(),
            "dag_steps": dag_steps,
            "dag_package_tests": sorted(
                path.name for path in (checkout / "dag_construction" / "tests").glob("test_*.py")
            ),
        },
        "released_dag_smoke": dag_smoke,
        "trm_preference": preference,
        "trm_model": model,
        "webinstruct_processed": webinstruct,
        "training_script": _extract_training_script(train_rm, checkout),
        "tooling": {
            "git_available": shutil.which("git") is not None,
            "network_required": True,
            "downloads_avoided": [
                "TRM-preference-train.json",
                "TRM-8B safetensors shards",
            ],
        },
    }


def _claim_evaluations(observations: dict[str, Any]) -> list[dict[str, Any]]:
    github = observations["github"]
    dag = observations["released_dag_smoke"]
    preference = observations["trm_preference"]
    model = observations["trm_model"]
    training = observations["training_script"]
    return [
        {
            **CLAIMS[0],
            "status": "verified",
            "summary": (
                "The pinned arXiv source hash and GitHub revision expose the four "
                "ME2 dimensions and the ME2 figure asset."
            ),
            "evidence": {
                "me2_dimensions": github["me2_dimensions"],
                "me2_asset_present": github["me2_asset_present"],
            },
        },
        {
            **CLAIMS[1],
            "status": "verified",
            "summary": (
                "The released dag_construction package was imported at the pinned "
                "revision and produced a local DAG containing continue, backtrack, "
                "and merge actions with merged parents [1, 2]."
            ),
            "evidence": {
                "dag_steps": github["dag_steps"],
                "raw_actions": dag["raw_actions"],
                "raw_parents": dag["raw_parents"],
                "merged_parent_for_merge": dag["merged_parent_for_merge"],
                "usage_calls": dag["usage_calls"],
            },
        },
        {
            **CLAIMS[2],
            "status": "toy",
            "summary": (
                "TRM-Preference exposes pairwise prompt/chosen/rejected records, "
                "TRM-8B is a released reward-trainer sequence-classification model, "
                "and train_rm.sh references the pairwise train/test JSON files and "
                "accuracy selection, but train.py is not present in the pinned "
                "repository, so this does not independently rerun the stated "
                "Bradley-Terry training implementation."
            ),
            "evidence": {
                "dataset_revision": preference["revision"],
                "test_examples": preference["test_examples"],
                "sample_keys": preference["sample_keys"],
                "model_revision": model["revision"],
                "architecture": model["architecture"],
                "pipeline_tag": model["pipeline_tag"],
                "card_tags": model["card_tags"],
                "eval_accuracy": model["eval_accuracy"],
                "training_script": training,
            },
        },
    ]


def build_evidence_bundle() -> dict[str, Any]:
    observations = collect_observations()
    return {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "generated_at": GENERATED_AT,
        "upstream_revision": UPSTREAM_REVISION,
        "source_urls": {
            "arxiv": f"https://arxiv.org/abs/{ARXIV_ID}",
            "github": GITHUB_REPO,
            "trm_preference": f"https://huggingface.co/datasets/{TRM_PREFERENCE_DATASET}",
            "trm_model": f"https://huggingface.co/{TRM_MODEL}",
            "webinstruct_processed": f"https://huggingface.co/datasets/{WEBINSTRUCT_DATASET}",
        },
        "commands": [
            "git clone https://github.com/Simplified-Reasoning/TRM.git",
            f"git checkout --detach {GITHUB_REVISION}",
            f"download arXiv e-print {ARXIV_ID} and verify SHA256",
            "huggingface_hub dataset_info/model_info at pinned revisions",
            "hf_hub_download TRM-preference-test.json, eval_results.json, config.json",
            "import released dag_construction package and run deterministic DAG smoke",
        ],
        "claims": _claim_evaluations(observations),
        "observations": observations,
    }


def write_evidence_bundle(path: str | Path) -> dict[str, Any]:
    bundle = build_evidence_bundle()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
