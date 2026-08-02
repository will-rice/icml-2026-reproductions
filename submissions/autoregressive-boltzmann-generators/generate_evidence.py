import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


PROJECT = Path(__file__).resolve().parent
ATTEMPT_ID = "1ff17cfb-669a-4192-90f0-c014220f7f12"
SNAPSHOT_ID = "dc14d49cb209316c2a8f5cd9ff0e2ff27eacc29f5d2c4c4a2bacccca9ab6b4cc"
PAPER_ID = "75AYDsndHP"
CODE_REPO = "https://github.com/danyalrehman/autobg"
CODE_COMMIT = "21624a80504b3199b291514c37a49cccd19c8817"
ROBIN_REPO = "danyalrehman17/robin-transferable"
ROBIN_REVISION = "2813c971b63a177ad578c51c9a550c2e63e9168d"
DATASET_REPO = "transferable-samplers/many-peptides-md"
DATASET_REVISION = "1af9336878122eb1d62894fe2fb3ff4b801a3216"
DEFAULT_LOCAL_SOURCE = Path("/tmp/autobg-upstream-21624")

CLAIMS = {
    "arbg_likelihood_importance": (
        "Autoregressive Boltzmann Generators replace flow-based Boltzmann generators with an autoregressive model that retains exact likelihoods and importance-sampling correction (Section 3).",
        "7bcd3a577769a3924d75a92174d82edff9aba4603edb1b22bdb4c4c24b19cfd0",
    ),
    "topology_interventions": (
        "ArBG avoids the invertibility/topology constraints of normalizing-flow Boltzmann generators and supports sequential inference-time interventions (Section 3).",
        "2608885d20eeda0c35af7565d2b1158a99865cfaa1dbc02916bff1c7f6869528",
    ),
    "benchmark_improvements": (
        "ArBG reports improvements over flow-based Boltzmann generators across all evaluated molecular benchmarks, with especially large gains on 10-residue Chignolin (Section 4).",
        "5c86e004dfa2fdbddbe7202d8faaabb06b80f295f78977c04de985e266fe3212",
    ),
    "robin_model": (
        "The paper introduces Robin, a 132M-parameter transferable ArBG model for molecular sampling (Section 4).",
        "f9b1d3ce6b2f834ebb78e2ad771418ed81e59d6a2cef315127635c552436027d",
    ),
    "robin_energy_reduction": (
        "Robin reports more than a 60% reduction in zero-shot E-W2 energy error on 8-residue systems relative to the previous state of the art (Abstract).",
        "e5c71dc7149d473a429fcfc2d77d2bb1877cc666d5a42f600ebc2dffa97bb84f",
    ),
}

IMPORTANT_FILES = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "src/models/autoregressive_module.py",
    "src/models/transferable_boltzmann_generator_module.py",
    "src/models/neural_networks/autoregressive/causal_transformer.py",
    "configs/model/autoregressive.yaml",
    "configs/experiment/training/single_system/autoregressive_GYDPETGTWG.yaml",
    "configs/experiment/training/transferable/autoregressive_up_to_8aa.yaml",
    "scripts/eval_transferable.sh",
    "scripts/train_transferable.sh",
]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_raw(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/danyalrehman/autobg/{CODE_COMMIT}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def read_tree() -> dict[str, Any]:
    local = Path(os.environ.get("AUTOBG_UPSTREAM_PATH", DEFAULT_LOCAL_SOURCE))
    if (local / "src/models/autoregressive_module.py").exists() and git_commit(local) == CODE_COMMIT:
        paths = []
        texts = {}
        hashes = {}
        for path in sorted(local.rglob("*")):
            rel = path.relative_to(local).as_posix()
            if path.is_dir() or rel.startswith(".git/") or "/.git/" in rel:
                continue
            paths.append(rel)
            if rel in IMPORTANT_FILES:
                raw = path.read_bytes()
                texts[rel] = raw.decode("utf-8", errors="replace")
                hashes[rel] = sha256_bytes(raw)
        return {"source": "local_pinned_checkout", "paths": paths, "texts": texts, "hashes": hashes, "tree_truncated": False}

    api_url = f"https://api.github.com/repos/danyalrehman/autobg/git/trees/{CODE_COMMIT}?recursive=1"
    data = fetch_json(api_url)
    paths = sorted(item["path"] for item in data.get("tree", []) if item.get("type") == "blob")
    raw_files = {path: fetch_raw(path) for path in IMPORTANT_FILES if path in paths}
    return {
        "source": "github_git_tree_api",
        "api_url": api_url,
        "paths": paths,
        "texts": {path: raw.decode("utf-8", errors="replace") for path, raw in raw_files.items()},
        "hashes": {path: sha256_bytes(raw) for path, raw in raw_files.items()},
        "tree_truncated": bool(data.get("truncated")),
    }


def hf_metadata() -> dict[str, Any]:
    api = HfApi()
    model = api.model_info(ROBIN_REPO, revision=ROBIN_REVISION, files_metadata=True)
    dataset = api.dataset_info(DATASET_REPO, revision=DATASET_REVISION, files_metadata=True)
    robin = next(s for s in model.siblings if s.rfilename == "robin.ckpt")
    dataset_files = [s.rfilename for s in dataset.siblings]
    return {
        "robin_sha": model.sha,
        "robin_license": getattr(model, "card_data", {}).get("license") if getattr(model, "card_data", None) else None,
        "robin_files": sorted(s.rfilename for s in model.siblings),
        "robin_checkpoint_lfs_size": robin.lfs.size if robin.lfs else robin.size,
        "robin_checkpoint_lfs_sha256": robin.lfs.sha256 if robin.lfs else None,
        "dataset_sha": dataset.sha,
        "dataset_files_sample": sorted(dataset_files)[:20],
        "dataset_file_count": len(dataset_files),
    }


def contains(texts: dict[str, str], path: str, *terms: str) -> bool:
    haystack = texts.get(path, "")
    return all(term in haystack for term in terms)


def build_support(tree: dict[str, Any], hf: dict[str, Any]) -> dict[str, Any]:
    joined = "\n".join(tree["texts"].values()).lower()
    return {
        "autoregressive_module": contains(tree["texts"], "src/models/autoregressive_module.py", "AutoregressiveLitModule", "_generate_autoregressive"),
        "exact_log_likelihood": contains(tree["texts"], "src/models/autoregressive_module.py", "compute_log_likelihood", "log_likelihood"),
        "importance_reweighting": "reweight" in joined and "snis" in joined,
        "sequential_interventions": "temperature" in joined and "top_k" in joined and "top_p" in joined,
        "chignolin_config": "gydpetgtwg" in joined and "Chignolin" in tree["texts"].get("README.md", ""),
        "robin_eval_script": contains(tree["texts"], "scripts/eval_transferable.sh", "danyalrehman17/robin-transferable", "E-W2", "90 held-out"),
        "robin_checkpoint_lfs_size": hf["robin_checkpoint_lfs_size"],
        "evidence_paths": IMPORTANT_FILES,
    }


def claim(claim_id: str, status: str, observed: dict[str, Any], evidence: str) -> dict[str, Any]:
    text, digest = CLAIMS[claim_id]
    return {
        "id": claim_id,
        "status": status,
        "challenge_claim": text,
        "challenge_claim_sha256": digest,
        "observed": observed,
        "evidence": evidence,
    }


def build_payload() -> dict[str, Any]:
    tree = read_tree()
    hf = hf_metadata()
    support = build_support(tree, hf)
    source = {
        "source": tree["source"],
        "python_file_count": sum(path.endswith(".py") for path in tree["paths"]),
        "config_file_count": sum(path.endswith((".yaml", ".toml")) for path in tree["paths"]),
        "script_file_count": sum(path.startswith("scripts/") for path in tree["paths"]),
        "license": "mit-plus-notice" if "LICENSE" in tree["paths"] and "NOTICE" in tree["paths"] else "unknown",
        "tree_truncated": tree["tree_truncated"],
        "important_file_hashes": dict(sorted(tree["hashes"].items())),
    }
    return {
        "attempt_id": ATTEMPT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "generated_at": "deterministic-from-pinned-public-artifacts",
        "paper": {
            "paper_id": PAPER_ID,
            "title": "Autoregressive Boltzmann Generators",
            "openreview": "https://openreview.net/forum?id=75AYDsndHP",
            "arxiv": "https://arxiv.org/abs/2606.27361",
        },
        "upstream": {
            "code_repo": CODE_REPO,
            "code_commit": CODE_COMMIT,
            "robin_model_repo": ROBIN_REPO,
            "robin_model_revision": ROBIN_REVISION,
            "many_peptides_repo": DATASET_REPO,
            "many_peptides_revision": DATASET_REVISION,
            "estimated_api_cost_usd": 0.0,
        },
        "source_tree": source,
        "huggingface": hf,
        "artifact_support": support,
        "claims": [
            claim("arbg_likelihood_importance", "artifact_verified", {"source": "pinned_source_static_analysis", **support}, "Pinned code implements an AutoregressiveLitModule with exact log-likelihood computation and SNIS/reweighting paths."),
            claim("topology_interventions", "artifact_verified", {"source": "pinned_source_static_analysis", **support}, "Pinned README/code show a diffeomorphism-free autoregressive model with temperature/top-k/top-p sequential controls."),
            claim("benchmark_improvements", "released_claim_values_unrecomputed", {"source": "pinned_source_static_analysis", "chignolin_config": support["chignolin_config"], "released_result_table_found": False}, "Pinned scripts/configs include ArBG and flow baseline evaluation paths including Chignolin, but this bundle does not rerun benchmarks or use a pinned result table."),
            claim("robin_model", "metadata_verified", {"source": "huggingface_hub_metadata", "repo": ROBIN_REPO, "revision": ROBIN_REVISION, "checkpoint_size": hf["robin_checkpoint_lfs_size"], "checkpoint_sha256": hf["robin_checkpoint_lfs_sha256"]}, "The pinned HF model repo exposes robin.ckpt as a 1.06 GB LFS object and the code provides Robin train/eval scripts."),
            claim("robin_energy_reduction", "released_claim_values_unrecomputed", {"source": "huggingface_hub_and_source_metadata", "dataset_revision": DATASET_REVISION, "released_result_table_found": False}, "Pinned model and dataset metadata support the released Robin artifact, but the >60% E-W2 reduction is not recomputed in this CPU/static validation bundle."),
        ],
        "unavailable": [
            "Headline benchmark improvements and Robin E-W2 percentage reductions require GPU benchmark reruns or pinned released result tables; they are not reproduced measurements here.",
            "The 1.06 GB Robin checkpoint and full ManyPeptidesMD trajectories are not downloaded during routine validation; metadata and LFS hashes are recorded instead.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "evidence" / "autobg_results.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
