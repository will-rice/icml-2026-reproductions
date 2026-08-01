from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


PAPER_ID = "7MlfE2Da2W"
ATTEMPT_ID = "092b120a-9cf1-4351-b4fc-fcbf51f6565f"
SNAPSHOT_ID = "00263d6d9b331596d4be77a4cd17a4b1b6592f2ac7a72401cd62b751eaaef9bb"
UPSTREAM_REPO = "https://github.com/snumprlab/scale.git"
UPSTREAM_LABEL = "snumprlab/scale"
UPSTREAM_COMMIT = "b4ad2a69d14f91712704711e810cf9830e2b7121"
EVIDENCE_GENERATED_AT = "2026-08-01T12:32:47+00:00"

MODEL_PATH = Path("prismatic/extern/hf/modeling_prismatic.py")
CONFIG_PATH = Path("configs/scale.yaml")
RUNNER_PATH = Path("experiments/robot/libero/run_libero_eval.py")
README_PATH = Path("README.md")
LICENSE_PATH = Path("LICENSE")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text(root: Path, relative: Path) -> str:
    return (root / relative).read_text(encoding="utf-8")


def file_record(root: Path, relative: Path) -> dict[str, Any]:
    data = (root / relative).read_bytes()
    return {
        "path": relative.as_posix(),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }


def run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def ensure_upstream_source(source_root: str | None = None) -> Path:
    configured = source_root or os.environ.get("SCALE_SOURCE_ROOT")
    if configured:
        root = Path(configured).resolve()
        commit = run_git(["rev-parse", "HEAD"], root)
        if commit != UPSTREAM_COMMIT:
            raise ValueError(f"source root is at {commit}, expected {UPSTREAM_COMMIT}")
        return root

    cache_root = Path(os.environ.get("SCALE_UPSTREAM_CACHE", "/tmp/scale-upstream-cache"))
    root = cache_root / UPSTREAM_COMMIT
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", "--filter=blob:none", UPSTREAM_REPO, str(root)])
    run_git(["checkout", "--detach", UPSTREAM_COMMIT], root)
    commit = run_git(["rev-parse", "HEAD"], root)
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"cached checkout is at {commit}, expected {UPSTREAM_COMMIT}")
    return root


def snippet(text: str, marker: str, radius: int = 180) -> str:
    index = text.find(marker)
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(text), index + len(marker) + radius)
    return " ".join(text[start:end].split())


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def collect_observations(root: Path) -> dict[str, Any]:
    model = read_text(root, MODEL_PATH)
    config_text = read_text(root, CONFIG_PATH)
    runner = read_text(root, RUNNER_PATH)
    readme = read_text(root, README_PATH)
    config_values = yaml.safe_load(config_text)

    files = [
        file_record(root, README_PATH),
        file_record(root, LICENSE_PATH),
        file_record(root, CONFIG_PATH),
        file_record(root, RUNNER_PATH),
        file_record(root, MODEL_PATH),
    ]

    source_indicators = {
        "self_uncertainty": {
            "present": contains_all(
                model,
                [
                    "def _compute_self_uncertainty",
                    "q_low",
                    "q_high",
                    "torch.log(q_high / q_low)",
                ],
            ),
            "path": MODEL_PATH.as_posix(),
            "snippet": snippet(model, "def _compute_self_uncertainty"),
        },
        "action_temperature": {
            "present": "tau_k = T0 * torch.sigmoid" in model
            and "torch.softmax(top_logits / tau_k" in model,
            "path": MODEL_PATH.as_posix(),
            "snippet": snippet(model, "tau_k = T0 * torch.sigmoid"),
        },
        "visual_attention_temperature": {
            "present": contains_all(
                model,
                [
                    "def apply_visual_attention_temperature",
                    "gamma = kappa ** np.tanh",
                    "block.attn.scale = original_scale / max(temperature, 1e-8)",
                ],
            ),
            "path": MODEL_PATH.as_posix(),
            "snippet": snippet(model, "def apply_visual_attention_temperature"),
        },
        "scale_config": {
            "present": config_values == {
                "T0": 1.0,
                "epsilon": 1.0e-12,
                "num_logits": 256,
                "kappa": 2.0,
                "alpha": 0.8,
                "attn_sensitivity": 0.3,
            },
            "path": CONFIG_PATH.as_posix(),
            "values": config_values,
            "sha256": file_record(root, CONFIG_PATH)["sha256"],
        },
        "decoding_modes": {
            "present": 'VALID_DECODING_MODES = ("greedy", "temp", "topk", "topp", "scale")' in runner,
            "path": RUNNER_PATH.as_posix(),
            "snippet": snippet(runner, "VALID_DECODING_MODES"),
        },
    }

    no_training_indicators = {
        "readme_claims_no_training_no_verifier": contains_all(
            readme.lower(),
            ["no additional training", "no verifier", "single forward pass"],
        ),
        "inference_uses_no_grad": "with torch.no_grad()" in model,
        "runner_loads_pretrained_checkpoint": "openvla/openvla-7b-finetuned" in runner,
        "training_entrypoint_absent": "def train(" not in model and "Trainer(" not in model,
        "verifier_module_absent": "verifier" not in "\n".join(
            path.as_posix().lower() for path in root.rglob("*") if path.is_file()
        ),
    }

    result_artifacts = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".jsonl", ".parquet", ".feather"}
    ]

    return {
        "source_tree": {
            "repo": UPSTREAM_REPO,
            "commit": run_git(["rev-parse", "HEAD"], root),
            "tracked_file_count": int(run_git(["ls-files"], root).count("\n") + 1),
        },
        "files": files,
        "source_indicators": source_indicators,
        "no_training_no_verifier_indicators": no_training_indicators,
        "runtime_constraints": {
            "cpu_only_reproduction": True,
            "gpu_robot_eval_required": True,
            "reason": "LIBERO, SIMPLER-WidowX, and real-world robot evaluations require GPU simulation or physical robot execution.",
        },
        "raw_result_artifacts": result_artifacts,
    }


def build_claim_results(observations: dict[str, Any]) -> dict[str, Any]:
    indicators = observations["source_indicators"]
    mechanism_verified = all(
        indicators[name]["present"]
        for name in (
            "self_uncertainty",
            "action_temperature",
            "visual_attention_temperature",
            "scale_config",
            "decoding_modes",
        )
    )
    no_training = observations["no_training_no_verifier_indicators"]
    no_training_supported = all(no_training.values())
    raw_results_available = bool(observations["raw_result_artifacts"])

    return {
        "claim-1": {
            "challenge_claim_sha256": "9c25ef590bdbf95cd8dfa64cbaf7ce7093649e4b304868d51d028bf9eedd135d",
            "status": "verified" if mechanism_verified else "inconclusive",
            "summary": "Pinned source implements self-uncertainty, action temperature, visual attention temperature, and SCALE decoding mode.",
            "evidence": [
                indicators["self_uncertainty"]["path"],
                indicators["action_temperature"]["path"],
                indicators["visual_attention_temperature"]["path"],
                indicators["scale_config"]["path"],
                indicators["decoding_modes"]["path"],
            ],
        },
        "claim-2": {
            "challenge_claim_sha256": "cbc5ac391b72f156b9229ebbd6fd474bcd84acb791c85b3e7d93fc537345c2f7",
            "status": "toy" if no_training_supported else "inconclusive",
            "summary": "source-level checks support inference-only execution with no verifier, but this CPU audit does not time a deployed robot control step.",
            "evidence": no_training,
        },
        "claim-3": {
            "challenge_claim_sha256": "e946bc551380587b07c7a39952aef70e9539dd26a46b5df3dc46d45963e348db",
            "status": "unavailable" if not raw_results_available else "inconclusive",
            "summary": "OpenVLA/LIBERO benchmark success-rate comparisons were not rerun and no machine-readable raw result artifact was found in the pinned repo.",
        },
        "claim-4": {
            "challenge_claim_sha256": "87c4598a405ffcb991b98ba21002ffadf3a7264de33d976ac9115ad0fa188b85",
            "status": "unavailable" if not raw_results_available else "inconclusive",
            "summary": "pi0-FAST/LIBERO average success rates were not rerun and are not backed by a machine-readable raw result artifact in the pinned repo.",
        },
        "claim-5": {
            "challenge_claim_sha256": "24299b74015bef4a7cda614376d95a087770cb7778111a19908e23f2760cbb28",
            "status": "unavailable" if not raw_results_available else "inconclusive",
            "summary": "SIMPLER-WidowX pi0-FAST and SpatialVLA claims require non-CPU benchmark reruns or raw logs absent from the pinned repo.",
        },
        "claim-6": {
            "challenge_claim_sha256": "08067e64cb7b51dab32b1853168b874e9914dd2231f273310e2799d8745c81e3",
            "status": "unavailable" if not raw_results_available else "inconclusive",
            "summary": "Real-world pick-and-place success rates require physical robot evaluations; no reproduced measurements are available here.",
        },
    }


def build_evidence(source_root: str | None = None) -> dict[str, Any]:
    root = ensure_upstream_source(source_root)
    observations = collect_observations(root)
    return {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "generated_at": EVIDENCE_GENERATED_AT,
        "upstream": {
            "github": f"{UPSTREAM_LABEL}@{UPSTREAM_COMMIT}",
            "url": UPSTREAM_REPO,
            "code_license": "MIT",
            "paper": "arxiv:2602.04208",
            "project_page": "https://dcahn12.github.io/projects/scale/",
        },
        "observations": observations,
        "claim_results": build_claim_results(observations),
        "unreplicated": [
            "LIBERO benchmark success rates were not rerun because they require GPU robot simulation and pretrained VLA checkpoints.",
            "SIMPLER-WidowX benchmark claims were not rerun because the pinned repository does not include a CPU-rerunnable evaluation path or raw result artifacts.",
            "real-world pick-and-place success rates were not rerun because they require physical robot hardware and task resets.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output", default="evidence/bundle.json")
    args = parser.parse_args(argv)

    bundle = build_evidence(args.source_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "paper_id": PAPER_ID, "claims": bundle["claim_results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
