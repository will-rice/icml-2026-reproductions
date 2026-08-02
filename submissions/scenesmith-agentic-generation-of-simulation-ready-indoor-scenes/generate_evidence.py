from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parent
UPSTREAM = PROJECT / "upstream"
EXPECTED_COMMIT = "67cc408fd38334b4a926efef45e284302ed5055b"
EXPECTED_STAGES = [
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(path: str) -> str:
    return (UPSTREAM / path).read_text(encoding="utf-8")


def require_paths(paths: list[str]) -> list[dict[str, str]]:
    evidence = []
    missing = []
    for relative in paths:
        path = UPSTREAM / relative
        if not path.exists():
            missing.append(relative)
            continue
        evidence.append({"path": relative, "sha256": sha256_file(path)})
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return evidence


def verify_pipeline_stages() -> dict:
    source = text("scenesmith/experiments/indoor_scene_generation.py")
    config = yaml.safe_load(text("configurations/experiment/base_experiment.yaml"))
    stage_literals = {
        stage: source.count(f'"{stage}"') + source.count(f"'{stage}'")
        for stage in EXPECTED_STAGES
    }
    return {
        "configured_start_stage": config["pipeline"]["start_stage"],
        "configured_stop_stage": config["pipeline"]["stop_stage"],
        "expected_order": EXPECTED_STAGES,
        "stage_literals": stage_literals,
        "evidence_files": require_paths(
            [
                "configurations/config.yaml",
                "configurations/experiment/base_experiment.yaml",
                "scenesmith/experiments/indoor_scene_generation.py",
            ]
        ),
        "verified": all(count > 0 for count in stage_literals.values()),
    }


def verify_agent_roles() -> dict:
    agent_modules = [
        "scenesmith/floor_plan_agents/stateful_floor_plan_agent.py",
        "scenesmith/furniture_agents/stateful_furniture_agent.py",
        "scenesmith/wall_agents/stateful_wall_agent.py",
        "scenesmith/ceiling_agents/stateful_ceiling_agent.py",
        "scenesmith/manipuland_agents/stateful_manipuland_agent.py",
    ]
    role_hits = {}
    for module in agent_modules:
        body = text(module)
        role_hits[module] = {
            "designer": "_create_designer_agent" in body,
            "critic": "_create_critic_agent" in body,
            "planner_or_orchestrator": (
                "_create_planner" in body
                or "planner" in body.lower()
                or "orchestrator" in body.lower()
            ),
        }
    return {
        "role_hits": role_hits,
        "evidence_files": require_paths(agent_modules),
        "verified": all(all(hits.values()) for hits in role_hits.values()),
    }


def verify_asset_integration() -> dict:
    source_groups = {
        "text_to_3d_generation": [
            "scenesmith/agent_utils/geometry_generation_server/sam3d_pipeline_manager.py",
            "scenesmith/agent_utils/geometry_generation_server/hunyuan3d_pipeline_manager.py",
            "scenesmith/agent_utils/geometry_generation_server/geometry_generation.py",
            "scripts/install_sam3d.sh",
            "scripts/install_hunyuan3d.sh",
        ],
        "dataset_retrieval": [
            "scenesmith/agent_utils/hssd_retrieval/retrieval.py",
            "scenesmith/agent_utils/objaverse_retrieval/retrieval.py",
            "scenesmith/agent_utils/articulated_retrieval_server/retrieval.py",
            "scripts/download_hssd_data.sh",
            "scripts/download_objaverse_data.sh",
        ],
        "physical_properties_and_validation": [
            "scenesmith/agent_utils/mesh_physics_analyzer.py",
            "scenesmith/agent_utils/articulated_physics_analyzer.py",
            "scenesmith/agent_utils/physical_feasibility.py",
            "scenesmith/agent_utils/physics_validation.py",
            "scenesmith/utils/inertia_utils.py",
        ],
        "export_and_materials": [
            "scenesmith/agent_utils/sceneeval_exporter.py",
            "scripts/export_scene_to_mujoco.py",
            "scripts/download_ambientcg.py",
            "scripts/compute_ambientcg_embeddings.py",
        ],
    }
    evidence = {name: require_paths(paths) for name, paths in source_groups.items()}
    return {"source_groups": evidence, "verified": True}


def verify_robot_eval() -> dict:
    paths = [
        "scenesmith/robot_eval/task_generation/scene_prompt_generator.py",
        "scenesmith/robot_eval/policy_interface/policy_agent.py",
        "scenesmith/robot_eval/policy_interface/predicate_resolver.py",
        "scenesmith/robot_eval/success_validation/validator_agent.py",
        "scenesmith/robot_eval/tools/state_tools.py",
        "scenesmith/robot_eval/tools/vision_tools.py",
        "scenesmith/prompts/data/robot_eval/scene_prompt_generator.yaml",
        "scenesmith/prompts/data/robot_eval/policy_agent.yaml",
        "scenesmith/prompts/data/robot_eval/success_validator.yaml",
    ]
    return {"evidence_files": require_paths(paths), "verified": True}


def build_claims() -> list[dict]:
    pipeline = verify_pipeline_stages()
    roles = verify_agent_roles()
    assets = verify_asset_integration()
    robot = verify_robot_eval()
    return [
        {
            "id": "pipeline_stages",
            "claim": "SceneSmith generates simulation-ready indoor environments through hierarchical stages.",
            "status": "source_verified" if pipeline["verified"] else "unavailable",
            "evidence": pipeline,
        },
        {
            "id": "agent_roles",
            "claim": "Each generation stage is implemented with designer, critic, and orchestrator/planner roles.",
            "status": "source_verified" if roles["verified"] else "unavailable",
            "evidence": roles,
        },
        {
            "id": "asset_integration",
            "claim": "SceneSmith integrates text-to-3D synthesis, dataset retrieval, and physical-property estimation.",
            "status": "source_verified" if assets["verified"] else "unavailable",
            "evidence": assets,
        },
        {
            "id": "large_scale_metrics",
            "claim": "SceneSmith generates 3-6x more objects with under 2% collisions and 96% physics-stable objects across 210 prompts.",
            "status": "unavailable",
            "evidence": {
                "reason": "No machine-readable released 210-prompt generated-scene metrics were bundled in the pinned code snapshot.",
                "required_artifact": "210 prompt generated-scene metrics",
            },
        },
        {
            "id": "user_study",
            "claim": "SceneSmith achieves 92.2% realism and 91.5% prompt-faithfulness win rates in a 205-participant user study.",
            "status": "unavailable",
            "evidence": {
                "reason": "No released participant-level or aggregate user-study records were found in the pinned code snapshot.",
                "required_artifact": "205 participant user-study records",
            },
        },
        {
            "id": "robot_eval",
            "claim": "SceneSmith includes an end-to-end robot policy evaluation pipeline.",
            "status": "source_verified" if robot["verified"] else "unavailable",
            "evidence": robot,
        },
    ]


def build_payload() -> dict:
    upstream_files = sorted(
        path.relative_to(UPSTREAM).as_posix()
        for path in UPSTREAM.rglob("*")
        if path.is_file()
    )
    return {
        "attempt_id": "4134f5fc-75c7-4c17-8f43-3685aa8c3ac2",
        "paper_id": "WwS8CTpUA6",
        "generated_at": "deterministic-recomputed-from-bundled-upstream",
        "upstream": {
            "repository": "https://github.com/nepfaff/scenesmith",
            "commit": EXPECTED_COMMIT,
            "bundled_file_count": len(upstream_files),
            "bundled_tree_sha256": hashlib.sha256(
                json.dumps(upstream_files, separators=(",", ":"), sort_keys=True).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "public_hf_datasets": [
                "nepfaff/scenesmith-example-scenes",
                "nepfaff/scenesmith-preprocessed-data",
                "nepfaff/scenesmith-sam3d-objects",
            ],
        },
        "claims": build_claims(),
        "missing_artifacts": [
            "210 prompt generated-scene metrics",
            "205 participant user-study records",
            "paper-scale generated scene outputs with collision/stability annotations",
            "API keys, SAM3D checkpoints, ArtVIP/AmbientCG data, and >=32GB GPU runtime for full generation",
        ],
        "commands": [
            "python generate_evidence.py --output evidence/scenesmith_results.json",
            "python -m pytest tests -q",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "evidence" / "scenesmith_results.json",
    )
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
