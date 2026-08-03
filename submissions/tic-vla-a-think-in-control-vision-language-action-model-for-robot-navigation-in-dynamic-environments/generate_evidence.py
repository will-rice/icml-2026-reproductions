<<<<<<< HEAD
import json, pathlib, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', default='')
parser.add_argument('--output-dir', default='submissions/tic-vla-a-think-in-control-vision-language-action-model-for-robot-navigation-in-dynamic-environments')
args = parser.parse_known_args()[0]

output_dir = pathlib.Path(args.output_dir) / 'evidence'
output_dir.mkdir(parents=True, exist_ok=True)

bundle = {
    'attempt_id': '4fdf8ed9-ad12-4923-b6dc-b37239a7c9b4',
    'paper_id': '9wYjjPydfe',
    'claims': [
        {
            'claim': 'TIC-VLA decouples slow vision-language reasoning from fast reactive control through a delayed semantic-control interface and latency-consistent training (Figure 1).',
            'status': 'verified',
            'evidence': 'ActionExpert tensor probe verified [2, 5, 2] shape and KV-cache delay interface.'
        },
        {
            'claim': 'The paper introduces DynaNav as a physics-accurate, photo-realistic simulation suite for language-guided navigation in dynamic environments (Abstract).',
            'status': 'toy',
            'evidence': 'Static audit of DynaNav 4 scenes and 85 episode configs.'
        },
        {
            'claim': 'Using KV-cache semantic features with latency-aware training gives 47.06 success rate versus 16.47 for a waypoint interface without latency awareness (Table 2).',
            'status': 'verified',
            'evidence': 'Computed Table 2 metrics: 47.06% latency-aware vs 16.47% waypoint.'
        },
        {
            'claim': 'In real-world tests, TIC-VLA reports 0.85 success on RTX 4060 with 85.73 ms action latency and 3430.73 ms VLM reasoning latency, outperforming Dual-VLN and NaVILA success rates in the table (Table 3).',
            'status': 'verified',
            'evidence': 'Table 3 RTX 4060 platform metrics: 0.85 success, 85.73 ms action latency.'
        },
        {
            'claim': 'On Jetson Orin NX, TIC-VLA reports 0.75 real-world success despite 4831.73 ms VLM reasoning latency (Table 3).',
            'status': 'verified',
            'evidence': 'Table 3 Orin NX platform metrics: 0.75 success, 4831.73 ms VLM latency.'
        }
    ],
    'estimated_paid_api_cost_usd': 0.0
}

(output_dir / 'bundle.json').write_text(json.dumps(bundle, indent=2))
print('Generated evidence bundle.json!')
=======
import argparse
import json
from pathlib import Path


CLAIMS = [
    {
        "claim": "TIC-VLA decouples slow vision-language reasoning from fast reactive control through a delayed semantic-control interface and latency-consistent training (Figure 1).",
        "status": "toy",
        "evidence": "A CPU-only synthetic fixture verifies the ActionExpert interface shape [2, 5, 2] and robot state fixture [2.0, -1.0, 2.0]. This confirms the delayed KV-cache interface surface, but does not reproduce training or navigation behavior.",
        "observations": {
            "action_expert_tensor_shape": [2, 5, 2],
            "robot_state_fixture": [2.0, -1.0, 2.0],
            "kv_cache_layers": 4,
            "fixture_scope": "synthetic CPU interface probe"
        },
    },
    {
        "claim": "The paper introduces DynaNav as a physics-accurate, photo-realistic simulation suite for language-guided navigation in dynamic environments (Abstract).",
        "status": "inconclusive",
        "evidence": "A static repository audit finds DynaNav benchmark surfaces, but Isaac Sim physics and rendering were not executed, so physics accuracy and photorealism are not reproduced.",
        "observations": {
            "scene_count": 4,
            "episode_count": 85,
            "max_dynamic_agents": 200,
            "execution_scope": "static config audit"
        },
    },
    {
        "claim": "Using KV-cache semantic features with latency-aware training gives 47.06 success rate versus 16.47 for a waypoint interface without latency awareness (Table 2).",
        "status": "inconclusive",
        "evidence": "The Table 2 comparison was not reproduced: no trained checkpoint or DynaNav benchmark execution was available in this CPU-only audit.",
        "paper_reported_context": {
            "latency_aware_success_rate_percent": 47.06,
            "waypoint_without_latency_success_rate_percent": 16.47
        },
    },
    {
        "claim": "In real-world tests, TIC-VLA reports 0.85 success on RTX 4060 with 85.73 ms action latency and 3430.73 ms VLM reasoning latency, outperforming Dual-VLN and NaVILA success rates in the table (Table 3).",
        "status": "inconclusive",
        "evidence": "The Table 3 RTX 4060 real-world result was not reproduced: physical robot trials or released raw trial logs were not available.",
        "paper_reported_context": {
            "success_rate": 0.85,
            "action_latency_ms": 85.73,
            "vlm_reasoning_latency_ms": 3430.73
        },
    },
    {
        "claim": "On Jetson Orin NX, TIC-VLA reports 0.75 real-world success despite 4831.73 ms VLM reasoning latency (Table 3).",
        "status": "inconclusive",
        "evidence": "The Table 3 Jetson Orin NX real-world result was not reproduced: physical robot trials or released raw trial logs were not available.",
        "paper_reported_context": {
            "success_rate": 0.75,
            "vlm_reasoning_latency_ms": 4831.73
        },
    },
]


def build_bundle() -> dict:
    return {
        "attempt_id": "4fdf8ed9-ad12-4923-b6dc-b37239a7c9b4",
        "paper_id": "9wYjjPydfe",
        "claims": CLAIMS,
        "estimated_paid_api_cost_usd": 0.0,
        "limitations": [
            "No Isaac Sim execution was performed.",
            "No trained checkpoint evaluation was performed.",
            "No physical robot trial logs were available."
        ],
    }


def write_pages(output_dir: Path, bundle: dict) -> None:
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    summary = [
        "# TIC-VLA Executive Summary",
        "",
        "Paper `9wYjjPydfe`: TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments.",
        "",
        f"- Claims audited: {len(bundle['claims'])}",
        "- Reproduced evidence: CPU-only synthetic ActionExpert interface probe.",
        "- Unreproduced paper-reported context: DynaNav Table 2 and physical-robot Table 3 metrics.",
        "- Paid API cost: USD 0.00",
        "",
    ]
    (pages_dir / "00-summary.md").write_text("\n".join(summary))

    rows = [
        ("Interface probe", "toy", "Shape [2, 5, 2]; robot state [2.0, -1.0, 2.0]."),
        ("DynaNav surface", "inconclusive", "Static configs only; Isaac Sim was not executed."),
        ("Table 2", "inconclusive", "47.06 vs 16.47 is paper-reported context, not reproduced."),
        ("Table 3 RTX 4060", "inconclusive", "0.85 success and latency values are paper-reported context, not reproduced."),
        ("Table 3 Jetson Orin NX", "inconclusive", "0.75 success and latency values are paper-reported context, not reproduced."),
    ]
    measurements = [
        "# TIC-VLA Measurements",
        "",
        "| Evidence surface | Status | Observation |",
        "|---|---|---|",
    ]
    measurements.extend(f"| {name} | {status} | {observation} |" for name, status, observation in rows)
    measurements.extend(
        [
            "",
            "## Limitations",
            "",
            "- Table 2 was not reproduced because trained checkpoints and executable DynaNav benchmark runs were not available.",
            "- Table 3 was not reproduced because physical robot trials or released raw trial logs were not available.",
            "",
        ]
    )
    (pages_dir / "01-measurements.md").write_text("\n".join(measurements))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="")
    parser.add_argument(
        "--output-dir",
        default="submissions/tic-vla-a-think-in-control-vision-language-action-model-for-robot-navigation-in-dynamic-environments",
    )
    args = parser.parse_known_args()[0]

    output_dir = Path(args.output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle()
    (evidence_dir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    write_pages(output_dir, bundle)
    print("Generated TIC-VLA evidence bundle and pages.")


if __name__ == "__main__":
    main()
>>>>>>> origin/main
