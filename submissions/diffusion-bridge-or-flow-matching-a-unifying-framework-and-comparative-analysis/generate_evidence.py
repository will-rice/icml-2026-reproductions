from pathlib import Path
import sys
import json


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dbfm_repro.evidence import generate_evidence_bundle  # noqa: E402


def main() -> None:
    bundle = generate_evidence_bundle(PROJECT_ROOT / "evidence")
    write_computed_page(bundle)
    print(f"Wrote evidence bundle to {bundle}")


def write_computed_page(bundle_path: Path) -> None:
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    bridge = data["observations"]["bridge_proxy"]
    interpolation = data["observations"]["flow_matching_interpolation"]
    page = PROJECT_ROOT / "pages" / "01-computed-evidence.md"
    page.write_text(
        "\n".join(
            [
                "# Computed Evidence",
                "",
                "## Flow Matching Formula Check",
                "",
                f"x_t: {interpolation['x_t']}",
                f"velocity: {interpolation['velocity']}",
                "",
                "## Bridge Proxy",
                "",
                f"samples: {bridge['samples']}",
                f"steps: {bridge['steps']}",
                f"seed: {bridge['seed']}",
                f"bridge_action: {bridge['bridge_action']}",
                f"flow_noisy_action: {bridge['flow_noisy_action']}",
                f"bridge_endpoint_abs_error: {bridge['bridge_endpoint_abs_error']}",
                f"flow_noisy_endpoint_abs_error: {bridge['flow_noisy_endpoint_abs_error']}",
                "",
                "The full image restoration, inpainting, data-scaling, and ablation claims remain unavailable because they require full GPU training and evaluation artifacts that were not independently recomputed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
