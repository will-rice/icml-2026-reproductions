from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_evidence import build_evidence


CLAIM_TEXTS = [
    "TD3B formulates allosteric binder design as control over sequence-conditioned transition operators rather than optimization toward static conformations (Section 4.2).",
    "The framework combines a target-aware Direction Oracle, a soft binding-affinity gate, and amortized fine-tuning of a masked discrete diffusion model (Figure 2).",
    "The Direction Oracle achieves 0.93 accuracy, 0.90 precision, 0.91 recall, and 0.90 F1 for binary direction classification (Table 1).",
    "TD3B obtains the highest gated reward among compared pre-trained, classifier guidance, SMC, TDS, and PepTune baselines (Table 2).",
    "For targeted transition control, generated binders achieve 61% success for forward transitions and 100% success for reverse transitions under the paper's success definition (Table 3).",
    "The paper evaluates TD3B-designed agonist and antagonist binders on GLP-1R and OX1R case studies (Figures 4 and 5).",
]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_build_evidence_binds_claims_and_marks_missing_metrics_unavailable(tmp_path: Path) -> None:
    """Break caught: metric claims cannot be marked reproduced without primary result artifacts."""
    source_root = tmp_path / "upstream"
    _write(
        source_root / "td3b" / "td3b_scoring.py",
        """
class TD3BReward:
    def gated_reward(self, direction_oracle, binding_affinity, target_direction, tau):
        return binding_affinity * sigmoid(target_direction * (direction_oracle - 0.5) / tau)
""",
    )
    _write(
        source_root / "td3b" / "direction_oracle.py",
        "class DirectionOracle: pass\n",
    )
    _write(
        source_root / "td3b" / "td3b_losses.py",
        "loss = L_WDCE + lambda_ctr * L_ctr + beta_kl * L_KL\n",
    )
    _write(
        source_root / "td3b" / "td3b_finetune.py",
        "def amortized_finetuning(masked_discrete_diffusion_model): return masked_discrete_diffusion_model\n",
    )
    _write(
        source_root / "models" / "diffusion.py",
        "MASK_TOKEN = '[MASK]'\nclass Diffusion: pass\n",
    )
    _write(
        source_root / "mcts" / "peptide_mcts.py",
        "class PeptideMCTS: pass\n",
    )
    _write(
        source_root / "inference.py",
        "def run():\n    direction_oracle = 1\n    affinity = 1\n    resample_alpha = 0.1\n",
    )
    _write(
        source_root / "README.md",
        "Data archive contains data/test.csv, generated_binders/agonist, antagonist.tar.gz, GLP-1R, and OX1R case studies.\n",
    )

    bundle = build_evidence(
        source_root=source_root,
        generated_at="2026-08-01T13:50:00+00:00",
    )

    assert bundle["paper_id"] == "RNuC8Nj6rD"
    assert bundle["upstream"]["repo_id"] == "ChatterjeeLab/TD3B"
    assert bundle["upstream"]["revision"] == "7d3c9bfe171a1db77e7b5431c572dadce8520bb5"
    assert bundle["challenge"]["snapshot_id"] == "d32beb9e79859f40a37e565155ef84fb3bdc6bf3679e8f79e8f5414cc3f60600"

    claim_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in CLAIM_TEXTS]
    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == claim_hashes

    statuses = {claim["id"]: claim["status"] for claim in bundle["claims"]}
    assert statuses == {
        1: "verified",
        2: "verified",
        3: "unavailable",
        4: "unavailable",
        5: "unavailable",
        6: "toy",
    }
    assert "data/test.csv" in bundle["missing_artifacts"]
    assert "generated_binders/agonist" in bundle["missing_artifacts"]
    assert bundle["checkpoint_lfs"]["checkpoints/td3b.ckpt"]["sha256"] == "9b8aeecbfe29b4652860028135c2d7abd2688cfa51aa939b419dd3aec41495d4"


def test_cli_writes_json_bundle(tmp_path: Path) -> None:
    """Break caught: command-line evidence generation must persist machine-readable JSON."""
    source_root = tmp_path / "upstream"
    _write(source_root / "td3b" / "td3b_scoring.py", "DirectionOracle soft binding-affinity gate reward\n")
    _write(source_root / "td3b" / "direction_oracle.py", "DirectionOracle\n")
    _write(source_root / "td3b" / "td3b_losses.py", "L_WDCE L_ctr L_KL\n")
    _write(source_root / "td3b" / "td3b_finetune.py", "amortized fine-tuning\n")
    _write(source_root / "models" / "diffusion.py", "masked discrete diffusion [MASK]\n")
    _write(source_root / "mcts" / "peptide_mcts.py", "MCTS\n")
    _write(source_root / "inference.py", "direction_oracle affinity resample_alpha\n")
    _write(source_root / "README.md", "GLP-1R OX1R\n")
    output = tmp_path / "bundle.json"

    from generate_evidence import main

    main(["--source-root", str(source_root), "--output", str(output), "--generated-at", "2026-08-01T13:50:00+00:00"])

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["generated_at"] == "2026-08-01T13:50:00+00:00"
    assert len(data["claims"]) == 6
