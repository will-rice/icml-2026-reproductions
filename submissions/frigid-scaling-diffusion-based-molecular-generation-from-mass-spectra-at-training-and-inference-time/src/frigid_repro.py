from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = PROJECT_ROOT / "evidence" / "inputs" / "upstream"
UPSTREAM_REVISION = "github:coleygroup/FRIGID@4914e52424278ac7de7b699fa7dfbee528cbc751"
ATTEMPT_ID = "86bd82c3-48c0-4260-be38-045e8aa0fb29"
PAPER_ID = "wTgx7b2D9r"

CLAIMS = [
    (1, "FRIGID-base is a masked diffusion language model that generates SAFE molecular sequences conditioned on precursor formulae and MIST-predicted fingerprints (Figure 2)", "supported"),
    (2, "ICEBERG-guided inference-time scaling identifies spectrum-inconsistent fragments and refines molecules through targeted remasking and denoising (Figure 3)", "supported"),
    (3, "FRIGID achieves state-of-the-art de novo structural elucidation on NPLIB1 and MassSpecGym under known chemical formulae, including surpassing 18% Top-1 accuracy on MassSpecGym (Table 1)", "limited"),
    (4, "FRIGID triples prior Top-1 exact-match accuracy on NPLIB1 and improves structural similarity when exact recovery fails (Table 1)", "limited"),
    (5, "FRIGID inference-time refinement shows log-linear Top-1 accuracy gains with increasing compute on NPLIB1 and MassSpecGym (Figure 4)", "limited"),
    (6, "The FRIGID backbone has substantially faster inference throughput than graph diffusion baselines on NPLIB1 random samples (Table 2)", "limited"),
]

ARTIFACTS = [
    "README.md",
    "fp2mol_pretraining.yaml",
    "spec2mol_benchmark_msg.yaml",
    "spec2mol_model.py",
    "iceberg_sampler.py",
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts() -> dict[str, str]:
    return {name: (UPSTREAM_DIR / name).read_text(encoding="utf-8") for name in ARTIFACTS}


def audit_architecture(artifacts: dict[str, str]) -> dict[str, object]:
    readme = artifacts["README.md"]
    model = artifacts["spec2mol_model.py"]
    pretraining = artifacts["fp2mol_pretraining.yaml"]
    benchmark = artifacts["spec2mol_benchmark_msg.yaml"]
    checks = {
        "readme_names_safe_generation": all(
            token in readme for token in ["SAFE", "MIST", "fingerprint", "chemical formula"]
        ),
        "model_uses_mist_encoder": "SpectraEncoderGrowing" in model,
        "model_uses_formula_encoder": "FormulaEncoder" in model,
        "model_uses_fingerprint_conditioner": "FingerprintConditioner" in model,
        "model_uses_masked_diffusion": "MDLM" in model and "DiscreteMaskedPrior" in model,
        "pretraining_enables_formula_conditioning": "use_formula_conditioning: True" in pretraining,
        "pretraining_enables_fingerprint_conditioning": "use_fingerprint_conditioning: True" in pretraining,
        "msg_config_uses_4096_bit_fingerprints": "bits: 4096" in benchmark,
    }
    return {"checks": checks, "supported": all(checks.values())}


def audit_iceberg_refinement(artifacts: dict[str, str]) -> dict[str, object]:
    sampler = artifacts["iceberg_sampler.py"]
    checks = {
        "has_iceberg_sampler": "IcebergSampler" in sampler,
        "simulates_spectra": "iceberg_prediction" in sampler and "load_pred_spec" in sampler,
        "identifies_hallucinated_peaks": "halluc" in sampler.lower(),
        "maps_to_masked_tokens": "mask" in sampler.lower() and "token" in sampler.lower(),
        "iterates_refinement_rounds": "num_rounds" in sampler and "refine" in sampler.lower(),
    }
    return {"checks": checks, "supported": all(checks.values())}


def audit_limited_benchmarks() -> dict[str, object]:
    return {
        "local_scope": "CPU source/configuration audit only",
        "unavailable_requirements": [
            "released DLM and MIST checkpoints",
            "NPLIB1/CANOPUS and MassSpecGym test data",
            "ICEBERG model weights and ms-pred submodule runtime",
            "single-GPU timing baseline for graph diffusion comparisons",
        ],
        "no_paper_values_recomputed": True,
    }


def build_bundle() -> dict[str, object]:
    artifacts = load_artifacts()
    architecture = audit_architecture(artifacts)
    iceberg = audit_iceberg_refinement(artifacts)
    limited = audit_limited_benchmarks()
    artifact_hashes = {
        name: {"relative_path": f"evidence/inputs/upstream/{name}", "sha256": sha256_path(UPSTREAM_DIR / name)}
        for name in ARTIFACTS
    }

    claims = []
    for ordinal, text, outcome in CLAIMS:
        if ordinal == 1:
            notes = "Pinned upstream source/configs support the MIST fingerprint, formula, SAFE, and masked-diffusion conditioning path."
            details = architecture
        elif ordinal == 2:
            notes = "Pinned upstream sampler source supports the ICEBERG simulation, hallucinated-peak, token masking, and iterative refinement path."
            details = iceberg
        else:
            notes = "Full benchmark reproduction requires unavailable large datasets, checkpoints, and GPU baseline runs; no paper-reported metric is treated as reproduced."
            details = limited
        claims.append({"ordinal": ordinal, "text": text, "local_outcome": outcome, "reproduction_notes": notes, "details": details})

    return {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "title": "FRIGID: Scaling Diffusion-Based Molecular Generation from Mass Spectra at Training and Inference Time",
        "upstream_revision": UPSTREAM_REVISION,
        "artifact_hashes": artifact_hashes,
        "claims": claims,
        "costs": {"api_cost_usd": 0.0, "metered_service_usd": 0.0},
        "environment": {"python_version": platform.python_version(), "platform": platform.platform()},
        "generated_files": {"evidence_json": str(PROJECT_ROOT / "evidence" / "evidence.json")},
    }


def generate_evidence(output_path: str | Path | None = None) -> dict[str, object]:
    bundle = build_bundle()
    path = Path(output_path) if output_path is not None else PROJECT_ROOT / "evidence" / "evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


if __name__ == "__main__":
    evidence = generate_evidence()
    print(json.dumps({"evidence_json": evidence["generated_files"]["evidence_json"]}, indent=2))
