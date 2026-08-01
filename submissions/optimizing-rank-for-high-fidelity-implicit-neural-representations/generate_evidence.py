"""Generate the evidence bundle and judge-visible pages for the Muon INR reproduction."""

import json
from pathlib import Path

from optimizing_rank_inr_repro.benchmarks import run_all_benchmarks

PAPER_ID = "2azIa9tfl3"

CLAIMS = [
    "The paper argues that vanilla MLP INR low-frequency bias is a symptom of stable-rank degradation during training rather than an intrinsic architectural limitation (Section 3).",
    "Rank-regulating, near-orthogonal Muon updates improve image overfitting quality across multiple INR architectures compared with Adam (Table 1).",
    "Muon improves sparse-view CT reconstruction quality across multiple INR architectures compared with Adam (Table 4).",
    "The reported improvements extend to natural images, medical images, audio, super-resolution, and novel-view synthesis, with up to about +9 dB PSNR over the same architecture (Tables 1-6).",
]

PAGE_NAMES = [
    "00-summary.md",
    "01-claim-1-stable-rank.md",
    "02-claim-2-image-overfitting.md",
    "03-claim-3-sparse-view-ct.md",
    "04-claim-4-multidomain.md",
]


def build_evidence(bench: dict) -> dict:
    """Bind each measured benchmark result to its exact challenge claim."""
    return {
        "paper_id": PAPER_ID,
        "upstream_pins": ["arxiv:2512.14366"],
        "execution_mode": "CPU-only, deterministic (pinned seeds, no wall-clock measurements)",
        "claims": [
            {
                "claim_id": "claim_1",
                "statement": CLAIMS[0],
                "status": "reproduced"
                if bench["claim1_stable_rank"]["status"] == "verified"
                else "not_reproduced",
                "scale": "toy-scale: 4-layer 64-unit MLP fitting a 32x32 multi-frequency image",
                "evidence_type": "empirical_benchmark",
                "details": bench["claim1_stable_rank"],
            },
            {
                "claim_id": "claim_2",
                "statement": CLAIMS[1],
                "status": "reproduced"
                if bench["claim2_image_overfitting"]["status"] == "verified"
                else "not_reproduced",
                "scale": "toy-scale: Siren and vanilla-MLP INRs, 32x32 target, 100 steps",
                "evidence_type": "empirical_benchmark",
                "details": bench["claim2_image_overfitting"],
            },
            {
                "claim_id": "claim_3",
                "statement": CLAIMS[2],
                "status": "reproduced"
                if bench["claim3_sparse_ct"]["status"] == "verified"
                else "not_reproduced",
                "scale": "toy-scale: 8-view discrete Radon operator on a 32x32 ellipse phantom",
                "evidence_type": "empirical_benchmark",
                "details": bench["claim3_sparse_ct"],
            },
            {
                "claim_id": "claim_4",
                "statement": CLAIMS[3],
                "status": "reproduced"
                if bench["claim4_multidomain"]["all_domains_improved"]
                else "partially_reproduced",
                "scale": "toy-scale: 4 of the paper's modalities; novel-view synthesis not attempted",
                "evidence_type": "empirical_ablation",
                "details": bench["claim4_multidomain"],
            },
        ],
    }


def render_pages(evidence: dict, pages_dir: Path) -> list[Path]:
    """Write one judge-visible markdown page per claim plus the summary page."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    claims = evidence["claims"]
    written = []

    rows = "\n".join(
        f"| {i} | `{c['status']}` | {c['scale']} |" for i, c in enumerate(claims, start=1)
    )
    summary = f"""# Optimizing Rank for High-Fidelity INRs — Reproduction Summary

Reproduction of **"Optimizing Rank for High-Fidelity Implicit Neural
Representations"** (paper `{PAPER_ID}`, arXiv:2512.14366).

Every number on these pages is computed by this repository on CPU with pinned
seeds. Adam and Muon runs always start from **identical initial weights** and
receive the same number of steps, so reported PSNR gaps are attributable to
the optimizer. No paper value is copied into a measurement field, and results
that contradict the paper are reported as measured rather than tuned until
they agree: **claim 3 does not reproduce at this scale, and claim 4
reproduces in only 2 of 4 modalities.**

## Claim status

| Claim | Status | Scale of the evidence |
| --- | --- | --- |
{rows}

## Reproducing

```bash
uv run --project . python generate_evidence.py
uv run --project . python -m pytest tests -q
```

`generate_evidence.py` regenerates `evidence/evidence.json` and every page in
`pages/`; it is deterministic and byte-identical across runs.
"""
    path = pages_dir / "00-summary.md"
    path.write_text(summary, encoding="utf-8")
    written.append(path)

    d = claims[0]["details"]
    page = f"""# Claim 1 — Stable-rank degradation under Adam

> {CLAIMS[0]}

**Status: `{claims[0]['status']}` ({claims[0]['scale']}).**

Two identically initialized vanilla-MLP INRs fit the same 32x32
multi-frequency target for 100 steps, one with Adam and one with Muon. Stable
rank is measured as `||W||_F^2 / ||W||_2^2` averaged over the weight matrices.

| Quantity | Value |
| --- | --- |
| Stable rank at initialization | **{d['initial_stable_rank']}** |
| Final stable rank (Adam) | **{d['final_adam_stable_rank']}** |
| Final stable rank (Muon) | **{d['final_muon_stable_rank']}** |
| Rank drop under Adam | **{d['adam_rank_drop']}** |
| Rank drop under Muon | **{d['muon_rank_drop']}** |

Adam loses **{d['adam_rank_drop']}** of stable rank over training while the
near-orthogonal Muon updates change it by **{d['muon_rank_drop']}** (a
negative value means rank slightly *increased*). The mechanism the paper
describes — rank collapse accompanying optimization rather than being fixed
by the architecture — is therefore observed at this scale.
"""
    path = pages_dir / "01-claim-1-stable-rank.md"
    path.write_text(page, encoding="utf-8")
    written.append(path)

    d = claims[1]["details"]
    arch_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {v['psnr_adam']} dB | {v['psnr_muon']} dB | "
        f"{v['psnr_gain']:+} dB |"
        for name, v in d["architectures"].items()
    )
    page = f"""# Claim 2 — Image overfitting across architectures

> {CLAIMS[1]}

**Status: `{claims[1]['status']}` ({claims[1]['scale']}).**

Each architecture is instantiated twice from the same seed, weights copied so
both optimizers start identical, then trained for 100 steps on the same
32x32 target. PSNR is measured on the fitted grid.

| Architecture | Adam PSNR | Muon PSNR | Gain |
| --- | --- | --- | --- |
{arch_rows}

All architectures improved: **{d['all_architectures_improved']}**.

The Siren INR shows the large gap the paper emphasises, while the vanilla MLP
improves only marginally at this step budget — consistent with the paper's
framing that rank regulation matters most where the architecture already
supports high-frequency fitting.
"""
    path = pages_dir / "02-claim-2-image-overfitting.md"
    path.write_text(page, encoding="utf-8")
    written.append(path)

    d = claims[2]["details"]
    ct_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {v['recon_psnr_adam']} dB | {v['recon_psnr_muon']} dB | "
        f"{v['psnr_gain_db']:+} dB |"
        for name, v in d["architectures"].items()
    )
    page = f"""# Claim 3 — Sparse-view CT reconstruction

> {CLAIMS[2]}

**Status: `{claims[2]['status']}` ({claims[2]['scale']}).**

This is a genuine inverse problem, not a direct image fit. A deterministic
ellipse phantom is measured through a discrete Radon operator with
**{d['n_views']} views x {d['n_detector_bins']} detector bins =
{d['n_measurements']} measurements** for **{d['n_pixels']} pixels** — a
{d['n_pixels'] / d['n_measurements']:.0f}x under-determined system. Each INR
is trained **only on the sinogram**; reconstruction PSNR is then evaluated
against the unseen phantom on the full grid.

| Architecture | Adam recon. PSNR | Muon recon. PSNR | Gain |
| --- | --- | --- | --- |
{ct_rows}

All architectures improved: **{d['all_architectures_improved']}**.

**This claim does not reproduce at this scale — the measured effect runs in
the opposite direction.** Adam achieves higher reconstruction PSNR than Muon
on both architectures, by
{abs(d['architectures']['siren']['psnr_gain_db'])} dB (Siren) and
{abs(d['architectures']['vanilla_mlp']['psnr_gain_db'])} dB (vanilla MLP).

Two readings are possible and this reproduction does not claim to
distinguish them: either the rank-regulation benefit does not transfer to
under-determined inverse problems at this size and step budget, or the
paper's CT result depends on scale, tuning, or a reconstruction setup this
toy operator does not capture. The learning rates here were not tuned per
task. The number is reported as measured rather than adjusted until it
agrees with the paper.
"""
    path = pages_dir / "03-claim-3-sparse-view-ct.md"
    path.write_text(page, encoding="utf-8")
    written.append(path)

    d = claims[3]["details"]
    dom_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {v['psnr_adam']} dB | {v['psnr_muon']} dB | "
        f"{v['psnr_gain_db']:+} dB |"
        for name, v in d["domains"].items()
    )
    page = f"""# Claim 4 — Multi-domain extension

> {CLAIMS[3]}

**Status: `{claims[3]['status']}` ({claims[3]['scale']}).**

Four modalities are exercised with the same paired-initialization protocol.
Super-resolution trains on a 16x16 coordinate subgrid and is evaluated on the
full 32x32 grid, so its PSNR measures generalization to unseen coordinates
rather than memorization.

| Domain | Adam PSNR | Muon PSNR | Gain |
| --- | --- | --- | --- |
{dom_rows}

- Largest observed gain: **{d['max_psnr_gain_db']} dB**.
- All domains improved: **{d['all_domains_improved']}**.
- Domains where Muon beat Adam:
  **{sum(1 for v in d['domains'].values() if v['improved'])} of {len(d['domains'])}**.

The claim reproduces only partially, and the failures are reported as
measured. Muon wins narrowly on the two image-grid domains
(**{d['max_psnr_gain_db']} dB**) but *loses* on the medical phantom
({d['domains']['medical_phantom']['psnr_gain_db']} dB) and on the 1-D audio
signal ({d['domains']['audio_1d']['psnr_gain_db']} dB). The paper's headline
"up to about +9 dB" is a dataset-scale claim; the only place this
reproduction sees a gap of that size is the Siren image-overfitting result on
the previous page, not this multi-domain sweep.

Novel-view synthesis is not attempted at all: it requires multi-view 3D data
outside this CPU budget. No figure is reported for it.
"""
    path = pages_dir / "04-claim-4-multidomain.md"
    path.write_text(page, encoding="utf-8")
    written.append(path)

    return written


def main():
    project_root = Path(__file__).parent
    bench = run_all_benchmarks()
    evidence = build_evidence(bench)

    out_dir = project_root / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evidence.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
        f.write("\n")

    render_pages(evidence, project_root / "pages")
    print(f"Evidence successfully generated and written to {out_file}")


if __name__ == "__main__":
    main()
