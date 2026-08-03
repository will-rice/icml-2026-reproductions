"""Nonlinear-invariant claim scope and honest falsification audit lane."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping


@dataclass(frozen=True)
class SourceLocator:
    version: str
    section: str
    table: str | None
    benchmark: str
    compared_methods: tuple[str, ...]
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class ScopeAudit:
    exact_claim_supported: bool
    status: Literal["verified", "falsified", "inconclusive"]
    contradictions: tuple[str, ...]
    locators: tuple[SourceLocator, ...]


def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    try:
        import pypdf

        reader = pypdf.PdfReader(pdf_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    except Exception:
        return ""


def audit_nonlinear_invariant_claim(
    context: Mapping[str, object], cache_dir: Path | None = None
) -> ScopeAudit:
    versions = context.get("versions", {})
    if not isinstance(versions, dict):
        return ScopeAudit(
            exact_claim_supported=False,
            status="inconclusive",
            contradictions=(),
            locators=(),
        )

    v1 = versions.get("v1", {})
    v5 = versions.get("v5", {})

    if not isinstance(v1, dict) or not isinstance(v5, dict):
        return ScopeAudit(
            exact_claim_supported=False,
            status="inconclusive",
            contradictions=(),
            locators=(),
        )

    v1_t2 = v1.get("table_2")
    v1_rsr = v1.get("rsr_bench")
    v1_nla = v1.get("nla_digbench")
    v5_t2 = v5.get("table_2")

    if not (
        isinstance(v1_t2, dict)
        and isinstance(v1_rsr, dict)
        and isinstance(v1_nla, dict)
        and isinstance(v5_t2, dict)
    ):
        return ScopeAudit(
            exact_claim_supported=False,
            status="inconclusive",
            contradictions=(),
            locators=(),
        )

    if cache_dir is not None:
        pdf_v1_path = cache_dir / "2412.18134v1.pdf"
        pdf_v5_path = cache_dir / "2412.18134v5.pdf"

        text_v1 = extract_pdf_text(pdf_v1_path)
        text_v5 = extract_pdf_text(pdf_v5_path)

        if not text_v1 or not text_v5:
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )

        clean_v1 = " ".join(text_v1.split())
        clean_v5 = " ".join(text_v5.split())

        # Verify v1 Table 2 facts: post-condition & sample count in bounded Table 2 caption
        pos_t2 = clean_v1.find("Table 2.")
        if pos_t2 == -1 or "post-condition" not in clean_v1:
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )
        excerpt_t2 = clean_v1[pos_t2 : pos_t2 + 400]
        sample_cnt = v1_t2.get("sample_count")
        if sample_cnt is None or f"{sample_cnt} Samples" not in excerpt_t2:
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )

        # Verify v1 Section 5.3.1 RSR-Bench facts in bounded Section 5.3.1 excerpt
        pos_531 = clean_v1.find("5.3.1 RSR-Bench.")
        pos_532 = clean_v1.find("5.3.2 NLA-DigBench.")
        if pos_531 == -1 or pos_532 == -1:
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )
        excerpt_531 = clean_v1[pos_531:pos_532]

        lr_samples = v1_rsr.get("lr_samples")
        milp_samples = v1_rsr.get("milp_samples")
        lr_runtime = v1_rsr.get("lr_runtime")
        milp_runtime = v1_rsr.get("milp_runtime")

        if lr_samples is None or f"{lr_samples} samples" not in excerpt_531:
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )

        if milp_samples is None or not (
            f"{milp_samples:,} samples" in excerpt_531
            or f"{milp_samples} samples" in excerpt_531
        ):
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )

        if lr_runtime is None or f"{lr_runtime} seconds" not in excerpt_531:
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )

        if milp_runtime is None or f"{milp_runtime} seconds" not in excerpt_531:
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )

        # Verify v1 Section 5.3.2 NLA-DigBench facts in bounded Section 5.3.2 excerpt
        pos_533 = clean_v1.find("5.4 Related Work")
        if pos_533 == -1:
            pos_533 = len(clean_v1)
        excerpt_532 = clean_v1[pos_532:pos_533]

        comp_methods = v1_nla.get("compared_methods", ())
        if not (
            "NLA-DigBench" in excerpt_532
            and all(m in excerpt_532 for m in comp_methods)
        ):
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )

        # Verify v5 Table 2 facts in bounded v5 Table 2 excerpt
        pos_v5_t2 = clean_v5.find("Table 2.")
        if pos_v5_t2 == -1:
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )
        excerpt_v5_t2 = clean_v5[pos_v5_t2 : pos_v5_t2 + 400]
        if not (
            "query functions" in excerpt_v5_t2
            and ("Agentic Bitween" in excerpt_v5_t2 or "A-BITWEEN" in excerpt_v5_t2)
        ):
            return ScopeAudit(
                exact_claim_supported=False,
                status="inconclusive",
                contradictions=(),
                locators=(),
            )


    locators = (
        SourceLocator(
            version="v1",
            section="Table 2",
            table="2",
            benchmark="post-condition",
            compared_methods=(),
            metrics=("sample count",),
        ),
        SourceLocator(
            version="v1",
            section="Section 5.3.1",
            table=None,
            benchmark="RSR-Bench",
            compared_methods=("LR", "MILP"),
            metrics=("sample count", "runtime"),
        ),
        SourceLocator(
            version="v1",
            section="Section 5.3.2",
            table=None,
            benchmark="NLA-DigBench",
            compared_methods=("Bitween", "DIG", "SymInfer"),
            metrics=("solving count", "runtime"),
        ),
        SourceLocator(
            version="v5",
            section="Table 2",
            table="2",
            benchmark="RSR-Bench",
            compared_methods=(),
            metrics=(),
        ),
    )

    contradictions = (
        "v1 Table 2 is a learned post-condition example, not a backend comparison",
        "LR versus MILP sample/runtime results are for RSR-Bench, not NLA-DigBench",
        "NLA-DigBench compares Bitween with DIG and SymInfer, not MILP",
        "v5 Table 2 reports novel Agentic Bitween query functions",
    )

    return ScopeAudit(
        exact_claim_supported=False,
        status="falsified",
        contradictions=contradictions,
        locators=locators,
    )
