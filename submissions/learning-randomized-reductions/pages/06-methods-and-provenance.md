# Lane 6: Reproduction Methods, Environment & Provenance

## System Architecture

The reproduction repository is structured into modular Python components:

- `lrr_repro.provenance`: Validates manifest schema, SHA-256 hashes, file sizes, Git blob IDs, and paper context locators.
- `lrr_repro.theory`: Executes finite-model audits for correlated query distributions and uniform marginal bounds.
- `lrr_repro.benchmark`: AST-parses RSR-Bench registration calls and reconciles primary CSV rows.
- `lrr_repro.results`: Aggregates raw CSV results, computes backend coverage and runtime statistics, and verifies symbolic identities.
- `lrr_repro.claim_scope`: Performs conjunctive locator matching across arXiv paper versions.
- `lrr_repro.evidence`: Aggregates audit outputs into canonical JSON and validates against `schema/evidence-v1.schema.json`.
- `lrr_repro.cli`: Command-line tool `lrr-repro` providing `acquire`, `audit`, `validate`, and `propose`.

## Pinned Upstream Dependencies

| Artifact ID | Source / URL | SHA-256 Digest | Git Blob ID |
|---|---|---|---|
| `paper-v1` | arXiv:2412.18134v1 | `abaac08eabec2e77c8af7ae3ca028691b9cd862e21bfa779452b9fd729e3222f` | N/A |
| `paper-v5` | arXiv:2412.18134v5 | `93cab4aa8cec06434b704e639bab87dd15ea95ac46a335961138a94fc1bae2b8` | N/A |
| `LICENSE` | GitHub commit `e13d4b59` | `04601314559ab36aa7403fbaa56ccba106be0de6671190497e6835bbd3107bdb` | `2eb9ad588c5b6e720b168588a640d7a653265c96` |
| `results-csv` | GitHub commit `e13d4b59` | `7198413f93830f7903bf3b670b718f2ccfbab1a41496a1fc3fe085850af0df0b` | `0432241ef42d1be06179546c7b96d6bf6f598986` |
| `eval-rsr-bench-paper` | GitHub commit `e13d4b59` | `6afe05589eeb08f34d63f98ed55fc38a3856a84ce7fc1d21e47327baad54ffbf` | `1aa8de34e60dcdbf77c0af53e1d5af25a673522f` |
| `eval-rsr-bench-paper-extended` | GitHub commit `e13d4b59` | `02fcfb7805e1704e040ae9e854b22ab199ae7ea95a18b5e45f2f9c886c0f40e2` | `88f1ef6b3c1b280afd8d2754509a1f6f0b30df7c` |
| `eval-rsr-bench-agentic` | GitHub commit `e13d4b59` | `c99842f831d6bf0296452e632a5b0eb24f8ae9438acc42bad92604ac61c64bb0` | `e660ab8d39b8083117097641983fe69c5efaeffb` |
| `rsr-checker` | GitHub commit `e13d4b59` | `28e9dc80cec82d8e11dee4e867b55cba707b295c8f9d2065c7e9d286967fd3aa` | `417f5f0a8ef6789be4f01c3108f4427f2580c9d0` |
| `pac-py` | GitHub commit `e13d4b59` | `69da63ffce26b7f85f306a69e13eb345a8c12be7257f60297b96c5122c2274a4` | `4dd47da9cde937e7f7074424a9cddb5f3aa523ab` |

## Environment & Metered Resource Accounting

- Operating System: Linux x86_64
- Runtime: Python 3.12, SymPy 1.14, pypdf 6.1, jsonschema 4.25, Gradio 6.20, pytest 8.4
- Resource Consumption: CPU-only execution; total wall time under 2 seconds; GPU hours = 0; API cost = USD 0.00.

## Controller Correction & PDF Source Verification

The reproduction audit was updated under strict TDD to address controller validation requirements:
1. **Full Manifest & PDF Authentication**: `lrr-repro audit` authenticates all 9 manifest artifacts, including cached paper PDFs `2412.18134v1.pdf` and `2412.18134v5.pdf` via SHA-256 and Git blob IDs, failing closed on missing or tampered files before reading paper context.
2. **Direct PDF Source Text Extraction**: `lrr_repro.theory` and `lrr_repro.claim_scope` use `pypdf` to extract text from authenticated PDFs, verifying that Definitions 4.1, 4.3, 4.5 and Claims A.1, A.2 occur in v5 PDF text, and that v1 Table 2 (20 post-condition samples), Section 5.3.1 (RSR-Bench LR 594 vs MILP 1,095 samples, 130.53s vs 187.47s), Section 5.3.2 (NLA-DigBench DIG and SymInfer), and v5 Table 2 (Agentic Bitween query functions) match paper text.
3. **Restricted Property Scanning**: `novel_agentic_queries` restricts scanning to the exact released Claude-Opus-4.1 Agentic Bitween columns (columns 53–55), preserving the extraction of `x+log(k)` without scanning non-agentic fields.
