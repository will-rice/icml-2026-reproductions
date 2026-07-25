# DeMix Evidence Integrity Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DeMix's synthetic scores and self-reported verification with a
deterministic, independently recomputed audit of the exact released mixture
manifest.

**Architecture:** A standard-library artifact module verifies the vendored
manifest hash, validates its values, and derives canonical mixture observations.
The pipeline combines those observations with immutable checked-in provenance
and conservative per-claim dispositions. The Gradio Space reads only committed
evidence and released mixtures; it never fabricates targets or proxy scores.

**Tech Stack:** Python 3.12.11 standard library, Gradio 6.20.0, pytest 8.4.2,
canonical JSON, Docker.

## Global Constraints

- Work only in
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/`
  plus this plan and its approved design.
- Use paper `arXiv:2602.00747v3`, code commit
  `d0c945ca84d5632c6ed1bfe469337cf880757422`, and dataset revision
  `82a2effc58eb79bec691280a4e4fc50be0968b1e`.
- Accept only mixture input SHA-256
  `2be00152f98c44a740bc2f8e2098be3740ea2f1cd31b7158ade9d54c8e852dc2`.
- Recompute only manifest counts, domain membership, raw sums, normalized
  weights, and validation facts. Do not compute or report proxy scores,
  benchmark scores, correlations, or model behavior.
- Overall status is `partial`; weighted linear merging is `partial`; Spearman
  proxy accuracy and mixture optimization/benchmarking are `unavailable`.
- Paper-reported values are permitted only under `paper_context`, labeled
  non-reproduced.
- Preserve `demo.launch(server_name="0.0.0.0", server_port=7860)`.
- Pin the Space base to
  `python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7`,
  Gradio to `6.20.0`, and pytest to `8.4.2`.
- Do not deploy, publish, mutate `state/repro-loop.json` or `docs/HANDOFF.md`,
  or inspect, run, modify, test, or format NAPE.

---

### Task 1: Pinned released artifact and deterministic analysis

**Files:**

- Create:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/evidence/inputs/sampled_mixture.json`
- Create:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/evidence/provenance.json`
- Create:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/src/demix/artifacts.py`
- Replace:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/tests/test_demix.py`
- Delete:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/src/demix/merging.py`
- Delete:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/src/demix/eval.py`

**Interfaces:**

- Consumes: a `Path` to the exact vendored JSON manifest and an optional
  expected SHA-256.
- Produces:
  `load_pinned_manifest(path: Path, expected_sha256: str = PINNED_MIXTURE_SHA256) -> dict[str, dict[str, Decimal]]`,
  `normalize_weights(weights: Mapping[str, Decimal]) -> dict[str, str]`, and
  `analyze_manifest(manifest: Mapping[str, Mapping[str, Decimal]], reference_model_count: int = 16) -> dict[str, object]`.

- [ ] **Step 1: Vendor the exact pinned manifest and provenance**

Use `apply_patch` to add the exact 17-mixture JSON payload already acquired at
`/tmp/demix-corpora-82a2eff/DeMix_reproduce/reference_models/sampled_mixture.json`,
then verify its bytes:

```bash
sha256sum submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/evidence/inputs/sampled_mixture.json
```

The hash output must be:

```text
2be00152f98c44a740bc2f8e2098be3740ea2f1cd31b7158ade9d54c8e852dc2
```

Create `evidence/provenance.json` with the exact paper, code, and dataset
revisions; paper and inspected upstream-file hashes; dataset acquisition command;
the 1,469-path release inventory; 16 reference roots; seven component roots;
zero CSV and OpenCompass-result paths; and all fourteen component shard LFS
hashes and sizes totaling 48,176,346,736 bytes.

- [ ] **Step 2: Write failing artifact-integrity and observation tests**

Replace the synthetic tests with contracts equivalent to:

```python
def test_pinned_manifest_observations():
    manifest = load_pinned_manifest(MANIFEST)
    observations = analyze_manifest(manifest)
    assert observations["mixture_count"] == 17
    assert observations["domain_names"] == [
        "general_target", "math_very_high", "math_high", "math_medium",
        "code_very_high", "code_high", "code_medium",
    ]
    assert observations["raw_weight_sums"]["mix_0"] == "2933"
    assert observations["raw_weight_sums"]["mix_2"] == "0.9998"
    assert (
        observations["normalized_weights"]["mix_0"]["general_target"]
        == "0.399931810433"
    )
    assert (
        observations["normalized_weights"]["mix_2"]["general_target"]
        == "0.390678135627"
    )
    assert observations["normalization_required"] == [
        "mix_0", "mix_1", "mix_2", "mix_6",
        "mix_11", "mix_12", "mix_13", "mix_14",
    ]
    assert observations["reference_model_count"] == 16
    assert observations["manifest_reference_count_match"] is False


def test_modified_manifest_is_rejected(tmp_path):
    modified = tmp_path / "sampled_mixture.json"
    modified.write_bytes(MANIFEST.read_bytes() + b"\\n")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_pinned_manifest(modified)


@pytest.mark.parametrize(
    "weights",
    [{}, {"x": Decimal("0")}, {"x": Decimal("-1")},
     {"x": Decimal("NaN")}, {"x": Decimal("Infinity")}],
)
def test_normalization_rejects_invalid_weights(weights):
    with pytest.raises(ArtifactValidationError):
        normalize_weights(weights)
```

Also assert that every normalized vector sums to exactly one when parsed back
to `Decimal`.

- [ ] **Step 3: Run the new tests and record RED**

Run from the submission directory:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q tests/test_demix.py
```

Expected: FAIL during collection because `demix.artifacts` does not exist. Keep
the exact failing output for the handoff.

- [ ] **Step 4: Implement strict parsing and canonical observations**

Implement:

```python
PINNED_MIXTURE_SHA256 = (
    "2be00152f98c44a740bc2f8e2098be3740ea2f1cd31b7158ade9d54c8e852dc2"
)


class ArtifactIntegrityError(ValueError):
    pass


class ArtifactValidationError(ValueError):
    pass


def load_pinned_manifest(path, expected_sha256=PINNED_MIXTURE_SHA256):
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ArtifactIntegrityError(
            f"manifest SHA-256 {actual} does not match pinned {expected_sha256}"
        )
    return json.loads(
        payload.decode("utf-8"), parse_float=Decimal, parse_int=Decimal
    )
```

`normalize_weights` must reject empty, boolean, non-numeric, non-finite,
negative, and zero-sum values. Use `Decimal` and a fixed
`localcontext(prec=28)`. Quantize each normalized value except the final
ordered domain to `Decimal("0.000000000001")` with `ROUND_HALF_EVEN`; set the
final domain to `Decimal("1") - sum(previous_values)`. Return every normalized
value with exactly twelve decimal places. `analyze_manifest` must enforce
identical ordered domains across all mixtures and return mixture IDs, domain
names, plain-decimal raw sums, normalized weights, the exact non-unit list,
non-negativity/positive-sum facts, and the 17-versus-16 release count
comparison.

Remove `eval.py` and `merging.py` rather than preserving synthetic tensor and
correlation helpers.

- [ ] **Step 5: Run artifact tests GREEN**

Run:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q tests/test_demix.py
```

Expected: all Task 1 tests PASS.

- [ ] **Step 6: Review and commit Task 1**

Run `git diff --check`, inspect the exact vendored hash, and verify
`rg -n 'np\\.diag|spearmanr|ground_truth|proxy_scores' src tests` returns no
matches. Commit:

```bash
git add evidence/inputs/sampled_mixture.json evidence/provenance.json \
  src/demix/artifacts.py src/demix/eval.py src/demix/merging.py \
  tests/test_demix.py
git commit -m "test(demix): require pinned released evidence"
```

### Task 2: Conservative bundle pipeline and byte-identical regeneration

**Files:**

- Replace:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/src/demix/pipeline.py`
- Replace:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/evidence/bundle.json`
- Modify:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/tests/test_demix.py`
- Modify:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/src/demix/__init__.py`

**Interfaces:**

- Consumes: Task 1's `load_pinned_manifest` and `analyze_manifest`, plus a
  provenance JSON path.
- Produces:
  `build_bundle(input_path: Path, provenance_path: Path) -> dict[str, object]`,
  `write_bundle(bundle: Mapping[str, object], output_path: Path) -> None`, and a
  CLI accepting required `--input`, `--provenance`, and `--output` paths.

- [ ] **Step 1: Write failing conservative-bundle tests**

Add:

```python
def test_bundle_has_conservative_per_claim_statuses():
    bundle = build_bundle(MANIFEST, PROVENANCE)
    assert bundle["reproduction_status"] == "partial"
    claims = {claim["id"]: claim for claim in bundle["claims"]}
    assert claims["weighted-linear-model-merging"]["status"] == "partial"
    assert claims["spearman-proxy-accuracy"]["status"] == "unavailable"
    assert claims["mixture-optimization-benchmarking"]["status"] == "unavailable"
    assert claims["weighted-linear-model-merging"]["input_artifacts"] == [
        "evidence/inputs/sampled_mixture.json",
        "upstream:model_merge/generate_merge_yaml.py",
    ]
    assert claims["spearman-proxy-accuracy"]["observation"] is None
    assert claims["mixture-optimization-benchmarking"]["observation"] is None


def test_bundle_contains_no_synthetic_or_verified_evidence():
    encoded = json.dumps(build_bundle(MANIFEST, PROVENANCE))
    for forbidden in (
        '"verified"', '"macro_spearman"', '"ground_truth"',
        '"proxy_scores"', '"multi_seed_stability"',
    ):
        assert forbidden not in encoded


def test_cli_regeneration_is_byte_identical(tmp_path):
    output = tmp_path / "bundle.json"
    subprocess.run(
        [sys.executable, "-m", "demix.pipeline", "--input", str(MANIFEST),
         "--provenance", str(PROVENANCE), "--output", str(output)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(SUBMISSION_ROOT / "src")},
    )
    assert output.read_bytes() == BUNDLE.read_bytes()
```

Also assert exact provenance revisions and hashes, acquisition commands, pinned
environment, 48,176,346,736-byte resource limitation, and that every paper
value is nested below `paper_context` with `reproduced: false`.

- [ ] **Step 2: Run the bundle tests and record RED**

Run:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q tests/test_demix.py
```

Expected: FAIL because the existing pipeline has no path-based `build_bundle`
or deterministic CLI and still reports `verified`.

- [ ] **Step 3: Implement the deterministic conservative pipeline**

Build a bundle with these stable top-level keys:

```python
bundle = {
    "schema_version": 1,
    "paper_id": "uyRIOjFgOn",
    "title": PAPER_TITLE,
    "reproduction_status": "partial",
    "provenance": provenance,
    "environment": {
        "python": "3.12.11",
        "gradio": "6.20.0",
        "pytest": "8.4.2",
    },
    "regeneration": {"command": REGENERATION_COMMAND},
    "released_artifact_observations": observations,
    "claims": claims,
    "paper_context": {
        "reproduced": False,
        "note": "Paper-reported values below are context only.",
        "table_2": {
            "demix_30b_x7_macro_spearman": 0.81,
            "demix_30b_x7_top25_spearman": 0.59,
            "demix_30b_x7_capability_recovery": 0.83,
        },
    },
}
```

Use `json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\\n"`.
Validate the provenance's exact input hash and immutable revisions before
building. Use `argparse` for the three required CLI options. Do not include a
zero-argument fallback that could silently generate evidence from handwritten
defaults.

- [ ] **Step 4: Generate the committed bundle**

From the submission directory, run:

```bash
PYTHONPATH=src python -m demix.pipeline \
  --input evidence/inputs/sampled_mixture.json \
  --provenance evidence/provenance.json \
  --output evidence/bundle.json
```

Run the same command a second time and confirm `sha256sum evidence/bundle.json`
is unchanged.

- [ ] **Step 5: Run bundle tests GREEN**

Run:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q tests/test_demix.py
```

Expected: all artifact and bundle tests PASS.

- [ ] **Step 6: Review and commit Task 2**

Run `git diff --check`, inspect the entire generated bundle, and verify:

```bash
rg -n '"verified"|macro_spearman|ground_truth|proxy_scores|multi_seed_stability' \
  src evidence/bundle.json tests
```

The only permitted matches are forbidden-string assertions in tests. Commit:

```bash
git add src/demix/__init__.py src/demix/pipeline.py \
  evidence/bundle.json tests/test_demix.py
git commit -m "feat(demix): generate conservative artifact evidence"
```

### Task 3: Read-only Space, pinned runtime, and reproducibility documentation

**Files:**

- Replace:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/app.py`
- Modify:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/tests/test_app_startup.py`
- Modify:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/requirements.txt`
- Modify:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/requirements-test.txt`
- Modify:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/Dockerfile`
- Replace:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/README.md`
- Modify:
  `submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/tests/test_demix.py`

**Interfaces:**

- Consumes: committed `evidence/bundle.json` and
  `evidence/inputs/sampled_mixture.json`.
- Produces: `get_evidence() -> dict[str, object]`,
  `get_mixture_observation(mixture_id: str) -> dict[str, object]`, and a
  read-only Gradio app listening on wildcard port 7860.

- [ ] **Step 1: Write failing app and runtime-integrity tests**

Add assertions equivalent to:

```python
def test_space_source_contains_no_synthetic_calculator():
    source = APP.read_text()
    for forbidden in (
        "numpy", "calculate_merge", "merge_parameters",
        "evaluate_merged_model", "Predicted Benchmark Performance",
    ):
        assert forbidden not in source
    assert 'server_name="0.0.0.0", server_port=7860' in source


def test_runtime_dependencies_are_exactly_pinned():
    assert REQUIREMENTS.read_text() == "gradio==6.20.0\\n"
    assert REQUIREMENTS_TEST.read_text() == (
        "-r requirements.txt\\npytest==8.4.2\\n"
    )
    assert DOCKERFILE.read_text().startswith(
        "FROM python:3.12.11-slim-bookworm@sha256:"
        "519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7\\n"
    )
```

Retain the process-level startup test that inspects `/proc/<pid>/net/tcp`.

- [ ] **Step 2: Run app/runtime tests and record RED**

Run:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q tests/test_demix.py tests/test_app_startup.py
```

Expected: FAIL on the current NumPy simulator, dependency ranges, and unpinned
base image.

- [ ] **Step 3: Implement the read-only evidence inspector**

Load the committed JSON and expose a dropdown over released mixture IDs. The
selection callback returns only the corresponding entry from
`released_artifact_observations`, including raw sum and normalized weights.
Label claim evidence as `partial`/`unavailable`; never say “verified” or
“predicted benchmark.” End the file exactly with:

```python
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

Pin the runtime:

```text
# requirements.txt
gradio==6.20.0

# requirements-test.txt
-r requirements.txt
pytest==8.4.2
```

Start the Dockerfile with the exact digest from Global Constraints. Keep
`WORKDIR /app`, the non-root UID 1000 user, requirements installation, source
copy, and `CMD ["python", "app.py"]`.

- [ ] **Step 4: Rewrite the README evidence boundary and commands**

Document the three claim statuses, exact revisions and input hash, 17-versus-16
observation, absent benchmark outputs, upstream random placeholder, 48.18 GB
resource limitation, exact regeneration command, exact isolated test command,
and that paper values are context only. Remove every “interactive simulator”
and “verified reproduction” claim.

- [ ] **Step 5: Run app/runtime tests GREEN**

Run:

```bash
uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q tests/test_demix.py tests/test_app_startup.py
```

Expected: all tests PASS and the real Gradio process listens on `0.0.0.0:7860`.

- [ ] **Step 6: Review and commit Task 3**

Run `git diff --check`, review the complete app and README, and ensure `rg -n
'Interactive Model Merging Simulator|Verified Reproduction|numpy|scipy'` over
the submission returns no stale production/documentation claims. Commit:

```bash
git add app.py Dockerfile README.md requirements.txt requirements-test.txt \
  tests/test_app_startup.py tests/test_demix.py
git commit -m "fix(demix): expose only released artifact evidence"
```

### Task 4: End-to-end verification and scoped hygiene

**Files:**

- Verify all modified DeMix files.
- Modify this plan only to check completed task boxes if desired; task tracking
  is not evidence and must not change the generated bundle.

**Interfaces:**

- Consumes: Tasks 1–3.
- Produces: final test, regeneration, pre-commit, and diff evidence for handoff.

- [ ] **Step 1: Verify byte-identical regeneration**

From the submission directory:

```bash
before="$(sha256sum evidence/bundle.json)"
PYTHONPATH=src python -m demix.pipeline \
  --input evidence/inputs/sampled_mixture.json \
  --provenance evidence/provenance.json \
  --output evidence/bundle.json
after="$(sha256sum evidence/bundle.json)"
test "$before" = "$after"
```

Expected: exit 0 with identical hashes.

- [ ] **Step 2: Run the complete isolated DeMix suite**

Run:

```bash
env UV_CACHE_DIR=/tmp/demix-repro-uv-cache \
  uv run --isolated --no-project \
  --with-requirements requirements-test.txt \
  python -m pytest -q
```

Expected: all tests PASS.

- [ ] **Step 3: Run scoped pre-commit without touching NAPE**

From the workspace root, run pre-commit only on the changed design, plan, and
DeMix paths:

```bash
env UV_CACHE_DIR=/tmp/demix-repro-uv-cache \
  PRE_COMMIT_HOME=/tmp/demix-repro-pre-commit \
  uv run pre-commit run --files \
  docs/superpowers/specs/2026-07-25-demix-reproduction-design.md \
  docs/superpowers/plans/2026-07-25-demix-evidence-integrity-remediation.md \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/Dockerfile \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/README.md \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/app.py \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/requirements-test.txt \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/requirements.txt \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/src/demix/__init__.py \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/src/demix/artifacts.py \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/src/demix/pipeline.py \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/tests/conftest.py \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/tests/test_app_startup.py \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training/tests/test_demix.py
```

Expected: all configured hooks PASS or SKIP. Do not run `pre-commit -a`.

- [ ] **Step 4: Perform final evidence and scope audit**

Run:

```bash
git diff --check
git status --short
git diff --stat HEAD~4..HEAD
rg -n '"verified"|macro_spearman|ground_truth|proxy_scores|multi_seed_stability' \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training
```

Inspect every remaining match. Only tests that reject forbidden fields and
explicit README language saying they are absent are acceptable. Confirm no
NAPE, coordinator-state, or HANDOFF path appears in the branch diff.

- [ ] **Step 5: Apply only mechanical verification fixes and rerun checks**

If a formatter changes scoped files, inspect those changes, rerun Steps 1–4,
and commit them as:

```bash
git add docs/superpowers/specs/2026-07-25-demix-reproduction-design.md \
  docs/superpowers/plans/2026-07-25-demix-evidence-integrity-remediation.md \
  submissions/decouple-searching-from-training-scaling-data-mixing-via-model-merging-for-large-language-model-pre-training
git commit -m "chore(demix): finalize evidence verification"
```

If no files changed, do not create an empty commit.
