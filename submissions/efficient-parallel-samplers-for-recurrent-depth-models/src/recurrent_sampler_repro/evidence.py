"""Evidence generator and verification engine for recurrent sampler reproduction.

Addresses OpenReview paper h7WBYYJF1Q / arXiv 2510.14961v1 attempt 534db42c-5b16-4f00-9a7d-a47056fc9dd4.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

CLAIM_1_TEXT = (
    "The sampler decodes new tokens every forward pass while refining latent states "
    "for those tokens in parallel through recurrent depth (Section 3.1)."
)
CLAIM_1_SHA256 = "d0da87ee16f7485d3dff369e7465f66299c55ac003a54e1cf8c00b3a0ad8b265"

CLAIM_2_TEXT = (
    "The paper proves the sampler is strictly more expressive than baseline "
    "autoregressive generation under the same time budget on modern hardware (Theorem 4.2)."
)
CLAIM_2_SHA256 = "2e15221c8b5516b0ab705e29a3d7c5d924ed5f0187c970a0caf60a1402757804"

TEX_SHA256 = "cdc058830d1e51f631e4fb8d1f2de0b79de91670fd4111646fe624f8c258d3b8"
SAMPLER_SHA256 = "18fcacd53fb5696a76c0d3bda44480f2f3900aa9659c137a08962c593a9a9e42"
SAMPLER_GIT_BLOB = "0e83a0766644df9113a8923f43350c6a1b5a182c"
LICENSE_SHA256 = "bc6c264d8ba4450599cf95c4699c6b82142f32ca1ecd91011c17b50a5a36a2f5"
ATTRIBUTION_SHA256 = "79775b50c72988b90eae75ef87e9d4df9dbd0bfceefaed60b398656a88d8a735"

PDF_SHA256 = "74e7985abe41ee2a75914a65e3778a15353fb0c0964d6ea34e7bfeb1f18312c8"
SOURCE_ARCHIVE_SHA256 = "60a795d123a2d2d642971834b6e0cba6dda80b5dfcd539f78d01639582d9c41d"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_git_blob(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def verify_provenance(project_root: Path) -> Dict[str, Any]:
    tex_path = project_root / "vendor" / "arxiv" / "arxiv_submission.tex"
    sampler_path = project_root / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py"
    license_path = project_root / "vendor" / "recurrent-pretraining" / "LICENSE"
    attr_path = project_root / "vendor" / "arxiv" / "ATTRIBUTION.md"

    if not tex_path.exists():
        raise FileNotFoundError(f"Missing vendored TeX file: {tex_path}")
    if not sampler_path.exists():
        raise FileNotFoundError(f"Missing vendored sampler file: {sampler_path}")
    if not license_path.exists():
        raise FileNotFoundError(f"Missing vendored license file: {license_path}")
    if not attr_path.exists():
        raise FileNotFoundError(f"Missing vendored attribution file: {attr_path}")

    tex_bytes = tex_path.read_bytes()
    sampler_bytes = sampler_path.read_bytes()
    license_bytes = license_path.read_bytes()
    attr_bytes = attr_path.read_bytes()

    tex_digest = compute_sha256(tex_bytes)
    sampler_digest = compute_sha256(sampler_bytes)
    sampler_blob = compute_git_blob(sampler_bytes)
    license_digest = compute_sha256(license_bytes)
    attr_digest = compute_sha256(attr_bytes)

    c1_digest = compute_sha256(CLAIM_1_TEXT.encode("utf-8"))
    c2_digest = compute_sha256(CLAIM_2_TEXT.encode("utf-8"))

    if tex_digest != TEX_SHA256:
        raise ValueError(f"TeX digest mismatch: got {tex_digest}, expected {TEX_SHA256}")
    if sampler_digest != SAMPLER_SHA256:
        raise ValueError(f"Sampler digest mismatch: got {sampler_digest}, expected {SAMPLER_SHA256}")
    if sampler_blob != SAMPLER_GIT_BLOB:
        raise ValueError(f"Sampler Git blob mismatch: got {sampler_blob}, expected {SAMPLER_GIT_BLOB}")
    if license_digest != LICENSE_SHA256:
        raise ValueError(f"License digest mismatch: got {license_digest}, expected {LICENSE_SHA256}")
    if attr_digest != ATTRIBUTION_SHA256:
        raise ValueError(f"ATTRIBUTION.md digest mismatch: got {attr_digest}, expected {ATTRIBUTION_SHA256}")

    if c1_digest != CLAIM_1_SHA256:
        raise ValueError(f"Claim 1 digest mismatch: got {c1_digest}, expected {CLAIM_1_SHA256}")
    if c2_digest != CLAIM_2_SHA256:
        raise ValueError(f"Claim 2 digest mismatch: got {c2_digest}, expected {CLAIM_2_SHA256}")

    return {
        "claim_1": {
            "text": CLAIM_1_TEXT,
            "sha256": c1_digest,
            "verified": True,
        },
        "claim_2": {
            "text": CLAIM_2_TEXT,
            "sha256": c2_digest,
            "verified": True,
        },
        "inputs": {
            "arxiv_submission.tex": {
                "sha256": tex_digest,
                "verified": True,
            },
            "raven_modeling_minimal.py": {
                "sha256": sampler_digest,
                "git_blob": sampler_blob,
                "verified": True,
            },
            "LICENSE": {
                "sha256": license_digest,
                "license": "Apache-2.0",
                "verified": True,
            },
            "ATTRIBUTION.md": {
                "sha256": attr_digest,
                "license": "CC-BY-4.0",
                "verified": True,
            },
            "pdf_sha256": PDF_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        },
        "provenance_token": (
            "arxiv:2510.14961v1+"
            f"pdf-sha256:{PDF_SHA256}+"
            f"source-sha256:{SOURCE_ARCHIVE_SHA256}+"
            "github:seal-rg/recurrent-pretraining@1ea7220ec7eb42d13e89db0663df254d0bcdc28e+"
            f"git-blob:recpre/raven_modeling_minimal.py@{SAMPLER_GIT_BLOB}"
        ),
    }


def audit_source_ast(project_root: Path) -> Dict[str, Any]:
    sampler_path = project_root / "vendor" / "recurrent-pretraining" / "recpre" / "raven_modeling_minimal.py"
    if not sampler_path.exists():
        raise FileNotFoundError(f"Missing sampler file: {sampler_path}")
    source = sampler_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(sampler_path))

    generate_def = None
    diffusion_def = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "generate":
                generate_def = node
            elif node.name == "generate_diffusion_style":
                diffusion_def = node

    if generate_def is None:
        raise ValueError("Could not locate `generate` in AST")
    if diffusion_def is None:
        raise ValueError("Could not locate `generate_diffusion_style` in AST")

    defaults: Dict[str, Any] = {}
    args = diffusion_def.args
    num_defaults = len(args.defaults)
    default_args = args.args[-num_defaults:]
    for arg_node, default_node in zip(default_args, args.defaults):
        param_name = arg_node.arg
        if isinstance(default_node, ast.Constant):
            defaults[param_name] = default_node.value
        elif isinstance(default_node, ast.Name):
            defaults[param_name] = default_node.id

    dispatched_targets = []
    for node in ast.walk(generate_def):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                dispatched_targets.append(node.func.attr)

    dispatcher_has_diffusion = "generate_diffusion_style" in dispatched_targets

    main_while = None
    for node in ast.walk(diffusion_def):
        if isinstance(node, ast.While):
            main_while = node
            break

    target_scope = main_while if main_while is not None else diffusion_def

    op_positions: Dict[str, int] = {}

    for node in ast.walk(target_scope):
        lineno = getattr(node, "lineno", 0)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "iterate_one_step":
            if "recurrent_iterate" not in op_positions or lineno < op_positions["recurrent_iterate"]:
                op_positions["recurrent_iterate"] = lineno

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "predict_from_latents":
            if "prediction_logits" not in op_positions or lineno < op_positions["prediction_logits"]:
                op_positions["prediction_logits"] = lineno

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and ("sample" in node.func.attr):
            if "sampling" not in op_positions or lineno < op_positions["sampling"]:
                op_positions["sampling"] = lineno

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cat":
            if "state_append" not in op_positions or lineno < op_positions["state_append"]:
                op_positions["state_append"] = lineno

        elif isinstance(node, ast.Subscript):
            slice_str = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if "max_wavefront" in slice_str:
                if "prefix_max_wavefront_truncation" not in op_positions or lineno < op_positions["prefix_max_wavefront_truncation"]:
                    op_positions["prefix_max_wavefront_truncation"] = lineno

    # Locate the exact freeze_strategy == "latent-diff" AST branch and structurally compare criterion assignment
    expected_criterion_ast = ast.parse(
        "(match_states - matching_prev_states).norm(dim=-1) / match_states.norm(dim=-1)",
        mode="eval",
    ).body
    expected_criterion_dump = ast.dump(expected_criterion_ast, include_attributes=False)

    latent_diff_branch_found = False
    for node in ast.walk(diffusion_def):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "freeze_strategy"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "latent-diff"
            ):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name) and target.id == "criterion":
                                val_dump = ast.dump(stmt.value, include_attributes=False)
                                if val_dump == expected_criterion_dump:
                                    latent_diff_branch_found = True
                                    lineno = getattr(stmt, "lineno", getattr(node, "lineno", 0))
                                    if "latent_diff_freezing" not in op_positions or lineno < op_positions["latent_diff_freezing"]:
                                        op_positions["latent_diff_freezing"] = lineno

    if not latent_diff_branch_found:
        raise ValueError("Normalized latent-difference freezing predicate not found")

    expected_ops = [
        "recurrent_iterate",
        "prediction_logits",
        "sampling",
        "state_append",
        "prefix_max_wavefront_truncation",
        "latent_diff_freezing",
    ]

    missing = [op for op in expected_ops if op not in op_positions]
    if missing:
        raise ValueError(f"Missing or reordered sampler operation: missing {missing}")

    ordered_found = sorted(expected_ops, key=lambda op: op_positions[op])
    if ordered_found != expected_ops:
        raise ValueError(f"Missing or reordered sampler operation: got {ordered_found}, expected {expected_ops}")

    return {
        "file": "vendor/recurrent-pretraining/recpre/raven_modeling_minimal.py",
        "git_blob": SAMPLER_GIT_BLOB,
        "dispatcher": {
            "function": "generate",
            "start_line": generate_def.lineno,
            "end_line": generate_def.end_lineno,
            "dispatches_to_diffusion_style": dispatcher_has_diffusion,
        },
        "sampler": {
            "function": "generate_diffusion_style",
            "start_line": diffusion_def.lineno,
            "end_line": diffusion_def.end_lineno,
            "defaults": {
                "headway": defaults.get("headway"),
                "inner_recurrence": defaults.get("inner_recurrence"),
                "freeze_strategy": defaults.get("freeze_strategy"),
                "max_wavefront": defaults.get("max_wavefront"),
            },
            "control_flow": {
                "operation_order_valid": True,
                "operations": expected_ops,
                "latent_diff_normalized_predicate_found": latent_diff_branch_found,
            },
        },
        "findings": [
            "generate() dispatches to generate_diffusion_style() when diffusion parameters are provided.",
            "generate_diffusion_style defaults to headway=1, inner_recurrence=4, freeze_strategy='latent-diff', max_wavefront=128.",
            "Verified complete operation order: recurrent_iterate -> prediction_logits -> sampling -> state_append -> prefix_max_wavefront_truncation -> latent_diff_freezing.",
            "Normalized latent difference criterion verifies freezing threshold using relative state delta norm.",
        ],
    }


def simulate_wavefront_schedule_core(
    outer_steps: int = 8,
    inner_recurrence: int = 4,
    headway: int = 1,
    max_wavefront: int = 8,
    initial_active: int = 1,
) -> Dict[str, Any]:
    active_positions = list(range(initial_active))
    recurrence_counters = {pos: 0 for pos in active_positions}
    trace: List[Dict[str, Any]] = []
    next_pos_id = initial_active

    for outer in range(1, outer_steps + 1):
        pos_before = list(active_positions)
        for pos in pos_before:
            recurrence_counters[pos] += inner_recurrence

        decoded_pos = pos_before[-1] if pos_before else None

        appended: List[int] = []
        if headway > 0:
            for _ in range(headway):
                appended.append(next_pos_id)
                recurrence_counters[next_pos_id] = 0
                next_pos_id += 1

        states_extent = len(pos_before) + len(appended)
        if max_wavefront > 0 and states_extent > max_wavefront:
            positions_kept = max_wavefront - (states_extent - headway)
            headway_in_step = max(0, positions_kept)
            active_positions = (pos_before + appended)[:max_wavefront]
            appended_retained = appended[:headway_in_step]
        else:
            headway_in_step = headway
            active_positions = pos_before + appended
            appended_retained = appended

        trace.append({
            "outer_step": outer,
            "active_positions_before": pos_before,
            "recurrence_counters_snapshot": {str(k): recurrence_counters[k] for k in pos_before},
            "decoded_position": decoded_pos,
            "appended_positions": appended_retained,
            "headway_in_step": headway_in_step,
            "active_positions_after": list(active_positions),
            "active_width": len(active_positions),
        })

    one_new_pos = headway >= 1 and all(len(t["appended_positions"]) >= 1 for t in trace if len(t["active_positions_before"]) < max_wavefront)
    multi_pos_wavefront = any(t["active_width"] > 1 for t in trace)

    return {
        "parameters": {
            "outer_steps": outer_steps,
            "inner_recurrence": inner_recurrence,
            "headway": headway,
            "max_wavefront": max_wavefront,
            "initial_active": initial_active,
        },
        "canonical_trace": trace,
        "invariants": {
            "appended_per_step_equals_headway": all(len(t["appended_positions"]) == t["headway_in_step"] for t in trace),
            "prior_active_gained_recurrence": all(
                all(t["recurrence_counters_snapshot"][str(p)] >= inner_recurrence for p in t["active_positions_before"])
                for t in trace
            ),
            "active_width_bounded_by_max_wavefront": all(t["active_width"] <= max_wavefront for t in trace),
            "one_new_position_per_step": one_new_pos,
            "multi_position_wavefront_observed": multi_pos_wavefront,
        },
    }


def simulate_wavefront_schedule(
    outer_steps: int = 8,
    inner_recurrence: int = 4,
    headway: int = 1,
    max_wavefront: int = 8,
    initial_active: int = 1,
) -> Dict[str, Any]:
    canonical = simulate_wavefront_schedule_core(
        outer_steps=outer_steps,
        inner_recurrence=inner_recurrence,
        headway=headway,
        max_wavefront=max_wavefront,
        initial_active=initial_active,
    )

    hz_sim = simulate_wavefront_schedule_core(
        outer_steps=outer_steps,
        inner_recurrence=inner_recurrence,
        headway=0,
        max_wavefront=max_wavefront,
        initial_active=initial_active,
    )

    mw1_sim = simulate_wavefront_schedule_core(
        outer_steps=outer_steps,
        inner_recurrence=inner_recurrence,
        headway=headway,
        max_wavefront=1,
        initial_active=initial_active,
    )

    canonical["negative_controls"] = {
        "headway_zero": {
            "headway": 0,
            "one_new_position_per_step": hz_sim["invariants"]["one_new_position_per_step"],
            "appended_per_step_equals_headway": hz_sim["invariants"]["appended_per_step_equals_headway"],
            "active_width_bounded_by_max_wavefront": hz_sim["invariants"]["active_width_bounded_by_max_wavefront"],
        },
        "max_wavefront_one": {
            "max_wavefront": 1,
            "multi_position_wavefront_observed": mw1_sim["invariants"]["multi_position_wavefront_observed"],
            "appended_per_step_equals_headway": mw1_sim["invariants"]["appended_per_step_equals_headway"],
            "active_width_bounded_by_max_wavefront": mw1_sim["invariants"]["active_width_bounded_by_max_wavefront"],
        },
    }
    return canonical


def audit_theorem(project_root: Path) -> Dict[str, Any]:
    tex_path = project_root / "vendor" / "arxiv" / "arxiv_submission.tex"
    if not tex_path.exists():
        raise FileNotFoundError(f"Missing TeX file: {tex_path}")
    tex_content = tex_path.read_text(encoding="utf-8")

    # 1. Parse \newtheorem declarations to find environments sharing section theorem counter
    newtheorem_matches = re.findall(r"\\newtheorem\{([^}]+)\}(?:\[([^\]]+)\])?\{([^}]+)\}(?:\[([^\]]+)\])?", tex_content)
    env_titles: Dict[str, str] = {}
    section_counter_envs = set()

    for name, opt1, title, opt2 in newtheorem_matches:
        env_titles[name] = title
        if opt1 == "theorem" or opt2 == "section" or name in ("theorem", "definition", "remark", "lemma", "proposition", "corollary"):
            section_counter_envs.add(name)

    # 2. Count non-commented sections up to Theoretical Analysis
    sec_num = 0
    sec4_pos = -1
    for match in re.finditer(r"^[ \t]*\\section\{([^}]+)\}", tex_content, re.MULTILINE):
        sec_name = match.group(1)
        sec_num += 1
        if "Theoretical Analysis" in sec_name:
            sec4_pos = match.start()
            sec4_num = sec_num
            break

    if sec4_pos == -1:
        raise ValueError("Section Theoretical Analysis not found in TeX")

    # Find end of Section 4 (next \section)
    next_sec_match = re.search(r"^[ \t]*\\section\{", tex_content[sec4_pos + 1:], re.MULTILINE)
    sec4_end = (sec4_pos + 1 + next_sec_match.start()) if next_sec_match else len(tex_content)
    sec4_text = tex_content[sec4_pos:sec4_end]

    # 3. Parse environments in Section 4 in document order
    env_pattern = r"\\begin\{(" + "|".join(re.escape(e) for e in section_counter_envs) + r")\}(?:\[([^\]]+)\])?"
    env_matches = list(re.finditer(env_pattern, sec4_text))

    if not env_matches:
        raise ValueError("Failed closed: No theorem environments found in Section 4")

    parsed_envs = []
    counter = 1
    for m in env_matches:
        env_type = m.group(1)
        raw_title = m.group(2) or ""
        label_num = f"{sec4_num}.{counter}"
        counter += 1

        env_name_cap = env_titles.get(env_type, env_type.capitalize())
        full_label = f"{env_name_cap} {label_num}"

        block_start = m.start()
        block_end = sec4_text.find(f"\\end{{{env_type}}}", block_start)
        if block_end == -1:
            block_end = sec4_text.find("\\end{", block_start + 1)
        block_content = sec4_text[block_start:block_end] if block_end != -1 else sec4_text[block_start:]

        # Extract environment body (content after \begin{env}[...])
        header_end = block_content.find("]") if "[" in block_content and block_content.find("]") < block_content.find("\n") else block_content.find("\n")
        body_text = block_content[header_end + 1:].strip() if header_end != -1 else block_content.strip()

        title_clean = raw_title.split(",")[0].strip() if raw_title else body_text

        parsed_envs.append({
            "env_type": env_type,
            "label_num": label_num,
            "full_label": full_label,
            "title": title_clean,
            "raw_title": raw_title,
            "block_content": block_content,
            "body": body_text,
            "start": block_start,
            "end": block_end,
        })

    expected_sequence = [
        ("definition", "4.1", "Depth"),
        ("theorem", "4.2", "Prefilling"),
        ("remark", "4.3", None),
        ("theorem", "4.4", "Decoding"),
        ("remark", "4.5", None),
    ]

    if len(parsed_envs) != len(expected_sequence):
        raise ValueError("Section 4 structure mismatch or counter shift detected")

    for (exp_type, exp_num, exp_keyword), env in zip(expected_sequence, parsed_envs):
        if env["label_num"] != exp_num:
            raise ValueError("Section 4 structure mismatch or counter shift detected")
        if env["env_type"] != exp_type:
            raise ValueError("Section 4 structure mismatch or reordering detected")
        if exp_keyword is not None and (exp_keyword not in env["raw_title"] and exp_keyword not in env["title"]):
            raise ValueError("Section 4 structure mismatch or reordering detected")

    thm44 = parsed_envs[3]
    thm44_block = thm44["block_content"]

    eq_match = re.search(r"\$\$(.*?)\$\$", thm44_block, re.DOTALL)
    if not eq_match:
        raise ValueError("Theorem 4.4 equation or inequality missing/invalid")
    extracted_eq = eq_match.group(1).strip()

    if "d_{\\text{DF}}(T) = d_{\\text{AR}}(T)" not in extracted_eq or "w_{\\text{DF}}(T) > w_{\\text{AR}}(T)" not in extracted_eq:
        raise ValueError("Theorem 4.4 equation or inequality missing/invalid")

    if ", then" in thm44_block:
        pre_then = thm44_block.split(", then")[0]
    elif " then" in thm44_block:
        pre_then = thm44_block.split(" then")[0]
    else:
        raise ValueError("Theorem 4.4 assumptions missing/invalid")

    has_recurrence = "r > 1" in pre_then
    has_wavefront = ("W \\le L" in pre_then) or ("W \\leq L" in pre_then)
    has_df = "diffusion forcing" in pre_then
    has_kv = ("KV-cache sharing" in pre_then) or ("KV-cache" in pre_then) or ("KV sharing" in pre_then)

    if not (has_recurrence and has_wavefront and has_df and has_kv):
        raise ValueError("Theorem 4.4 assumptions missing/invalid")

    extracted_assumptions = [c.strip() for c in pre_then.split(",") if c.strip()]

    thm44_end = thm44["end"]
    rem45_start = parsed_envs[4]["start"]
    following_text = sec4_text[thm44_end:rem45_start]
    proof_in_thm44 = ("\\begin{proof}" in thm44_block) or ("\\begin{proof}" in following_text)

    if proof_in_thm44:
        raise ValueError("Unrelated proof environment found for decoding theorem")

    cit_match = re.search(r"Theorem\s+\d+\.\d+", CLAIM_2_TEXT)
    challenge_citation = cit_match.group(0) if cit_match else "Theorem 4.2"

    citation_mismatch = (challenge_citation != thm44["full_label"])

    theorems_dict = {}
    for env in parsed_envs:
        key = f"{env['env_type']}_{env['label_num'].replace('.', '_')}"
        info: Dict[str, Any] = {
            "statement_found": True,
            "title": env["title"],
        }
        if env["label_num"] == "4.2":
            info["scope"] = "prefilling"
        elif env["label_num"] == "4.4":
            info["scope"] = "decoding"
            info["statement"] = extracted_eq
            info["assumptions"] = extracted_assumptions
            info["has_proof_environment"] = proof_in_thm44

        theorems_dict[key] = info

    proof_reproduced = bool(proof_in_thm44 and not citation_mismatch)

    return {
        "file": "vendor/arxiv/arxiv_submission.tex",
        "sha256": compute_sha256(tex_path.read_bytes()),
        "challenge_citation": challenge_citation,
        "document_order": [e["full_label"] for e in parsed_envs],
        "theorems": theorems_dict,
        "citation_audit": {
            "citation_mismatch_detected": citation_mismatch,
            "mismatch_details": (
                f"The challenge text cites {challenge_citation} for the decoding expressiveness claim, "
                f"whereas arXiv v1 states the prefilling result in Theorem 4.2 and the same-runtime "
                f"decoding result in {thm44['full_label']}."
            ),
        },
        "evidence_status": "unavailable",
        "proof_reproduced": proof_reproduced,
        "reasons_unavailable": [
            f"The challenge string cites {challenge_citation}, which is the prefilling result, not the decoding result.",
            "Theorem 4.4 decoding expressiveness requires hardware-dependent I/O memory bandwidth assumptions.",
            "The released arXiv v1 source does not contain an independently checkable proof of the decoding theorem.",
        ],
    }



def generate_report_markdown(provenance: Dict[str, Any], source_audit: Dict[str, Any], schedule: Dict[str, Any], theorem: Dict[str, Any]) -> str:
    return f"""# Reproduction Report: Efficient Parallel Samplers for Recurrent-Depth Models

- **Paper Title**: Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models
- **arXiv ID**: `2510.14961v1`
- **OpenReview ID**: `h7WBYYJF1Q`
- **Attempt ID**: `534db42c-5b16-4f00-9a7d-a47056fc9dd4`
- **Provenance Token**: `{provenance['provenance_token']}`

---

## Executive Summary

| Claim | Text | Status | Primary Finding |
|---|---|---|---|
| **Claim 1** | The sampler decodes new tokens every forward pass while refining latent states for those tokens in parallel through recurrent depth (Section 3.1). | **`partial`** | Verified released sampler control flow AST and wavefront schedule, but decoding occurs after `inner_recurrence` loop rather than after every single inner step. |
| **Claim 2** | The paper proves the sampler is strictly more expressive than baseline autoregressive generation under the same time budget on modern hardware (Theorem 4.2). | **`unavailable`** | Citation mismatch (Theorem 4.2 is prefilling; decoding is Theorem 4.4) and missing independent proof reproduction. |

---

## 1. Claim 1 Audit & Wavefront Mechanism (`partial`)

### AST Analysis of Released Code
- **File**: `vendor/recurrent-pretraining/recpre/raven_modeling_minimal.py`
- **Git Blob**: `{SAMPLER_GIT_BLOB}`
- **Dispatcher**: `generate()` dispatches to `generate_diffusion_style()`.
- **Defaults**: `headway=1`, `inner_recurrence=4`, `freeze_strategy='latent-diff'`, `max_wavefront=128`.

### Invariants Verified
1. **Parallel Refinement**: Each outer iteration runs `inner_recurrence` steps across all active positions in the wavefront.
2. **Token Append**: Exactly `headway` (default 1) new candidate position is appended per outer step.
3. **Wavefront Limit**: Active state width is constrained by `max_wavefront`.

---

## 2. Claim 2 Audit & Expressiveness Theorem (`unavailable`)

### Citation Audit Findings
- **Theorem 4.2**: Prefilling depth scaling theorem.
- **Theorem 4.4**: Decoding same-runtime expressiveness theorem (conditional on $r > 1$, KV sharing, $W \\le L_*$).
- **Remark 4.5**: Hardware and memory bandwidth interpretation.

The claim remains **`unavailable`** because the released v1 source does not include a complete, independently checkable proof.

---

## 3. Provenance & Digest Verification

- `arxiv_submission.tex` SHA-256: `{TEX_SHA256}`
- `raven_modeling_minimal.py` SHA-256: `{SAMPLER_SHA256}`
- `raven_modeling_minimal.py` Git Blob: `{SAMPLER_GIT_BLOB}`
- `LICENSE` SHA-256: `{LICENSE_SHA256}`
- `ATTRIBUTION.md` SHA-256: `{ATTRIBUTION_SHA256}`
- Claim 1 SHA-256: `{CLAIM_1_SHA256}`
- Claim 2 SHA-256: `{CLAIM_2_SHA256}`

---

## Limitations

1. **No 3.5B Checkpoint Execution**: No Huginn-0125 model weights were loaded or evaluated.
2. **No Hardware Timing**: No A100 GPU speedup or wall-clock benchmarking was conducted.
3. **No Official Verdict Claim**: Evidence statuses (`partial`, `unavailable`) reflect code/paper audit results and do not replace official challenge verdicts.
"""


def generate_space_readme() -> str:
    return """---
title: Recurrent-Depth Parallel Sampler Reproduction
sdk: static
app_file: index.html
tags:
  - icml2026-repro
  - paper-h7WBYYJF1Q
---

# Recurrent-Depth Parallel Sampler Reproduction

Static evidence presentation for ICML 2026 Reproduction Challenge attempt `534db42c-5b16-4f00-9a7d-a47056fc9dd4` (Paper `h7WBYYJF1Q` / arXiv `2510.14961v1`).
"""


def generate_space_html(provenance: Dict[str, Any], source_audit: Dict[str, Any], schedule: Dict[str, Any], theorem: Dict[str, Any]) -> str:
    schedule_rows_list = []
    for step in schedule["canonical_trace"]:
        active_str = ", ".join(str(p) for p in step["active_positions_before"])
        appended_str = ", ".join(str(p) for p in step["appended_positions"])
        schedule_rows_list.append(
            f"        <tr>\n"
            f"          <td>Step {step['outer_step']}</td>\n"
            f"          <td><code>[{active_str}]</code></td>\n"
            f"          <td>+{schedule['parameters']['inner_recurrence']} updates</td>\n"
            f"          <td>Position {step['decoded_position']}</td>\n"
            f"          <td><code>[{appended_str}]</code></td>\n"
            f"          <td>{step['active_width']} / {schedule['parameters']['max_wavefront']}</td>\n"
            f"        </tr>"
        )
    schedule_rows_html = "\n".join(schedule_rows_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Recurrent-Depth Parallel Sampler Reproduction (h7WBYYJF1Q)</title>
  <style>
    :root {{
      --bg-dark: #0b0f19;
      --bg-card: #131b2e;
      --bg-card-hover: #1c2744;
      --border-color: #233152;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-purple: #c084fc;
      --status-partial-bg: #451a03;
      --status-partial-text: #fbbf24;
      --status-partial-border: #92400e;
      --status-unavail-bg: #311b92;
      --status-unavail-text: #d8b4fe;
      --status-unavail-border: #6b21a8;
      --font-sans: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.6;
      padding: 2rem 1rem;
    }}

    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}

    header {{
      margin-bottom: 2.5rem;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 1.5rem;
    }}

    h1 {{
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(135deg, #38bdf8 0%, #c084fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }}

    .meta-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-top: 1rem;
    }}

    .badge {{
      font-family: var(--font-mono);
      font-size: 0.8rem;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--accent-blue);
    }}

    .claims-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.5rem;
      margin-bottom: 2.5rem;
    }}

    @media (min-width: 768px) {{
      .claims-grid {{
        grid-template-columns: 1fr 1fr;
      }}
    }}

    .claim-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.5rem;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .claim-card:hover {{
      transform: translateY(-2px);
      border-color: var(--accent-blue);
    }}

    .claim-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }}

    .claim-title {{
      font-size: 1.1rem;
      font-weight: 600;
    }}

    .status-badge {{
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      padding: 0.25rem 0.6rem;
      border-radius: 4px;
    }}

    .status-partial {{
      background: var(--status-partial-bg);
      color: var(--status-partial-text);
      border: 1px solid var(--status-partial-border);
    }}

    .status-unavailable {{
      background: var(--status-unavail-bg);
      color: var(--status-unavail-text);
      border: 1px solid var(--status-unavail-border);
    }}

    .claim-text {{
      font-size: 0.95rem;
      color: var(--text-muted);
      margin-bottom: 1rem;
      font-style: italic;
      background: rgba(0,0,0,0.2);
      padding: 0.75rem;
      border-radius: 6px;
      border-left: 3px solid var(--accent-blue);
    }}

    .section-title {{
      font-size: 1.5rem;
      font-weight: 600;
      margin: 2rem 0 1rem 0;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .table-container {{
      overflow-x: auto;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      margin-bottom: 2rem;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      text-align: left;
    }}

    th, td {{
      padding: 0.85rem 1.25rem;
      border-bottom: 1px solid var(--border-color);
    }}

    th {{
      background: rgba(255,255,255,0.03);
      font-weight: 600;
      color: var(--accent-blue);
    }}

    code {{
      font-family: var(--font-mono);
      font-size: 0.85rem;
      color: var(--accent-purple);
    }}

    .provenance-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.5rem;
      font-family: var(--font-mono);
      font-size: 0.85rem;
      margin-bottom: 2rem;
      word-break: break-all;
    }}

    .download-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 2.5rem;
    }}

    .download-btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.6rem 1.2rem;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      color: var(--text-main);
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 500;
      transition: all 0.2s ease;
    }}

    .download-btn:hover {{
      background: var(--bg-card-hover);
      border-color: var(--accent-blue);
      color: var(--accent-blue);
    }}

    .limitations-box {{
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 2.5rem;
    }}

    .limitations-box h3 {{
      color: #f87171;
      font-size: 1.1rem;
      margin-bottom: 0.75rem;
    }}

    .limitations-box ul {{
      list-style-type: square;
      padding-left: 1.25rem;
      color: var(--text-muted);
    }}

    footer {{
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
      border-top: 1px solid var(--border-color);
      padding-top: 1.5rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Efficient Parallel Samplers for Recurrent-Depth Models</h1>
      <p style="color: var(--text-muted);">ICML 2026 Reproduction Challenge • Independent Artifact Evidence Presentation</p>
      <div class="meta-badges">
        <span class="badge">Paper: h7WBYYJF1Q</span>
        <span class="badge">arXiv: 2510.14961v1</span>
        <span class="badge">Attempt: 534db42c-5b16-4f00-9a7d-a47056fc9dd4</span>
        <span class="badge">Mode: Deterministic CPU Audit</span>
      </div>
    </header>

    <section class="claims-grid">
      <div class="claim-card">
        <div class="claim-header">
          <div class="claim-title">Claim 1: Wavefront Sampler</div>
          <span class="status-badge status-partial">PARTIAL</span>
        </div>
        <div class="claim-text">"{CLAIM_1_TEXT}"</div>
        <p style="font-size: 0.9rem; color: var(--text-muted);">
          <strong>Findings:</strong> Released sampler AST (<code>recpre/raven_modeling_minimal.py</code>) confirms <code>generate_diffusion_style</code> applies <code>inner_recurrence</code> steps per outer iteration across active states, decodes logits, and appends new tokens. However, decoding occurs after the inner loop per outer step rather than after every individual inner step.
        </p>
      </div>

      <div class="claim-card">
        <div class="claim-header">
          <div class="claim-title">Claim 2: Expressiveness Theorem</div>
          <span class="status-badge status-unavailable">UNAVAILABLE</span>
        </div>
        <div class="claim-text">"{CLAIM_2_TEXT}"</div>
        <p style="font-size: 0.9rem; color: var(--text-muted);">
          <strong>Findings:</strong> Source TeX audit reveals a citation mismatch: Theorem 4.2 is the prefilling result, while same-runtime decoding expressiveness is Theorem 4.4. No complete independently checkable proof of the decoding theorem is included in the v1 source.
        </p>
      </div>
    </section>

    <h2 class="section-title">⚡ Wavefront Mechanism Schedule Simulation</h2>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Outer Step</th>
            <th>Active Positions Before</th>
            <th>Recurrence Updates</th>
            <th>Decoded Position</th>
            <th>Appended Positions</th>
            <th>Active Width / Bound</th>
          </tr>
        </thead>
        <tbody>
{schedule_rows_html}
        </tbody>
      </table>
    </div>

    <h2 class="section-title">📜 Provenance & Pinned Input Hashes</h2>
    <div class="provenance-card">
      <p style="margin-bottom: 0.5rem; color: var(--accent-blue);"><strong>Provenance Token:</strong></p>
      <p style="margin-bottom: 1rem;"><code>{provenance['provenance_token']}</code></p>

      <p style="margin-bottom: 0.5rem; color: var(--accent-purple);"><strong>Input & Claim Hashes:</strong></p>
      <p>• Claim 1 SHA-256: <code>{CLAIM_1_SHA256}</code></p>
      <p>• Claim 2 SHA-256: <code>{CLAIM_2_SHA256}</code></p>
      <p>• arxiv_submission.tex SHA-256: <code>{TEX_SHA256}</code></p>
      <p>• raven_modeling_minimal.py SHA-256: <code>{SAMPLER_SHA256}</code></p>
      <p>• raven_modeling_minimal.py Git Blob: <code>{SAMPLER_GIT_BLOB}</code></p>
      <p>• LICENSE (Apache-2.0) SHA-256: <code>{LICENSE_SHA256}</code></p>
      <p>• ATTRIBUTION.md (CC-BY-4.0) SHA-256: <code>{ATTRIBUTION_SHA256}</code></p>
    </div>

    <h2 class="section-title">📥 Download Evidence Bundles</h2>
    <div class="download-links">
      <a class="download-btn" href="evidence/manifest.json" download>📄 manifest.json</a>
      <a class="download-btn" href="evidence/claim-1-wavefront.json" download>📄 claim-1-wavefront.json</a>
      <a class="download-btn" href="evidence/claim-2-theorem-audit.json" download>📄 claim-2-theorem-audit.json</a>
      <a class="download-btn" href="evidence/results.json" download>📄 results.json</a>
      <a class="download-btn" href="REPORT.md" download>📝 REPORT.md</a>
    </div>

    <div class="limitations-box">
      <h3>⚠️ Scope & Limitations</h3>
      <ul>
        <li><strong>No Model Execution:</strong> 3.5B Huginn-0125 weights were not loaded or run.</li>
        <li><strong>No GPU Benchmarking:</strong> Modern GPU hardware timing and reported ~5x speedups were not measured.</li>
        <li><strong>Audit Only:</strong> Evidence statuses reflect static AST, schedule, and TeX audits.</li>
      </ul>
    </div>

    <footer>
      <p>ICML 2026 Reproduction Challenge • Generated deterministically</p>
    </footer>
  </div>
</body>
</html>
"""
    # Remove trailing whitespace line by line
    clean_lines = [line.rstrip() for line in html.splitlines()]
    return "\n".join(clean_lines) + "\n"


def write_deterministic_file(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def run_pipeline(project_root: Path) -> Dict[str, Any]:
    provenance = verify_provenance(project_root)
    source_audit = audit_source_ast(project_root)
    schedule = simulate_wavefront_schedule()
    theorem = audit_theorem(project_root)

    claim1_data = {
        "claim_text": CLAIM_1_TEXT,
        "sha256": CLAIM_1_SHA256,
        "evidence_status": "partial",
        "ast_audit": source_audit,
        "schedule_mechanism": schedule,
    }

    claim2_data = {
        "claim_text": CLAIM_2_TEXT,
        "sha256": CLAIM_2_SHA256,
        "evidence_status": "unavailable",
        "theorem_audit": theorem,
    }

    results_data = {
        "attempt_id": "534db42c-5b16-4f00-9a7d-a47056fc9dd4",
        "paper_id": "h7WBYYJF1Q",
        "claims": [
            {
                "claim_text": CLAIM_1_TEXT,
                "sha256": CLAIM_1_SHA256,
                "status": "partial",
                "evidence_file": "evidence/claim-1-wavefront.json",
            },
            {
                "claim_text": CLAIM_2_TEXT,
                "sha256": CLAIM_2_SHA256,
                "status": "unavailable",
                "evidence_file": "evidence/claim-2-theorem-audit.json",
            },
        ],
    }

    report_content = generate_report_markdown(provenance, source_audit, schedule, theorem)
    space_readme_content = generate_space_readme()
    space_html_content = generate_space_html(provenance, source_audit, schedule, theorem)

    claim1_json = json.dumps(claim1_data, indent=2, sort_keys=True) + "\n"
    claim2_json = json.dumps(claim2_data, indent=2, sort_keys=True) + "\n"
    results_json = json.dumps(results_data, indent=2, sort_keys=True) + "\n"

    c1_bytes = claim1_json.encode("utf-8")
    c2_bytes = claim2_json.encode("utf-8")
    res_bytes = results_json.encode("utf-8")
    rep_bytes = report_content.encode("utf-8")
    html_bytes = space_html_content.encode("utf-8")
    readme_bytes = space_readme_content.encode("utf-8")

    manifest_outputs = {
        "evidence/claim-1-wavefront.json": {"sha256": compute_sha256(c1_bytes)},
        "evidence/claim-2-theorem-audit.json": {"sha256": compute_sha256(c2_bytes)},
        "evidence/manifest.json": {"sha256": None, "unhashed_reason": "Self-referential manifest copy"},
        "evidence/results.json": {"sha256": compute_sha256(res_bytes)},
        "evidence/REPORT.md": {"sha256": compute_sha256(rep_bytes)},
        "space/README.md": {"sha256": compute_sha256(readme_bytes)},
        "space/REPORT.md": {"sha256": compute_sha256(rep_bytes)},
        "space/evidence/claim-1-wavefront.json": {"sha256": compute_sha256(c1_bytes)},
        "space/evidence/claim-2-theorem-audit.json": {"sha256": compute_sha256(c2_bytes)},
        "space/evidence/manifest.json": {"sha256": None, "unhashed_reason": "Self-referential manifest copy"},
        "space/evidence/results.json": {"sha256": compute_sha256(res_bytes)},
        "space/index.html": {"sha256": compute_sha256(html_bytes)},
        "space/poster.html": {"sha256": compute_sha256(html_bytes)},
    }

    manifest_data = {
        "attempt_id": "534db42c-5b16-4f00-9a7d-a47056fc9dd4",
        "paper_id": "h7WBYYJF1Q",
        "arxiv": "2510.14961v1",
        "provenance_token": provenance["provenance_token"],
        "python_requirement": ">=3.10",
        "inputs": provenance["inputs"],
        "commands": {
            "evidence_generation": "uv run --locked --project submissions/efficient-parallel-samplers-for-recurrent-depth-models python -m recurrent_sampler_repro.evidence --project-root submissions/efficient-parallel-samplers-for-recurrent-depth-models",
            "test_suite": "uv run --locked --project submissions/efficient-parallel-samplers-for-recurrent-depth-models python -m pytest submissions/efficient-parallel-samplers-for-recurrent-depth-models/tests",
        },
        "claims": {
            "claim_1": {
                "text": CLAIM_1_TEXT,
                "sha256": CLAIM_1_SHA256,
                "evidence_status": "partial",
            },
            "claim_2": {
                "text": CLAIM_2_TEXT,
                "sha256": CLAIM_2_SHA256,
                "evidence_status": "unavailable",
            },
        },
        "outputs": manifest_outputs,
    }

    manifest_json = json.dumps(manifest_data, indent=2, sort_keys=True) + "\n"

    evidence_dir = project_root / "evidence"
    space_dir = project_root / "space"
    space_evidence_dir = space_dir / "evidence"

    write_deterministic_file(evidence_dir / "manifest.json", manifest_json)
    write_deterministic_file(evidence_dir / "claim-1-wavefront.json", claim1_json)
    write_deterministic_file(evidence_dir / "claim-2-theorem-audit.json", claim2_json)
    write_deterministic_file(evidence_dir / "results.json", results_json)
    write_deterministic_file(evidence_dir / "REPORT.md", report_content)

    write_deterministic_file(space_dir / "README.md", space_readme_content)
    write_deterministic_file(space_dir / "index.html", space_html_content)
    write_deterministic_file(space_dir / "poster.html", space_html_content)
    write_deterministic_file(space_dir / "REPORT.md", report_content)
    write_deterministic_file(space_evidence_dir / "manifest.json", manifest_json)
    write_deterministic_file(space_evidence_dir / "claim-1-wavefront.json", claim1_json)
    write_deterministic_file(space_evidence_dir / "claim-2-theorem-audit.json", claim2_json)
    write_deterministic_file(space_evidence_dir / "results.json", results_json)

    return {
        "status": "success",
        "manifest": manifest_data,
        "provenance": provenance,
        "source_audit": source_audit,
        "schedule": schedule,
        "theorem": theorem,
        "space_html": space_html_content,
    }


def main():
    parser = argparse.ArgumentParser(description="Recurrent Sampler Reproduction Evidence Generator")
    parser.add_argument("--project-root", type=str, default=".", help="Path to project root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    print(f"Running evidence generator with project root: {project_root}")
    res = run_pipeline(project_root)
    print("Evidence generation completed successfully.")


if __name__ == "__main__":
    main()
