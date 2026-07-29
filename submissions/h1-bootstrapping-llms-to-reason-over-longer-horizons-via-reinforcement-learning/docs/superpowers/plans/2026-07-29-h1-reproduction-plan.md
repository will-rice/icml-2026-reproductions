# Reproduction Plan: h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning

## Overview

Paper ID: `3BW15kSPfN`
Title: h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning
ArXiv: 2510.07312
Upstream Revision: `arxiv:2510.07312v1+github:Oxford-AI-Safety-Lab/h1@871e89d078202c7d9d18d0924bd76cf161cd6606`

## Target Claims

1. **Claim 1**: `h1 synthesizes long-horizon reasoning examples by chaining existing short-horizon GSM8K-style problems without new human or teacher-model annotations (Section 3).`
2. **Claim 2**: `The training recipe uses outcome-only RL with a curriculum that automatically increases composed problem horizon length (Section 3).`

## Scope and Plan

1. **Synthetic Chaining Mechanism Verification**:
   - Implement synthetic problem-chaining logic that combines short-horizon GSM8K-style sub-problems into longer composed questions.
   - Verify deterministic structure, state variables, and annotation-free chaining without requiring external teacher models.

2. **Outcome-Only RL & Curriculum Mechanics**:
   - Implement curriculum management logic tracking horizon depth and outcome reward scoring.
   - Validate that curriculum state transitions correctly increment problem horizon length based on rollout performance thresholds.

3. **Validation & Evidence Harness**:
   - Unit tests covering problem composition, outcome reward validation, and curriculum depth updates.
   - Clean static/CPU verification harness returning structured json evidence.
