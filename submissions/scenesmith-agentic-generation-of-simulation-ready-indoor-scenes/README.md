---
title: SceneSmith Reproduction
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
license: apache-2.0
tags:
  - icml2026-repro
  - paper-WwS8CTpUA6
---

# SceneSmith Reproduction Evidence

This project contains an independently executable evidence harness for
`SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes`.

The upstream source snapshot is pinned to
`nepfaff/scenesmith@67cc408fd38334b4a926efef45e284302ed5055b`.

Run:

```sh
python generate_evidence.py --output evidence/scenesmith_results.json
python -m pytest tests -q
```

The harness verifies released-source support for the pipeline, agent roles,
asset integration, and robot-evaluation modules. It marks paper-scale generation
metrics and the user study as unavailable because the bundled public artifacts do
not include the records needed to recompute those values.
