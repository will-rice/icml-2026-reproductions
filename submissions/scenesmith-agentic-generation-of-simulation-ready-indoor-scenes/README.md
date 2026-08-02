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
