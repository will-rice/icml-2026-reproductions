# TerminalTraj Artifact Reproduction

This submission audits the released TerminalTraj artifacts against two selected challenge claims. It uses the pinned GitHub repository and Hugging Face dataset revisions to inspect the released repository/license manifest, dataset split metadata, and instance archive availability.

The evidence intentionally does not rerun model training, TerminalBench evaluation, or a large Docker build campaign. The public artifacts support a toy-level check of the claimed pipeline surface, while the full 32K Docker image and 50,733 trajectory claim is reported as unavailable from the released artifacts because the visible training dataset exposes 20,000 examples.
