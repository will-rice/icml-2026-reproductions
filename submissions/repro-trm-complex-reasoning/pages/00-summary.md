# Evidence Summary

The reproduction checks the released TRM repository, arXiv source bundle, and
Hugging Face artifacts at immutable revisions. It confirms that the repository
and paper source expose the four ME2 dimensions, that the released
`dag_construction` package can build a DAG with continue, backtrack, and merge
actions under a deterministic local client, and that the TRM-Preference and
TRM-8B artifacts expose pairwise preference data and reward-model metadata.

The reward-model claim is deliberately marked partial: the public artifacts
include the dataset, sequence-classification reward model metadata, validation
results, and a training command that references the pairwise train/test files,
but the root `train.py` invoked by `train_rm.sh` is not present in the pinned
repository.
