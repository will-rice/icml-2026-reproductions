# RBench Inputs

The nine prompt manifests acquired into `prompts/` come from
`DAGroup-PKU/RBench` at commit
`6bdccf349ff5a8f68302428351e94f34ecd62450` and are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The upstream dataset
card is preserved as `UPSTREAM_README.md` after acquisition.

Run all pinned standalone acquisition commands in one shell so cleanup covers
every destination:

```bash
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT HUP INT TERM
git clone --filter=blob:none --no-checkout https://github.com/DAGroup-PKU/ReVidgen.git "$workdir/ReVidgen"
git -C "$workdir/ReVidgen" fetch --no-tags origin b03df27f0376faa148dcd8cd620a1989a32ca979
git -C "$workdir/ReVidgen" sparse-checkout init --cone
git -C "$workdir/ReVidgen" sparse-checkout set eval scripts
git -C "$workdir/ReVidgen" checkout --detach b03df27f0376faa148dcd8cd620a1989a32ca979
git -C "$workdir/ReVidgen" rev-parse --verify HEAD
hf download DAGroup-PKU/RBench --type dataset --revision 6bdccf349ff5a8f68302428351e94f34ecd62450 --include README.md --include 'prompts/*.json' --local-dir "$workdir/RBench"
hf download DAGroup-PKU/RBench-Leaderboard --type space --revision 6b66282843a5d863af4271fb07ba1641d1d33334 --include README.md --include app.py --include utils.py --include leaderboard.json --include requirements.txt --local-dir "$workdir/RBench-Leaderboard-paper-era"
hf download DAGroup-PKU/RBench-Leaderboard --type space --revision 5dd6d55e454e22dbf7bd34ea5fbbeda5bc0f9b07 --include README.md --include app.py --include utils.py --include leaderboard.json --include leaderboard_qwen.json --include requirements.txt --local-dir "$workdir/RBench-Leaderboard-current"
```

ReVidgen and leaderboard Space files are not redistributable from the pinned
license evidence and remain only in the ignored content-addressed cache.
