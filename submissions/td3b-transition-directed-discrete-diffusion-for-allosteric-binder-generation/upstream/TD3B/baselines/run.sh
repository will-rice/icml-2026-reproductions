#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CSV_PATH="${1:-To Be Added}"
BASELINE="${2:-unguided}"
DEVICE="${3:-cuda:4}"
OUTPUT_DIR="${4:-${SCRIPT_DIR}/outputs}"
NGPUS="${5:-1}"
MASTER_PORT="${6:-29500}"

if [ "$NGPUS" -gt 1 ]; then
  echo "Running multi-GPU inference with $NGPUS GPUs (master port: $MASTER_PORT)"
  LAUNCH_DEVICE="cuda"
  python -m torch.distributed.run \
    --nproc_per_node="$NGPUS" \
    --master_port="$MASTER_PORT" \
    "${SCRIPT_DIR}/sampling_setup.py" \
    --ckpt_path "${ROOT_DIR}/pretrained/peptune-pretrained.ckpt" \
    --device "${LAUNCH_DEVICE}" \
    --baseline "${BASELINE}" \
    --targets_csv "${CSV_PATH}" \
    --batch_size 8 \
    --num_steps 128 \
    --num_batches 1 \
    --output_dir "${OUTPUT_DIR}"

  export OUTPUT_DIR BASELINE
  python - <<'PY'
import glob
import os
import pandas as pd

out_dir = os.environ["OUTPUT_DIR"]
baseline = os.environ["BASELINE"]

def merge(pattern, output_name):
    files = sorted(glob.glob(os.path.join(out_dir, pattern)))
    if not files:
        return
    dfs = []
    for path in files:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as exc:
            print(f"[merge] skip {path}: {exc}")
    if not dfs:
        return
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(os.path.join(out_dir, output_name), index=False)
    print(f"[merge] wrote {output_name} from {len(files)} shards")

merge(f"{baseline}_samples_rank*.csv", f"{baseline}_samples.csv")
merge("batch_times_rank*.csv", "batch_times.csv")
merge(f"{baseline}_metrics_rank*.csv", f"{baseline}_metrics.csv")
PY
  exit 0
fi

python "${SCRIPT_DIR}/sampling_setup.py" \
  --ckpt_path "${ROOT_DIR}/pretrained/peptune-pretrained.ckpt" \
  --device "${DEVICE}" \
  --baseline "${BASELINE}" \
  --targets_csv "${CSV_PATH}" \
  --batch_size 8 \
  --num_steps 128 \
  --num_batches 1 \
  --output_dir "${OUTPUT_DIR}"

# ./run.sh To Be Added peptune cuda:0 To Be Added
# ./run.sh To Be Added peptune cuda To Be Added 4 29501
# ./run.sh To Be Added tds cuda:1 To Be Added
# ./run.sh To Be Added smc cuda:2 To Be Added
# ./run.sh To Be Added cg cuda:3 To Be Added
# ./run.sh To Be Added unguided cuda:4 To Be Added
