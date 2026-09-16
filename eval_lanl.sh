#!/usr/bin/env bash
# Walk-length sweep: pretrain once, then LP fine-tune from the final and best
# pretraining checkpoints over several walk lengths and seeds.
#
# Static and temporal walks use different settings, based on measured walk lengths
# on lanl14argus-dirtyts (32-hop cap: static averages ~11 nodes, temporal ~2.6):
#   static:   pretrain 32 hops, fine-tune walk lengths 1 2 4 8 16 32
#   temporal: pretrain 4 hops,  fine-tune walk lengths 1 2 3 4
#
# Usage: ./sweep_walk_len.sh DATASET DEVICE [--trw]
#   e.g. ./sweep_walk_len.sh lanl14argus-dirtyts 2
#        ./sweep_walk_len.sh lanl14argus-dirtyts 2 --trw
set -euo pipefail

usage() { echo "Usage: $0 DATASET DEVICE [--trw]" >&2; exit 1; }

[ $# -ge 2 ] || usage
DS=$1
DEV=$2
shift 2

TRW=""
while [ $# -gt 0 ]; do
  case "$1" in
    --trw) TRW="--trw" ;;
    *) usage ;;
  esac
  shift
done

if [ -n "$TRW" ]; then
  PT_WL=4;  PT_BS=128; EVAL_EVERY=100; WLS="1 2 3 4"
else
  PT_WL=32; PT_BS=128;  EVAL_EVERY=14;  WLS="1 2 4 8 16 32"
fi
PT_TAG="_wl${PT_WL}"
SEEDS="0 1 2 3 4"

echo "dataset=$DS device=$DEV ${TRW:-static} pretrain_walk_len=$PT_WL fine-tune walk lengths: $WLS"

python pretrain.py --dataset "$DS" --device "$DEV" $TRW \
  --walk-len $PT_WL --mini-bs $PT_BS --eval-every $EVAL_EVERY --tag "$PT_TAG"

for ckpt in final best; do
  flag=""
  [ "$ckpt" = best ] && flag="--best-pretrained"
  for wl in $WLS; do
    for seed in $SEEDS; do
      python lp_finetune.py --dataset "$DS" --device "$DEV" --walk-len "$wl" $flag $TRW \
        --pretrain-tag "$PT_TAG" --seed "$seed" --tag "_${ckpt}_s${seed}"
    done
  done
done