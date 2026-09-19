#!/usr/bin/env bash
# Walk-length sweep over several seeds.
#
# Default: pretrain once, then LP fine-tune from the final and best pretraining checkpoints.
# With --model-fname: skip pretraining and sweep that checkpoint instead (e.g. a --snapshot-every
# checkpoint, or a model pretrained by hand with a different token budget).
#
# Static and temporal walks use different settings, based on measured walk lengths on
# lanl14argus-dirtyts (32-hop cap: static averages ~11 nodes, temporal ~2.6):
#   static:   pretrain 32 hops, fine-tune walk lengths 1 2 4 8 16 32
#   temporal: pretrain 4 hops,  fine-tune walk lengths 1 2 3 4
#
# Usage: ./sweep_walk_len.sh DATASET DEVICE [--trw] [--model-fname PATH] [--tag NAME]
#                                          [--wls "1 2 4"] [--seeds "0 1 2"]
#   ./sweep_walk_len.sh lanl14argus-dirtyts 2
#   ./sweep_walk_len.sh lanl14argus-dirtyts 2 --trw
#   ./sweep_walk_len.sh lanl14argus 0 \
#       --model-fname pretrained/static/lanl14argus/rw_bert_lanl14argus_wl4_2e9_tiny-snap200Mtok.pt
#
# --tag names the run in every results file; with --model-fname it defaults to the part of the
# checkpoint name after the last "-" (e.g. snap200Mtok), so snapshots don't overwrite each other.
set -euo pipefail

usage() {
  echo "Usage: $0 DATASET DEVICE [--trw] [--model-fname PATH] [--tag NAME] [--wls \"1 2 4\"] [--seeds \"0 1 2\"]" >&2
  exit 1
}

[ $# -ge 2 ] || usage
DS=$1
DEV=$2
shift 2

TRW=""
TAG=""
CKPT=""
WLS=""
SEEDS="0 1 2 3 4"
while [ $# -gt 0 ]; do
  case "$1" in
    --trw) TRW="--trw" ;;
    --model-fname) shift; [ $# -gt 0 ] || usage; CKPT="$1" ;;
    --model-fname=*) CKPT="${1#*=}" ;;
    --tag) shift; [ $# -gt 0 ] || usage; TAG="$1" ;;
    --tag=*) TAG="${1#*=}" ;;
    --wls) shift; [ $# -gt 0 ] || usage; WLS="$1" ;;
    --wls=*) WLS="${1#*=}" ;;
    --seeds) shift; [ $# -gt 0 ] || usage; SEEDS="$1" ;;
    --seeds=*) SEEDS="${1#*=}" ;;
    *) usage ;;
  esac
  shift
done

if [ -n "$TRW" ]; then
  PT_WL=4;  PT_BS=128; EVAL_EVERY=100; DEFAULT_WLS="1 2 3 4"
else
  PT_WL=32; PT_BS=128; EVAL_EVERY=14;  DEFAULT_WLS="1 2 4 8 16 32"
fi
[ -n "$WLS" ] || WLS="$DEFAULT_WLS"

if [ -n "$CKPT" ]; then
  [ -f "$CKPT" ] || { echo "No such checkpoint: $CKPT" >&2; exit 1; }
  if [ -z "$TAG" ]; then                      # rw_bert_..._tiny-snap200Mtok.pt -> snap200Mtok
    TAG=$(basename "$CKPT" .pt)
    TAG=${TAG##*-}
  fi
  echo "dataset=$DS device=$DEV ${TRW:-static} checkpoint=$CKPT tag=$TAG walk lengths: $WLS seeds: $SEEDS"
  for wl in $WLS; do
    for seed in $SEEDS; do
      python lp_finetune.py --dataset "$DS" --device "$DEV" --walk-len "$wl" $TRW \
        --model-fname "$CKPT" --pretrain-tag "_${TAG}" --seed "$seed" --tag "_s${seed}"
    done
  done
  exit 0
fi

PT_TAG="_wl${PT_WL}${TAG:+_$TAG}"
echo "dataset=$DS device=$DEV ${TRW:-static} pretrain_walk_len=$PT_WL tag=$PT_TAG walk lengths: $WLS seeds: $SEEDS"

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