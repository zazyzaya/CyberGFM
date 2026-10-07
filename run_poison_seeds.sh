#!/usr/bin/env bash
# Pretraining-seed replicates for the dirty/clean poisoning comparison.
#
# Poisoning acts at PRETRAINING, so the experimental unit is a pretraining run, not a fine-tuning
# seed. One checkpoint per arm gives n=1 vs n=1 however many walk lengths and fine-tuning seeds
# are swept on top of it. This runs several independent pretrainings per arm so the difference
# between arms can be compared against the variance between runs within an arm.
#
# Phase 1 pretrains  PSEEDS x DATASETS x MODES,  each tagged _ps<seed>.
# Phase 2 fine-tunes each of those checkpoints over the walk-length sweep.
# Nothing existing is touched: every checkpoint, log and results file carries _ps<seed>.
#
#   ./run_poison_seeds.sh                       # the full matrix
#   DRYRUN=1 ./run_poison_seeds.sh              # print the schedule and exit
#   PSEEDS="1 2" ./run_poison_seeds.sh          # fewer replicates
#   SKIP_PRETRAIN=1 ./run_poison_seeds.sh       # fine-tune existing _ps checkpoints only
#
# Run from the repo root (where pretrain.py and sweep_walk_len.sh live).
set -uo pipefail

PSEEDS=${PSEEDS:-"1 2 3 4"}                 # pretraining seeds; 0 is assumed to be your existing run
DATASETS=${DATASETS:-"lanl14argus-dirtyts lanl14argus-cleants"}
MODES=${MODES:-"static temporal"}
GPUS=${GPUS:-"0 1 2 3"}
PER_GPU=${PER_GPU:-2}
WLS=${WLS:-"1 2 4 8 16"}
FT_SEEDS=${FT_SEEDS:-"0"}                   # fine-tuning seeds PER checkpoint; see note at the end
# Pretraining args. The _cr recipe used DIFFERENT settings per mode (static: --mini-bs 512
# --n-tokens 20; temporal: --mini-bs 128 --n-tokens 80), so these are per-mode. Set them to match
# whatever produced the checkpoints you are comparing against, or the new seeds are not
# comparable to them.
EXTRA_PRETRAIN=${EXTRA_PRETRAIN:-""}                    # both modes
EXTRA_PRETRAIN_STATIC=${EXTRA_PRETRAIN_STATIC:-""}      # static only
EXTRA_PRETRAIN_TEMPORAL=${EXTRA_PRETRAIN_TEMPORAL:-""}  # temporal only
EXTRA_FT=${EXTRA_FT:-""}
DRYRUN=${DRYRUN:-0}
SKIP_PRETRAIN=${SKIP_PRETRAIN:-0}
LOGDIR=${LOGDIR:-logs/poison_seeds}

[ -f pretrain.py ] && [ -f sweep_walk_len.sh ] || {
  echo "run this from the repo root (pretrain.py and sweep_walk_len.sh must be here)" >&2; exit 1; }
mkdir -p "$LOGDIR"

# ---- lanes: PER_GPU slots on each GPU, jobs dealt round-robin ---------------------------------
lanes=()
for g in $GPUS; do for ((k = 0; k < PER_GPU; k++)); do lanes+=("$g"); done; done
L=${#lanes[@]}

specs=()
for ds in $DATASETS; do for mode in $MODES; do for s in $PSEEDS; do specs+=("$ds|$mode|$s"); done; done; done
N=${#specs[@]}

echo "$N jobs per phase over $L lanes ($PER_GPU per GPU on: $GPUS)"
echo "  datasets : $DATASETS"
echo "  modes    : $MODES"
echo "  pretrain seeds : $PSEEDS"
echo "  walk lengths   : $WLS        fine-tune seeds per checkpoint: $FT_SEEDS"
echo "  logs     : $LOGDIR"
echo

run() {  # run <logfile> <command...>
  local log="$1"; shift
  if [ "$DRYRUN" = 1 ]; then
    echo "    $*"
    return 0
  fi
  "$@" > "$log" 2>&1
}

# $1 = phase name, used to pick which command each spec becomes
phase() {
  local what="$1" rc=0
  local pids=()
  for ((i = 0; i < L; i++)); do
    (
      gpu=${lanes[$i]}
      for ((j = i; j < N; j += L)); do
        IFS='|' read -r ds mode s <<< "${specs[$j]}"
        trw=""; extra_pt="$EXTRA_PRETRAIN_STATIC"
        if [ "$mode" = temporal ]; then trw="--trw"; extra_pt="$EXTRA_PRETRAIN_TEMPORAL"; fi
        tag="_ps${s}"
        if [ "$what" = pretrain ]; then
          log="$LOGDIR/pretrain_${ds}_${mode}${tag}.log"
          echo "  [gpu $gpu] pretrain $ds $mode $tag -> $log"
          run "$log" python pretrain.py --dataset "$ds" $trw --seed "$s" --tag "$tag" \
              --device "$gpu" $EXTRA_PRETRAIN $extra_pt || { echo "  FAILED $log" >&2; exit 1; }
        else
          log="$LOGDIR/ft_${ds}_${mode}${tag}.log"
          echo "  [gpu $gpu] finetune $ds $mode $tag -> $log"
          if [ "$DRYRUN" = 1 ]; then
            echo "    TAG=$tag WLS=\"$WLS\" SEEDS=\"$FT_SEEDS\" ./sweep_walk_len.sh" \
                 "--dataset $ds $trw --device $gpu --pretrain-tag $tag $EXTRA_FT"
          else
            TAG="$tag" WLS="$WLS" SEEDS="$FT_SEEDS" ./sweep_walk_len.sh \
              --dataset "$ds" $trw --device "$gpu" --pretrain-tag "$tag" $EXTRA_FT \
              > "$log" 2>&1 || { echo "  FAILED $log" >&2; exit 1; }
          fi
        fi
      done
    ) &
    pids+=($!)
  done
  # bare `wait` returns the status of the LAST child only, so a failure in any other lane would
  # go unnoticed and phase 2 would start against a missing checkpoint. Wait on each lane.
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

if [ "$SKIP_PRETRAIN" != 1 ]; then
  echo "=== phase 1: pretraining ==="
  if ! phase pretrain; then
    echo >&2
    echo "a pretraining job failed; NOT starting fine-tuning (it would load a missing or partial" >&2
    echo "checkpoint). Check $LOGDIR/pretrain_*.log, then re-run with SKIP_PRETRAIN=1 once fixed." >&2
    exit 1
  fi
  echo "pretraining done"
  echo
fi

echo "=== phase 2: fine-tuning (walk-length sweep per checkpoint) ==="
if ! phase finetune; then
  echo >&2
  echo "a fine-tuning sweep failed; see $LOGDIR/ft_*.log. Sweeps that finished are still valid." >&2
  exit 1
fi

echo
echo "done. To compare arms across pretraining seeds:"
echo "  cd notebooks && python plot_poison_by_wl.py --pool 'ps[0-9]+'"
echo
echo "--pool groups the _ps<seed> runs together so the error bars span PRETRAINING runs, which is"
echo "the variance the poisoning manipulation actually acts through. Without it each _ps<seed>"
echo "is drawn as its own line, which is the useful view for seeing whether the arms separate"
echo "by more than the seeds do."