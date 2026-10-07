#!/usr/bin/env bash
# Fill in fine-tuning cells that are missing or unfinished, without cascading.
#
# One job here is ONE lp_finetune.py call (one mode x dataset x pretrain-seed x walk-len x
# ft-seed), not a whole sweep. sweep_walk_len.sh is bypassed on purpose: it runs
# `set -euo pipefail`, so one walk length failing kills the remaining walk lengths in that sweep.
# Here a failed cell is logged and the lane moves on to the next one.
#
# A cell counts as DONE when its results CSV exists and ends with the "# BEST ..." line that
# ResultTracker.close() writes. A CSV without that line is a run that died mid-way, so it is
# redone. Nothing finished is ever recomputed unless you pass --force.
#
#   ./fill_missing.sh --list              # what is missing, and why; runs nothing
#   ./fill_missing.sh                     # run the missing cells
#   WLS=16 PER_GPU=1 ./fill_missing.sh    # retry just the walk length that OOMed, one job per GPU
#   EXTRA_FT="--mini-bs 64" ./fill_missing.sh
#   ./fill_missing.sh --force             # recompute finished cells too
#
# Run from the repo root.
set -uo pipefail

PSEEDS=${PSEEDS:-"1 2 3 4"}
DATASETS=${DATASETS:-"lanl14argus-dirtyts lanl14argus-cleants"}
MODES=${MODES:-"static temporal"}
WLS=${WLS:-"1 2 4 8 16"}
FT_SEEDS=${FT_SEEDS:-"0"}
SIZE=${SIZE:-tiny}                  # must match what lp_finetune.py --size will be given
GPUS=${GPUS:-"0 1 2 3"}
PER_GPU=${PER_GPU:-2}
EXTRA_FT=${EXTRA_FT:-""}
LOGDIR=${LOGDIR:-logs/fill_missing}
RESULTS=${RESULTS:-results}
PRETRAINED=${RESULTS_PRETRAINED:-pretrained}

LIST_ONLY=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --list|-n) LIST_ONLY=1 ;;
    --force)   FORCE=1 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done

[ -f lp_finetune.py ] || { echo "run this from the repo root (lp_finetune.py must be here)" >&2; exit 1; }

# ---- enumerate the matrix ---------------------------------------------------------------------
# jobs[]  = "mode|ds|pseed|wl|seed"
# A cell is skipped as unrunnable (rather than queued and failed) if its checkpoint is absent.
jobs=(); n_done=0; n_nockpt=0; n_partial=0
nockpt_list=()
for mode in $MODES; do
  kind=$mode
  prefix=rw_bert; trw=""
  [ "$mode" = temporal ] && { prefix=trw_bert; trw="--trw"; }
  for ds in $DATASETS; do
    for s in $PSEEDS; do
      ckpt="$PRETRAINED/$kind/$ds/${prefix}_${ds}_ps${s}_${SIZE}.pt"
      have_ckpt=1; [ -f "$ckpt" ] || have_ckpt=0
      for wl in $WLS; do
        for seed in $FT_SEEDS; do
          csv="$RESULTS/lp-$kind/$ds/lp_ps${s}_${SIZE}_wl${wl}_ps${s}_s${seed}.csv"
          if [ -f "$csv" ] && grep -q '^# BEST' "$csv" 2>/dev/null; then
            if [ "$FORCE" = 0 ]; then n_done=$((n_done + 1)); continue; fi
          elif [ -f "$csv" ]; then
            n_partial=$((n_partial + 1))
          fi
          if [ "$have_ckpt" = 0 ]; then
            n_nockpt=$((n_nockpt + 1))
            nockpt_list+=("$ckpt")
            continue
          fi
          jobs+=("$mode|$ds|$s|$wl|$seed")
        done
      done
    done
  done
done

# Cheap cells first: short walks finish fast, so a long tail of wl16 jobs is not sitting behind
# them, and if the big ones OOM again you find out with most of the matrix already filled.
if [ "${#jobs[@]}" -gt 0 ]; then
  IFS=$'\n' jobs=($(printf '%s\n' "${jobs[@]}" | sort -t'|' -k4,4n)); unset IFS
fi

N=${#jobs[@]}
echo "already finished : $n_done"
[ "$n_partial" -gt 0 ] && echo "unfinished CSVs  : $n_partial  (will be redone)"
if [ "$n_nockpt" -gt 0 ]; then
  echo "no checkpoint    : $n_nockpt cells skipped; pretraining is missing for:"
  printf '%s\n' "${nockpt_list[@]}" | sort -u | sed 's/^/    /'
fi
echo "to run           : $N"
if [ "$N" = 0 ]; then echo; echo "nothing to do."; exit 0; fi

if [ "$LIST_ONLY" = 1 ]; then
  echo
  printf '%s\n' "${jobs[@]}" | awk -F'|' '{printf "  %-9s %-22s ps%-2s wl%-3s s%s\n", $1, $2, $3, $4, $5}'
  exit 0
fi

lanes=(); for g in $GPUS; do for ((k = 0; k < PER_GPU; k++)); do lanes+=("$g"); done; done
L=${#lanes[@]}
mkdir -p "$LOGDIR" "$LOGDIR/.status"
rm -f "$LOGDIR"/.status/*
echo "over $L lanes ($PER_GPU per GPU on: $GPUS), logs in $LOGDIR"
echo

for ((i = 0; i < L; i++)); do
  (
    gpu=${lanes[$i]}
    for ((j = i; j < N; j += L)); do
      IFS='|' read -r mode ds s wl seed <<< "${jobs[$j]}"
      trw=""; [ "$mode" = temporal ] && trw="--trw"
      cell="${mode}_${ds}_ps${s}_wl${wl}_s${seed}"
      log="$LOGDIR/$cell.log"
      echo "  [gpu $gpu] $cell"
      if python lp_finetune.py --dataset "$ds" $trw --device "$gpu" --size "$SIZE" \
           --walk-len "$wl" --seed "$seed" --pretrain-tag "_ps${s}" --tag "_ps${s}_s${seed}" \
           $EXTRA_FT > "$log" 2>&1; then
        : > "$LOGDIR/.status/ok_$cell"
      else
        # Logged, not fatal: the next cell in this lane is independent of this one.
        echo "  [gpu $gpu] FAILED $cell  ->  $log" >&2
        : > "$LOGDIR/.status/fail_$cell"
      fi
    done
  ) &
done
wait

ok=$(ls "$LOGDIR/.status" | grep -c '^ok_' || true)
bad=$(ls "$LOGDIR/.status" | grep -c '^fail_' || true)
echo
echo "finished: $ok    failed: $bad"
if [ "$bad" -gt 0 ]; then
  echo
  echo "failed cells:"
  ls "$LOGDIR/.status" | sed -n 's/^fail_/    /p'
  echo
  echo "the usual cause is CUDA OOM at the longest walk length with several lanes on one GPU."
  echo "Check the tail of a log, then retry just those with less concurrency, e.g."
  echo "  WLS=16 PER_GPU=1 ./fill_missing.sh"
  echo "or with a smaller micro-batch:"
  echo "  WLS=16 EXTRA_FT=\"--mini-bs 64\" ./fill_missing.sh"
  exit 1
fi