#!/usr/bin/env bash
# Tier-1 reruns for the resubmission main tables: 5 independent pretrainings per cell, with the
# fine-tuning walk-length sweep on each one.
#
# WHY 5 PRETRAININGS. The paper reports one pretraining per cell and averages 10 fine-tuning
# seeds on top of it (§4.5). The dirty/clean seed experiment measured what that hides: across
# independent pretraining runs the AP standard deviation is 25-40% of the mean. Fine-tuning seeds
# average away the wrong noise, so every single-pretrain AP in the paper carries ~±30% relative
# uncertainty that is nowhere reported. The pretraining run is the unit of replication.
#
# THE _cr RECIPE, as recovered from the six _cr pretraining logs (warmup-LR identity: for
# tokens < warmup, lr = 3e-4 * tokens/warmup with warmup = total/10; every _cr log solves to
# warmup = 2.0e8 and terminates at ~2.0e9):
#   LANL     --n-tokens 20   (2e9) for BOTH static and temporal.  NOT 80 for temporal.
#   OpTC     default 1e8
#   UNSW     default 1e8
#   all      --size tiny, --walk-len default (4 for LANL with edge features)
# --mini-bs is left unset on purpose: 512 static / 128 temporal are already pretrain.defaults(),
# and `accum = ceil(1024/mini_bs)` normalises 128, 512 and 1024 to the same 1024-walk effective
# batch, so the flag changes memory only.
#
# RESUMABLE. Phase 1 skips a cell whose checkpoint exists; phase 2 skips a cell whose results CSV
# carries the "# BEST" line ResultTracker writes on a clean finish. A failure is logged and the
# lane continues -- it does not take the rest of the lane's cells with it. Re-run the script to
# fill whatever is left; nothing finished is recomputed.
#
#   ./run_main_seeds.sh --list                  # the full matrix and what is already done
#   ./run_main_seeds.sh                         # everything outstanding
#   ./run_main_seeds.sh --pretrain-only
#   ./run_main_seeds.sh --finetune-only
#   DATASETS=lanl14argus ./run_main_seeds.sh    # one dataset
#   PSEEDS="1 2" ./run_main_seeds.sh            # fewer replicates
#
# Run from the repo root.
set -uo pipefail

PSEEDS=${PSEEDS:-"1 2 3 4 5"}
DATASETS=${DATASETS:-"lanl14argus lanl14argus-ts optc-ts optc optc-argus unsw"}
MODES=${MODES:-"static temporal"}
WLS=${WLS:-"1 2 4 8 16"}
SIZE=${SIZE:-tiny}
GPUS=${GPUS:-"0 1 2 3"}
PRETRAIN_PER_GPU=${PRETRAIN_PER_GPU:-1}   # LANL pretraining is ~3.4h at 2e9; one per GPU
FT_PER_GPU=${FT_PER_GPU:-2}
SOLO_WL=${SOLO_WL:-8}                     # wl >= this fine-tunes one job per GPU (OOM at 2/GPU)
EXTRA_PRETRAIN=${EXTRA_PRETRAIN:-""}
EXTRA_FT=${EXTRA_FT:-""}
LOGDIR=${LOGDIR:-logs/main_seeds}
RESULTS=${RESULTS:-results}
PRETRAINED=${PRETRAINED:-pretrained}

LIST_ONLY=0; DO_PT=1; DO_FT=1
for a in "$@"; do
  case "$a" in
    --list|-n)       LIST_ONLY=1 ;;
    --pretrain-only) DO_FT=0 ;;
    --finetune-only) DO_PT=0 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f pretrain.py ] || { echo "run this from the repo root" >&2; exit 1; }

# ---- per-dataset settings ---------------------------------------------------------------------
pt_args() {   # extra pretrain args for dataset $1
  case "$1" in
    lanl*) echo "--n-tokens 20" ;;   # the _cr budget, both modes
    *)     echo "" ;;                # optc-ts / unsw keep their 1e8 defaults
  esac
}
ft_script() { # OpTC has no edge features, so it uses classification fine-tuning (paper §4.6)
  case "$1" in optc*) echo cls_finetune.py ;; *) echo lp_finetune.py ;; esac
}
ft_kind() {   # results/<this>-<mode>/<dataset>/
  case "$1" in optc*) echo cls ;; *) echo lp ;; esac
}

ckpt_path() { # $1 dataset  $2 mode  $3 pseed
  local pre=rw_bert; [ "$2" = temporal ] && pre=trw_bert
  echo "$PRETRAINED/$2/$1/${pre}_${1}_ps${3}_${SIZE}.pt"
}
csv_path() {  # $1 dataset  $2 mode  $3 pseed  $4 wl
  echo "$RESULTS/$(ft_kind "$1")-$2/$1/$(ft_kind "$1")_ps${3}_${SIZE}_wl${4}_ps${3}_s0.csv"
}

# ---- enumerate --------------------------------------------------------------------------------
pt_jobs=(); ft_jobs=(); pt_done=0; ft_done=0; ft_blocked=0
for ds in $DATASETS; do
  for mode in $MODES; do
    for s in $PSEEDS; do
      ckpt=$(ckpt_path "$ds" "$mode" "$s")
      if [ -f "$ckpt" ]; then pt_done=$((pt_done + 1)); else pt_jobs+=("$ds|$mode|$s"); fi
      for wl in $WLS; do
        csv=$(csv_path "$ds" "$mode" "$s" "$wl")
        if [ -f "$csv" ] && grep -q '^# BEST' "$csv" 2>/dev/null; then
          ft_done=$((ft_done + 1)); continue
        fi
        # A cell whose checkpoint is neither present nor queued can never run.
        if [ -f "$ckpt" ] || [ "$DO_PT" = 1 ]; then
          ft_jobs+=("$ds|$mode|$s|$wl")
        else
          ft_blocked=$((ft_blocked + 1))
        fi
      done
    done
  done
done
# cheap walk lengths first, so a long tail of wl16 does not sit behind them
if [ "${#ft_jobs[@]}" -gt 0 ]; then
  IFS=$'\n' ft_jobs=($(printf '%s\n' "${ft_jobs[@]}" | sort -t'|' -k4,4n)); unset IFS
fi

echo "pretraining : ${#pt_jobs[@]} to run, $pt_done already have a checkpoint"
echo "fine-tuning : ${#ft_jobs[@]} to run, $ft_done already finished"
[ "$ft_blocked" -gt 0 ] && echo "              $ft_blocked cells have no checkpoint (--finetune-only)"
echo "  datasets: $DATASETS"
echo "  modes   : $MODES      pretrain seeds: $PSEEDS      walk lengths: $WLS"
echo

if [ "$LIST_ONLY" = 1 ]; then
  if [ "${#pt_jobs[@]}" -gt 0 ]; then
    echo "pretrainings:"
    for j in "${pt_jobs[@]}"; do IFS='|' read -r ds mode s <<< "$j"
      printf '  %-9s %-16s ps%-2s  %s\n' "$mode" "$ds" "$s" "$(pt_args "$ds")"; done
  fi
  if [ "${#ft_jobs[@]}" -gt 0 ]; then
    echo "fine-tunes:"
    printf '%s\n' "${ft_jobs[@]}" | awk -F'|' \
      '{printf "  %-9s %-16s ps%-2s wl%s\n", $2, $1, $3, $4}'
  fi
  exit 0
fi

mkdir -p "$LOGDIR" "$LOGDIR/.status"

# ---- lane runner ------------------------------------------------------------------------------
pass_jobs=()
run_pass() {  # run_pass <per_gpu> <label> <what: pretrain|finetune>
  local per_gpu="$1" label="$2" what="$3" n=${#pass_jobs[@]} lanes=() i j
  [ "$n" -gt 0 ] || return 0
  for g in $GPUS; do for ((i = 0; i < per_gpu; i++)); do lanes+=("$g"); done; done
  local L=${#lanes[@]}
  echo "== $label: $n jobs over $L lanes ($per_gpu per GPU on: $GPUS) =="
  for ((i = 0; i < L; i++)); do
    (
      gpu=${lanes[$i]}
      for ((j = i; j < n; j += L)); do
        if [ "$what" = pretrain ]; then
          IFS='|' read -r ds mode s <<< "${pass_jobs[$j]}"
          trw=""; [ "$mode" = temporal ] && trw="--trw"
          cell="pt_${mode}_${ds}_ps${s}"
          echo "  [gpu $gpu] $cell"
          # shellcheck disable=SC2046
          if python pretrain.py --dataset "$ds" $trw --size "$SIZE" --device "$gpu" \
               --seed "$s" --tag "_ps${s}" $(pt_args "$ds") $EXTRA_PRETRAIN \
               > "$LOGDIR/$cell.log" 2>&1; then
            : > "$LOGDIR/.status/ok_$cell"
          else
            echo "  [gpu $gpu] FAILED $cell -> $LOGDIR/$cell.log" >&2
            : > "$LOGDIR/.status/fail_$cell"
          fi
        else
          IFS='|' read -r ds mode s wl <<< "${pass_jobs[$j]}"
          trw=""; [ "$mode" = temporal ] && trw="--trw"
          cell="ft_${mode}_${ds}_ps${s}_wl${wl}"
          echo "  [gpu $gpu] $cell"
          if python "$(ft_script "$ds")" --dataset "$ds" $trw --size "$SIZE" --device "$gpu" \
               --walk-len "$wl" --seed 0 --pretrain-tag "_ps${s}" --tag "_ps${s}_s0" \
               $EXTRA_FT > "$LOGDIR/$cell.log" 2>&1; then
            : > "$LOGDIR/.status/ok_$cell"
          else
            echo "  [gpu $gpu] FAILED $cell -> $LOGDIR/$cell.log" >&2
            : > "$LOGDIR/.status/fail_$cell"
          fi
        fi
      done
    ) &
  done
  wait
  echo
}

if [ "$DO_PT" = 1 ] && [ "${#pt_jobs[@]}" -gt 0 ]; then
  echo "=== phase 1: pretraining ==="
  pass_jobs=("${pt_jobs[@]}")
  run_pass "$PRETRAIN_PER_GPU" "pretraining" pretrain
fi

if [ "$DO_FT" = 1 ] && [ "${#ft_jobs[@]}" -gt 0 ]; then
  echo "=== phase 2: fine-tuning ==="
  # Drop cells whose checkpoint still does not exist: a pretraining failed, and fine-tuning would
  # only fail again on the missing file.
  runnable=()
  for j in "${ft_jobs[@]}"; do
    IFS='|' read -r ds mode s wl <<< "$j"
    if [ -f "$(ckpt_path "$ds" "$mode" "$s")" ]; then runnable+=("$j"); fi
  done
  skipped=$(( ${#ft_jobs[@]} - ${#runnable[@]} ))
  [ "$skipped" -gt 0 ] && echo "skipping $skipped cells whose pretraining did not produce a checkpoint"

  pass_jobs=(); for j in "${runnable[@]}"; do
    [ "$(cut -d'|' -f4 <<< "$j")" -lt "$SOLO_WL" ] && pass_jobs+=("$j"); done
  run_pass "$FT_PER_GPU" "walk lengths under $SOLO_WL" finetune

  pass_jobs=(); for j in "${runnable[@]}"; do
    [ "$(cut -d'|' -f4 <<< "$j")" -ge "$SOLO_WL" ] && pass_jobs+=("$j"); done
  run_pass 1 "walk lengths $SOLO_WL and up, one per GPU" finetune
fi

ok=$(ls "$LOGDIR/.status" 2>/dev/null | grep -c '^ok_' || true)
bad=$(ls "$LOGDIR/.status" 2>/dev/null | grep -c '^fail_' || true)
echo "succeeded: $ok    failed: $bad"
if [ "$bad" -gt 0 ]; then
  echo
  echo "failed:"
  ls "$LOGDIR/.status" | sed -n 's/^fail_/    /p'
  echo
  echo "re-run ./run_main_seeds.sh to retry just these; finished work is skipped."
  echo "if a long walk length OOMed:  SOLO_WL=4 ./run_main_seeds.sh --finetune-only"
  echo "                         or:  EXTRA_FT=\"--mini-bs 64\" ./run_main_seeds.sh --finetune-only"
  exit 1
fi
echo
echo "then:  python show_results.py --metrics te_auc te_ap te_prec_r50 te_fp_day_r50"