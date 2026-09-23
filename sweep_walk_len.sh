#!/usr/bin/env bash
# Fine-tune over walk lengths x seeds, from both the final and the -best pretrained checkpoint.
# No pretraining is run here.
#
# Everything you pass is handed straight to lp_finetune.py, so any flag it accepts works.
# Loops are controlled by environment variables:
#   WLS    walk lengths   (default "1 2 4 8 16 32")
#   SEEDS  seeds          (default "0 1 2 3 4")
#   TAG    results prefix (default empty; "_s<seed>" / "_best_s<seed>" is appended)
#   SCRIPT which script   (default lp_finetune.py; e.g. cls_finetune.py, lp_finetune_raw.py)
#
#   TAG=_2e9 ./sweep_walk_len.sh --dataset lanl14argus --device 2 --pretrain-tag _wl4_2e9
#   WLS="1 2 3 4" SEEDS="0 1 2" ./sweep_walk_len.sh --dataset lanl14argus --device 3 --trw --pretrain-tag _wl4_2e9
set -euo pipefail

[ $# -gt 0 ] || { echo "Usage: [WLS=...] [SEEDS=...] [TAG=...] [SCRIPT=...] $0 <args for lp_finetune.py>" >&2; exit 1; }

WLS=${WLS:-"1 2 4 8 16"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TAG=${TAG:-""}
SCRIPT=${SCRIPT:-lp_finetune.py}

echo "$SCRIPT | walk lengths: $WLS | seeds: $SEEDS | args: $*"

for wl in $WLS; do
  for seed in $SEEDS; do
    echo "== walk-len $wl seed $seed | final checkpoint =="
    python "$SCRIPT" --walk-len "$wl" --seed "$seed" "$@" --tag "${TAG}_s${seed}"

    #echo "== walk-len $wl seed $seed | -best checkpoint =="
    #python "$SCRIPT" --walk-len "$wl" --seed "$seed" "$@" --best-pretrained --tag "${TAG}_best_s${seed}"
  done
done