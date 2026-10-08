#!/usr/bin/env bash
# Argus on one dataset, N replications in parallel on the LSF cluster.
#
# Submitted as a job ARRAY, not N separate jobs: one bsub, one job id, and LSB_JOBINDEX becomes
# --run-id inside each element, so the runs are guaranteed distinct ids without any bookkeeping
# here. `bjobs 12345` shows all N, `bkill 12345` kills all N, `bkill "12345[3]"` kills one.
#
# The snapshot cache is built by a separate job that the array depends on. The script would
# otherwise handle it -- load_splits() locks and the losers wait -- but that wastes N-1 GPU
# allocations sitting idle through a ten-minute CPU-bound bucketing pass.
#
#   ./submit_argus.sh                                  # 5 runs on optc-argus
#   RUNS=10 ./submit_argus.sh                          # 10
#   DATASET=optc-ts TAG=_ours ./submit_argus.sh        # the other split, separate results file
#   SCORE_MODE=predictive TAG=_pred ./submit_argus.sh  # separate file: NOT the same experiment
#   ./submit_argus.sh --dry-run
#
# Collect, whenever: python argus_test_optc.py --summarize [--dataset ...] [--tag ...]
set -uo pipefail

RUNS=${RUNS:-5}
DATASET=${DATASET:-optc-argus}
TAG=${TAG:-""}
SCORE_MODE=${SCORE_MODE:-contemporaneous}
EXTRA=${EXTRA:-""}                 # e.g. EXTRA="--epochs 50 --h-dim 64"

QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
WALL=${WALL:-"08:00"}
WALL_PREP=${WALL_PREP:-"02:00"}
MEM_MB=${MEM_MB:-32000}
CORES=${CORES:-4}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
LOGDIR=${LOGDIR:-logs/lsf}
REPO=${REPO:-$PWD}

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

[ -f "$REPO/argus_test_optc.py" ] || {
  echo "no argus_test_optc.py in $REPO -- run this from prior_works/ or set REPO" >&2; exit 1; }
command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
mkdir -p "$LOGDIR"

NAME="argus_${DATASET}${TAG}"
COMMON="--dataset $DATASET --score-mode $SCORE_MODE --tag '$TAG' $EXTRA"

# ---- phase 1: build the snapshot cache once ---------------------------------------------------
prep="$LOGDIR/${NAME}_prep.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -e'
  printf 'cd %q\n' "$REPO"
  echo "$PRELUDE"
  echo 'echo "host: $(hostname)"'
  echo "python argus_test_optc.py --prepare $COMMON"
} > "$prep"

# ---- phase 2: the array -----------------------------------------------------------------------
# --run-id defaults to $LSB_JOBINDEX, so nothing here has to pass it.
body="$LOGDIR/${NAME}.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -e'
  printf 'cd %q\n' "$REPO"
  echo "$PRELUDE"
  echo 'echo "host: $(hostname)  run ${LSB_JOBINDEX:-0}  gpus: ${CUDA_VISIBLE_DEVICES:-unset}"'
  echo 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true'
  echo "python argus_test_optc.py --device 0 $COMMON"
} > "$body"

prep_args=(-J "${NAME}_prep" -q "$QUEUE" -n "$CORES" -W "$WALL_PREP"
           -R "rusage[mem=$MEM_MB]" -o "$LOGDIR/${NAME}_prep.out" -e "$LOGDIR/${NAME}_prep.out")
# %I is the array index; each element gets its own log.
run_args=(-J "${NAME}[1-${RUNS}]" -q "$QUEUE" -gpu "$GPU_RES" -n "$CORES" -W "$WALL"
          -R "rusage[mem=$MEM_MB]" -o "$LOGDIR/${NAME}_%I.out" -e "$LOGDIR/${NAME}_%I.out")

if [ "$DRY" = 1 ]; then
  echo "bsub ${prep_args[*]} < $prep"
  echo "bsub ${run_args[*]} -w 'done(<prep id>)' < $body"
  echo
  echo "--- $prep ---";  cat "$prep"
  echo "--- $body ---";  cat "$body"
  exit 0
fi

pid=$(bsub "${prep_args[@]}" < "$prep" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
[ -n "$pid" ] || { echo "prep submit failed" >&2; exit 1; }
echo "cache build : $pid"

# done(), not ended(): if the cache build fails there is nothing to train on, so the array should
# stay pending rather than start and fail N times. (The pretraining jobs elsewhere use ended()
# because their in-job guard handles a missing checkpoint by exiting 0.)
aid=$(bsub "${run_args[@]}" -w "done($pid)" < "$body" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
[ -n "$aid" ] || { echo "array submit failed" >&2; exit 1; }
echo "runs 1-$RUNS : $aid  (after $pid)"
echo
echo "watch    : bjobs -A $aid"
echo "logs     : $LOGDIR/${NAME}_1.out ... ${NAME}_${RUNS}.out"
echo "collect  : python argus_test_optc.py --summarize --dataset $DATASET --tag '$TAG'"