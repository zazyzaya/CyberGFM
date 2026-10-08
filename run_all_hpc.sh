#!/usr/bin/env bash
# LSF version of run_main_seeds.sh: submits the Tier-1 matrix as one bsub job per cell and lets
# the scheduler sequence it. Same experiment definition, same _cr recipe, same resumability rules.
#
# WHY ONE JOB PER CELL rather than per sweep. A fine-tune that dies (the wl16 OOMs) then costs
# only itself. Fine-tune jobs are chained to their pretraining job with `-w "ended(<jobid>)"`,
# so the whole matrix can be queued in one go and LSF fills whatever is free.
#
# WHY `ended` AND NOT `done`. With `done(...)`, a failed pretraining leaves every dependent job
# PEND forever and you have to hunt them down with bkill. With `ended(...)` the dependent always
# becomes eligible, and the job body checks for the checkpoint itself and exits 0 with a message
# if the pretraining did not produce one. Failures stay local and nothing hangs in the queue.
#
# RESUMABLE. A pretraining whose checkpoint exists is not submitted; a fine-tune whose results CSV
# carries the "# BEST" line ResultTracker writes on a clean finish is not submitted. Re-run this
# script after a batch to queue only what is still outstanding.
#
#   ./submit_main_seeds.sh --dry-run      # print every bsub command, submit nothing
#   ./submit_main_seeds.sh --list         # just the outstanding matrix
#   QUEUE_PT=h100 QUEUE_FT=normal ./submit_main_seeds.sh
#   WLS=match ./submit_main_seeds.sh      # only the pretraining-matched walk lengths
#   DATASETS=lanl14argus ./submit_main_seeds.sh
#
# Run from the repo root on the submit host (login01/login02).
set -uo pipefail

# ---- CLUSTER SETTINGS -- fill these in before the first real submit ---------------------------
# This cluster, as of 2026-10-07 (lshosts -gpu / lsload -gpuload / bqueues):
#   node01-06   4x NVIDIA L40, 44.9 GB each  = 24 cards, ~16 idle
#   nodeh100    2x H100 PCIe,  79.6 GB each  = both idle
#   queues      h100 and short are EMPTY; normal has 790 pending, long has 272
# So VRAM is not a constraint at any walk length -- 44.9 GB is ample for BERT-tiny even at wl32,
# and the --mini-bs / one-job-per-GPU workarounds the local runner needed are unnecessary here.
# The binding constraint is queue wait: a job in `normal` sits behind ~790 others, while `h100`
# starts immediately. Hence separate queues per phase.
QUEUE=${QUEUE:-"normal"}                              # fallback for both phases
QUEUE_PT=${QUEUE_PT:-${QUEUE:?set QUEUE, or QUEUE_PT and QUEUE_FT (see \`bqueues\`)}}
QUEUE_FT=${QUEUE_FT:-$QUEUE}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
                                # with 44.9 GB cards you can instead pack two jobs onto one GPU
                                # with GPU_RES="num=1:mode=shared"
HOSTS=${HOSTS:-""}              # optional host restriction, e.g. HOSTS="nodeh100"
WALL_PT=${WALL_PT:-"12:00"}     # must be within the queue's RUNLIMIT (bqueues -l <queue>).
                                # LANL at 2e9 is ~3.4h on a V100, less on an L40 or H100.
WALL_FT=${WALL_FT:-"08:00"}     # wl16 is ~4x a wl4 run
MEM_MB=${MEM_MB:-"32000"}       # host RAM reservation per job (not VRAM)
CORES=${CORES:-4}               # slots per job; the walk sampler is the CPU-side cost
PRELUDE=${PRELUDE:-""}          # shell run inside each job before python, e.g.
                                #   PRELUDE='module load cuda/12.4; source ~/miniconda3/bin/activate base'
REPO=${REPO:-$PWD}

# ---- EXPERIMENT MATRIX -- identical to run_main_seeds.sh ---------------------------------------
PSEEDS=${PSEEDS:-"1 2 3 4 5"}
DATASETS=${DATASETS:-"lanl14argus-dirtyts lanl14argus-cleants lanl14argus lanl14argus-ts"}
MODES=${MODES:-"static temporal"}
WLS=${WLS:-"1 2 4 8"}        # or WLS=match for the pretraining walk length per dataset
SIZE=${SIZE:-tiny}
EXTRA_PRETRAIN=${EXTRA_PRETRAIN:-""}
EXTRA_FT=${EXTRA_FT:-""}
LOGDIR=${LOGDIR:-logs/lsf}
RESULTS=${RESULTS:-results}
PRETRAINED=${PRETRAINED:-pretrained}

DRY=0; LIST_ONLY=0; PREFLIGHT=0; PREFLIGHT_ONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --list|-n) LIST_ONLY=1 ;;
    --preflight) PREFLIGHT_ONLY=1 ;;
    --no-preflight) PREFLIGHT=0 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f "$REPO/pretrain.py" ] || { echo "no pretrain.py in $REPO -- run from the repo root or set REPO" >&2; exit 1; }
command -v bsub >/dev/null || { echo "bsub not found; run this on login01/login02" >&2; exit 1; }

# ---- per-dataset settings ---------------------------------------------------------------------
#pt_args()     { case "$1" in lanl*) echo "--n-tokens 20" ;; *) echo "" ;; esac; }
pt_args()     { case "$1" in lanl*) echo "--n-tokens 10" ;; *) echo "" ;; esac; }
ft_script()   { case "$1" in optc*) echo cls_finetune.py ;; *) echo lp_finetune.py ;; esac; }
ft_kind()     { case "$1" in optc*) echo cls ;; *) echo lp ;; esac; }
pt_walk_len() { case "$1" in lanl*) echo 4 ;; *) echo 64 ;; esac; }
wls_for()     { if [ "$WLS" = match ]; then pt_walk_len "$1"; else echo "$WLS"; fi; }

ckpt_path() { local pre=rw_bert; [ "$2" = temporal ] && pre=trw_bert
              echo "$PRETRAINED/$2/$1/${pre}_${1}_ps${3}_${SIZE}.pt"; }
csv_path()  { echo "$RESULTS/$(ft_kind "$1")-$2/$1/$(ft_kind "$1")_ps${3}_${SIZE}_wl${4}_ps${3}_s0.csv"; }

mkdir -p "$LOGDIR"

# ---- preflight --------------------------------------------------------------------------------
# Every environment fault so far has surfaced only once a job was already running on a compute
# node: a transformers major bump, a C++ ABI mismatch in the walk extension, then a CPU-only
# build of it. All three are detectable from the login node in a couple of seconds, so check
# before putting hundreds of jobs in the queue. Runs under PRELUDE so it tests the same
# interpreter the jobs will use. --no-preflight skips it; --preflight runs it and stops.
preflight() {
  local sh="$LOGDIR/.preflight.sh" rc=0
  {
    echo "set -e"
    [ -n "$PRELUDE" ] && echo "$PRELUDE"
    cat <<'CHK'
# From a neutral cwd: a directory named like an installed package shadows it on sys.path.
cd "$(mktemp -d)"
python - <<'PY'
import glob, importlib, os, sys
fail = []
print('python     ', sys.version.split()[0])
try:
    import torch, transformers
except ImportError as e:
    sys.exit(f'PREFLIGHT FAILED: {e}. Is the right conda env active? '
             'PRELUDE is what the jobs will use, so set it here too.')
print('torch      ', torch.__version__, '| cuda', torch.version.cuda,
      '| cxx11 abi', torch._C._GLIBCXX_USE_CXX11_ABI)
print('transformers', transformers.__version__)
if transformers.__version__.split('.')[0] != '4':
    fail.append('transformers is not 4.x: models/ declares _tied_weights_keys as a list, which '
                '5.x rejects (AttributeError on .keys() in get_expanded_tied_weights_keys)')

# the walk extension, under either name
mod = None
for name in ('graph_walks', 'temporal_walks'):
    try:
        mod = importlib.import_module(name); print('walks      ', name, 'ok'); break
    except ImportError:
        continue
if mod is None:
    fail.append('neither graph_walks nor temporal_walks imports')
else:
    sos = sorted(os.path.basename(p) for p in glob.glob(os.path.dirname(mod.__file__) + '/*.so'))
    print('walk kernels', ' '.join(sos) or '(none)')
    if not any('_cuda' in s for s in sos):
        fail.append('the walk extension has no *_cuda.so -- it was built CPU-only and will raise '
                    '"Not compiled with CUDA support" at the first batch. Rebuild with '
                    'FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9;9.0" and nvcc on PATH')

# torch-cluster registers random_walk as a torch op; nothing imports it by name
try:
    torch.ops.torch_cluster.random_walk
    print('torch_cluster ok')
except Exception as e:
    fail.append(f'torch.ops.torch_cluster.random_walk unavailable ({e.__class__.__name__}) -- '
                'install torch-cluster from the PyG wheel index for this torch')

if fail:
    print('\nPREFLIGHT FAILED:')
    for f in fail:
        print('  -', f)
    sys.exit(1)
print('\npreflight ok')
PY
CHK
  } > "$sh"
  bash "$sh" || rc=1
  return $rc
}

if [ "$PREFLIGHT_ONLY" = 1 ]; then preflight; exit $?; fi
if [ "$PREFLIGHT" = 1 ] && [ "$LIST_ONLY" = 0 ] && [ "$DRY" = 0 ]; then
  if ! preflight; then
    echo >&2
    echo "not submitting. Fix the above, or re-run with --no-preflight to submit anyway." >&2
    exit 1
  fi
  echo
fi

# ---- submit -----------------------------------------------------------------------------------
declare -A PT_JOBID=()
n_pt=0 n_ft=0 pt_have=0 ft_have=0

submit() {  # submit <jobname> <queue> <walllimit> <dependency-or-empty> <command...>; echoes job id
  local name="$1" queue="$2" wall="$3" dep="$4"; shift 4
  local out="$LOGDIR/${name}.out" jobsh="$LOGDIR/${name}.sh"

  # Each job's body is written to its own script, so a single cell can be inspected or re-run by
  # hand (`bash logs/lsf/ft_static_lanl14argus_ps1_wl16.sh`) without going through LSF at all.
  {
    echo "#!/usr/bin/env bash"
    echo "set -e"
    printf 'cd %q\n' "$REPO"
    [ -n "$PRELUDE" ] && echo "$PRELUDE"
    echo 'echo "host: $(hostname)   gpus: ${CUDA_VISIBLE_DEVICES:-unset}"'
    echo 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true'
    # Record the environment in every log. Over a multi-day run across a shared cluster this is
    # the only way to tell afterwards that all cells ran under the same torch/transformers, which
    # is the whole premise of putting their numbers in one table.
    cat <<'ENVCHK'
python -c 'import sys,torch,transformers;print("env: python",sys.version.split()[0],"torch",torch.__version__,"transformers",transformers.__version__,"cuda",torch.version.cuda,"avail",torch.cuda.is_available())' || true
ENVCHK
    echo "$*"
  } > "$jobsh"

  local args=(-J "$name" -q "$queue" -gpu "$GPU_RES" -n "$CORES" -W "$wall"
              -R "rusage[mem=$MEM_MB]" -o "$out" -e "$out")
  [ -n "$dep" ] && args+=(-w "ended($dep)")
  [ -n "$HOSTS" ] && args+=(-m "$HOSTS")

  if [ "$DRY" = 1 ]; then
    # to stderr: stdout is the job id this function returns, and the caller captures it
    printf '      bsub %s < %s\n' "${args[*]}" "$jobsh" >&2
    echo "DRYRUN"
    return 0
  fi
  local id
  id=$(bsub "${args[@]}" < "$jobsh" 2>&1 | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
  [ -n "$id" ] || { echo "    SUBMIT FAILED: $name (see above)" >&2; return 1; }
  echo "$id"
}

for ds in $DATASETS; do
  for mode in $MODES; do
    trw=""; [ "$mode" = temporal ] && trw="--trw"
    for s in $PSEEDS; do
      ck=$(ckpt_path "$ds" "$mode" "$s")
      ptname="pt_${mode}_${ds}_ps${s}"
      if [ -f "$REPO/$ck" ]; then
        pt_have=$((pt_have + 1))
      elif [ "$LIST_ONLY" = 1 ]; then
        echo "  pretrain  $ptname"; n_pt=$((n_pt + 1))
      else
        echo "  pretrain  $ptname"
        jid=$(submit "$ptname" "$QUEUE_PT" "$WALL_PT" "" \
          python pretrain.py --dataset "$ds" $trw --size "$SIZE" --device 0 \
                 --seed "$s" --tag "_ps${s}" $(pt_args "$ds") $EXTRA_PRETRAIN) || continue
        PT_JOBID["$ptname"]=$jid
        n_pt=$((n_pt + 1))
      fi

      for wl in $(wls_for "$ds"); do
        csv=$(csv_path "$ds" "$mode" "$s" "$wl")
        if [ -f "$REPO/$csv" ] && grep -q '^# BEST' "$REPO/$csv" 2>/dev/null; then
          ft_have=$((ft_have + 1)); continue
        fi
        ftname="ft_${mode}_${ds}_ps${s}_wl${wl}"
        if [ "$LIST_ONLY" = 1 ]; then
          echo "  finetune  $ftname"; n_ft=$((n_ft + 1)); continue
        fi
        echo "  finetune  $ftname${PT_JOBID[$ptname]+  (after ${PT_JOBID[$ptname]})}"
        # The guard: if the pretraining failed, exit 0 rather than failing on a missing file.
        submit "$ftname" "$QUEUE_FT" "$WALL_FT" "${PT_JOBID[$ptname]-}" \
          "[ -f $ck ] || { echo 'no checkpoint $ck -- pretraining did not finish; skipping'; exit 0; };" \
          python "$(ft_script "$ds")" --dataset "$ds" $trw --size "$SIZE" --device 0 \
                 --walk-len "$wl" --seed 0 --pretrain-tag "_ps${s}" --tag "_ps${s}_s0" \
                 $EXTRA_FT >/dev/null || true
        n_ft=$((n_ft + 1))
      done
    done
  done
done

echo
echo "pretrainings: $n_pt submitted to $QUEUE_PT, $pt_have already have a checkpoint"
echo "fine-tunes  : $n_ft submitted to $QUEUE_FT, $ft_have already finished"
if [ "$WLS" = match ]; then
  echo "walk lengths: matched to pretraining -- $(for d in $DATASETS; do printf '%s=%s ' "$d" "$(pt_walk_len "$d")"; done)"
fi
[ "$DRY" = 1 ] || [ "$LIST_ONLY" = 1 ] && exit 0

cat <<'TIPS'

  bjobs -w                      # queued and running, full job names
  bjobs -p                      # pending, with the reason
  bjobs -l <jobid>              # why one job is where it is
  bhist -l <jobid>              # what happened to a finished job
  tail -f logs/lsf/pt_*.out     # a running job's output
  bkill -J 'ft_*'               # drop all fine-tunes, keep the pretrainings

Each job's log starts with its host, CUDA_VISIBLE_DEVICES and nvidia-smi's GPU name and total
memory, so the first few lines tell you which card a cell landed on -- which is how you compare
L40 against H100 throughput on the same work. Every job body is also a standalone script under
logs/lsf/, so one cell can be re-run by hand without LSF:

  bash logs/lsf/ft_static_lanl14argus_ps1_wl16.sh

Escape hatches, should a cell still not fit (unlikely at 44.9 GB):

  WLS=16 HOSTS=nodeh100 ./submit_main_seeds.sh
  WLS=16 EXTRA_FT="--mini-bs 64" ./submit_main_seeds.sh

Then:  python show_results.py --metrics te_auc te_ap te_prec_r50 te_fp_day_r50
TIPS