#!/usr/bin/env bash
# Submit every Argus baseline to LSF and return. Nothing runs in your terminal.
#
# SEVEN EXPERIMENTS x SEEDS JOBS, all submitted at once:
#
#   lanl            argus_test_lanl.py                           random LANL split
#   lanl_ts         argus_test_lanl_ts.py                        temporal, default --cutoff 41
#   lanl_ts_dirty   argus_test_lanl_ts.py --cutoff 172            attack edges IN training
#   lanl_ts_clean   argus_test_lanl_ts.py --cutoff 172 --scrub    the same window, scrubbed
#   optc            argus_test_optc.py --dataset optc            original random edge split
#   optc_argus      argus_test_optc.py                           optc-argus, their temporal split
#   unsw            argus_test_unsw.py                           random UNSW split
#
# Each experiment gets SEEDS jobs (default 5). Every job is ONE training run at its own seed --
# `--runs 1`, always -- so a cell is SEEDS replications, one per GPU, exactly the shape the
# CyberGFM seed jobs use. Every job writes its own `_s<seed>` results file, so nothing overwrites
# anything; collect_argus_results.py globs them back together.
#
# The 172h cutoff makes the poisoning arms line up with CyberGFM's dirtyts/cleants splits, whose
# training window ends at ts 619200 = 172.0h exactly. Both arms score the identical test set, so
# the gap between them is attributable to training on unlabeled attack traffic.
#
#   ./run_all_argus.sh                       # submit all 35 jobs, print the ids, exit
#   ./run_all_argus.sh --dry-run             # print the bsub lines, submit nothing
#   ./run_all_argus.sh --only optc,unsw      # just those experiments
#   SEEDS=10 ./run_all_argus.sh              # 10 replications per experiment instead of 5
#   EXTRA="--beta 0" TAG=_b0 ./run_all_argus.sh    # the "their released code" arm, side by side
#
# Then, whenever the queue has drained:
#   python collect_argus_results.py --table         # -> argus_main.csv
set -uo pipefail

SEEDS=${SEEDS:-5}                # parallel jobs per experiment == replications; one seed each
SEED0=${SEED0:-1}                # first seed; jobs use SEED0 .. SEED0+SEEDS-1
# --runs is deliberately NOT a knob: one training run per job, parallelism comes from the seeds.
# Several runs in one process would serialise on a single GPU and, for argus_test_optc.py, share
# one results file.
EXTRA=${EXTRA:-""}               # appended to every run
TAG=${TAG:-""}                   # extra output suffix, for running variants side by side
LOGDIR=${LOGDIR:-logs/argus}
REPO=${REPO:-$PWD}

QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
WALL=${WALL:-"24:00"}
MEM_MB=${MEM_MB:-32000}
CORES=${CORES:-4}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}

DRY=0; LIST=0; FORCE=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --list|-n) LIST=1 ;;
    --force)   FORCE=1 ;;
    --only)    ONLY="$2"; shift ;;
    --only=*)  ONLY="${1#*=}" ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

[ -f "$REPO/argus.py" ] || { echo "run this from prior_works/ (no argus.py here)" >&2; exit 1; }
mkdir -p "$LOGDIR"

NAMES=(lanl lanl_ts lanl_ts_dirty lanl_ts_clean optc optc_argus unsw)

# name -> "script|flags|results-stem". The results stem is what the script writes BEFORE the
# per-seed suffix; argus_test_lanl_ts.py builds its own {clean|dirty}ts-{cutoff}h part and then
# appends --tag, so the stems below match what each script actually produces.
spec() {
  case "$1" in
    lanl)          echo "argus_test_lanl.py||argus_results_lanl14argus" ;;
    lanl_ts)       echo "argus_test_lanl_ts.py||argus_results_lanl14argus-dirtyts-41h" ;;
    lanl_ts_dirty) echo "argus_test_lanl_ts.py|--cutoff 172|argus_results_lanl14argus-dirtyts-172h" ;;
    lanl_ts_clean) echo "argus_test_lanl_ts.py|--cutoff 172 --scrub|argus_results_lanl14argus-cleants-172h" ;;
    optc)          echo "argus_test_optc.py|--dataset optc|argus_results_optc" ;;
    optc_argus)    echo "argus_test_optc.py|--dataset optc-argus|argus_results_optc-argus" ;;
    unsw)          echo "argus_test_unsw.py||argus_results_unsw" ;;
    *) return 1 ;;
  esac
}

selected() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; esac
  return 1
}

n_sel=0
for n in "${NAMES[@]}"; do selected "$n" && n_sel=$((n_sel + 1)); done
echo "$n_sel experiment(s) x $SEEDS seed(s) = $((n_sel * SEEDS)) jobs, one training run each"
[ -n "$EXTRA" ] && echo "extra flags on every run: $EXTRA"
echo

if [ "$LIST" = 1 ]; then
  for n in "${NAMES[@]}"; do
    selected "$n" || continue
    IFS='|' read -r script flags stem <<< "$(spec "$n")"
    for s in $(seq "$SEED0" $((SEED0 + SEEDS - 1))); do
      have=""; [ -f "$REPO/${stem}${TAG}_s${s}.csv" ] && have="   [exists, would skip]"
      echo "  $n s$s: python $script --runs 1 --device 0 --seed $s --tag ${TAG}_s${s} $flags $EXTRA$have"
    done
  done
  exit 0
fi

command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }

submitted=0; skipped=0; failed=0
for n in "${NAMES[@]}"; do
  selected "$n" || continue
  IFS='|' read -r script flags stem <<< "$(spec "$n")"

  for s in $(seq "$SEED0" $((SEED0 + SEEDS - 1))); do
    jtag="${TAG}_s${s}"
    # optc records --run-id in its CSV; the others have no such flag
    run_id=""; case "$n" in optc|optc_argus) run_id="--run-id $s" ;; esac
    results="${stem}${jtag}.csv"
    jname="argus_${n}${TAG}_s${s}"
    log="$LOGDIR/${jname}.out"
    jobsh="$LOGDIR/${jname}.sh"

    if [ -f "$REPO/$results" ] && [ "$FORCE" = 0 ]; then
      skipped=$((skipped + 1)); continue
    fi

    # No `set -e`: rc=$? after python has to be reachable, or a failed run records no timing.
    {
      echo '#!/usr/bin/env bash'
      printf 'cd %q\n' "$REPO"
      echo "$PRELUDE"
      echo 'echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}"'
      echo 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true'
      echo 'python -c "import sys,torch;print(\"env: python\",sys.version.split()[0],\"torch\",torch.__version__,\"cuda\",torch.version.cuda)" || true'
      echo 'start=$(date +%s)'
      echo 'echo "started: $(date -Is)"'
      echo "python $script --runs 1 --device 0 --seed $s --tag '$jtag' $run_id $flags $EXTRA"
      echo 'rc=$?'
      echo 'el=$(( $(date +%s) - start ))'
      # Runtime in the job's own output, which is where you will be looking.
      echo "printf '\\n=== %s: exit %d after %d:%02d:%02d (%d s) ===\\n' '$jname' \"\$rc\" \
\$((el/3600)) \$(((el%3600)/60)) \$((el%60)) \"\$el\""
      echo "printf '%s,%s,%d,%d,%s,%s,%d,%d:%02d:%02d,%d,%s,%s,%s\\n' '$jname' '$script' $s 1 \
\"\$(date -Is -d @\$start)\" \"\$(date -Is)\" \"\$el\" \$((el/3600)) \$(((el%3600)/60)) \
\$((el%60)) \"\$rc\" \"\$(hostname)\" '$log' '$results' >> '$LOGDIR/runtimes.csv'"
      echo 'exit $rc'
    } > "$jobsh"

    args=(-J "$jname" -q "$QUEUE" -gpu "$GPU_RES" -n "$CORES" -W "$WALL"
          -R "rusage[mem=$MEM_MB]" -o "$log" -e "$log")

    if [ "$DRY" = 1 ]; then
      echo "  bsub ${args[*]} < $jobsh"
      submitted=$((submitted + 1))
      continue
    fi

    id=$(bsub "${args[@]}" < "$jobsh" 2>&1 | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
    if [ -n "$id" ]; then
      printf '  %-28s %s\n' "$jname" "$id"
      submitted=$((submitted + 1))
    else
      echo "  SUBMIT FAILED $jname" >&2
      failed=$((failed + 1))
    fi
  done
done

[ -s "$LOGDIR/runtimes.csv" ] || echo \
  "job,script,seed,runs,started,finished,seconds,hms,exit,host,log,results" > "$LOGDIR/runtimes.csv"

echo
echo "submitted $submitted, skipped $skipped (results exist; --force to rerun), failed $failed"
echo
echo "watch    : bjobs -J 'argus_*' -o 'jobid stat job_name run_time' -noheader"
echo "logs     : $LOGDIR/argus_<name>_s<seed>.out   (each ends with its own runtime banner)"
echo "runtimes : $LOGDIR/runtimes.csv               (one row per job, appended as it finishes)"
echo "collect  : python collect_argus_results.py --table --expected $SEEDS"
[ "$failed" = 0 ] || exit 1