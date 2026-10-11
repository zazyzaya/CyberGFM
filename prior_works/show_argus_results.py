'''
Pool every Argus results file into one CSV with show_main_results.py --csv's exact columns, so
the Argus rows drop into the same spreadsheet as the CyberGFM rows.

    experiment,dataset,walk_len,n,n_expected,epoch,te_auc,te_auc_se,te_ap,te_ap_se,
    te_prec_r50,te_prec_r50_se,te_fp_day_r50,te_fp_day_r50_se

The four scripts record `auc`, `ap`, and (all but OpTC) `prec_r50` and `fp_day_r50`; those are
renamed to the te_* names show_main_results uses. walk_len is blank -- Argus has no walk length,
and leaving the column present but empty keeps the header identical so the two CSVs concatenate.

Each (experiment, dataset) pools every replication across every seed job: run_all_argus.sh runs
one training run per job and gives each its own `_s<seed>` output file, so a cell is as many
replications as there were seeds. The mean and SE are computed from the raw per-run rows; pandas
also writes `mean` and `sem` rows into each file and this skips them, so that a file holding
several runs cannot contribute its own average as if it were another replication.

    python collect_argus_results.py                              # -> argus_main.csv
    python collect_argus_results.py --csv argus_main.csv --table
    python collect_argus_results.py --expected 25
'''
from argparse import ArgumentParser
import csv
import glob
import math
import os
import re

# show_main_results.py:232 write_csv, exactly
FIELDS = ['experiment', 'dataset', 'walk_len', 'n', 'n_expected', 'epoch']
METRICS = ['te_auc', 'te_ap', 'te_prec_r50', 'te_fp_day_r50']   # show_main_results.DEFAULT_SHOW
for _m in METRICS:
    FIELDS += [_m, f'{_m}_se']

# what the Argus scripts call them -> what show_main_results calls them
RENAME = {'auc': 'te_auc', 'ap': 'te_ap',
          'prec_r50': 'te_prec_r50', 'fp_day_r50': 'te_fp_day_r50'}

# results filename -> (experiment, dataset). The _s<seed> suffix run_all_argus.sh adds is stripped
# first, so all of a cell's seed jobs land in one group.
PATTERNS = [
    (re.compile(r'^argus_results_lanl14argus-(dirty|clean)ts-(?P<h>[\d.]+)h$'),
     lambda m: ('argus', f'lanl14argus-{m.group(1)}ts-{m.group("h")}h')),
    (re.compile(r'^argus_results_(?P<ds>.+)$'), lambda m: ('argus', m.group('ds'))),
]


def mean_se(xs):
    '''Mean and standard error over replications; SE is undefined for n < 2.'''
    xs = [x for x in xs if x == x]                      # drop NaN
    n = len(xs)
    if n == 0:
        return None, None, 0
    mu = sum(xs) / n
    if n < 2:
        return mu, None, n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var) / math.sqrt(n), n


def read_runs(path):
    '''
    Per-run rows from one results file, as dicts of floats.

    Skips the `mean` and `sem` rows pandas appends (their index column is not a number), and
    OpTC's non-numeric bookkeeping columns.
    '''
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            idx = next(iter(row.values()))              # the pandas index, or OpTC's run_id
            first = next(iter(row))
            if first in ('', 'Unnamed: 0') and not str(idx).strip().lstrip('-').isdigit():
                continue                                # the mean / sem rows
            out = {}
            for k, v in row.items():
                if k in (None, '', 'Unnamed: 0'):
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    continue                            # host, finished, dataset, ...
            if out:
                rows.append(out)
    return rows


def classify(stem):
    stem = re.sub(r'_s\d+$', '', stem)                  # the per-seed-job suffix
    for pat, fn in PATTERNS:
        m = pat.match(stem)
        if m:
            return fn(m)
    return None


def main():
    ap = ArgumentParser()
    ap.add_argument('--glob', default='argus_results_*.csv',
                    help='results files to pool (default: every argus_results_*.csv here)')
    ap.add_argument('--csv', default='argus_main.csv', help='output path')
    ap.add_argument('--expected', type=int, default=5,
                    help='n_expected column: one run per seed job, so this is the seed count '
                         '(default 5)')
    ap.add_argument('--table', action='store_true', help='also print it readably')
    args = ap.parse_args()

    groups, files = {}, {}
    for path in sorted(glob.glob(args.glob)):
        stem = os.path.basename(path)[:-4]
        key = classify(stem)
        if key is None:
            print(f'  ? skipping {path} (name does not parse)')
            continue
        runs = read_runs(path)
        if not runs:
            print(f'  ! {path} has no completed runs')
            continue
        groups.setdefault(key, []).extend(runs)
        files.setdefault(key, []).append(os.path.basename(path))

    if not groups:
        raise SystemExit(f'nothing matched {args.glob} in {os.getcwd()}')

    out_rows = []
    for key in sorted(groups, key=lambda k: (k[1], k[0])):
        exp, ds = key
        runs = groups[key]
        row = {'experiment': exp, 'dataset': ds, 'walk_len': '',
               'n': len(runs), 'n_expected': args.expected}

        ep = mean_se([r['epoch'] for r in runs if 'epoch' in r])
        row['epoch'] = f'{ep[0]:.4f}' if ep[0] is not None else ''

        for src, dst in RENAME.items():
            mu, se, n = mean_se([r[src] for r in runs if src in r])
            row[dst] = f'{mu:.4f}' if mu is not None else ''
            row[f'{dst}_se'] = f'{se:.4f}' if se is not None else ''
        out_rows.append(row)
        print(f'  {ds:34s} n={len(runs):3d} from {len(files[key])} file(s)')

    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(out_rows)
    print(f'\nWrote {args.csv}  ({len(out_rows)} rows, columns identical to '
          f'show_main_results.py --csv)')

    if args.table:
        print()
        hdr = f'{"dataset":34s} {"n":>4s} ' + ' '.join(f'{m[3:]:>20s}' for m in METRICS)
        print(hdr)
        print('-' * len(hdr))
        for r in out_rows:
            cells = []
            for m in METRICS:
                if r[m] == '':
                    cells.append(f'{"--":>20s}')
                elif r[f'{m}_se'] == '':
                    cells.append(f'{float(r[m]):>20.4f}')
                else:
                    cells.append(f'{float(r[m]):>11.4f} +-{float(r[f"{m}_se"]):7.4f}')
            print(f'{r["dataset"]:34s} {r["n"]:>4d} ' + ' '.join(cells))
        print('\nte_prec_r50 / te_fp_day_r50 are "--" for OpTC: argus_test_optc.py records only '
              'auc and ap.')


if __name__ == '__main__':
    main()