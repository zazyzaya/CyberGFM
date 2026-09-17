'''
Aggregate fine-tuning results across seeds.

Reads results/<method>-<mode>/<dataset>/*_wl<W>_<ckpt>_s<seed>.csv (one CSV per run, one row per
epoch) and groups runs that differ only by seed. For each group it reports mean ± standard error
over seeds for two ways of picking a row from each run:

  final : the last epoch of a finished run
  best  : the epoch with the highest validation AUC (or --select-by ap)

A run counts as finished once its CSV has the "# BEST" summary line. Runs still in progress are
left out of the summary (the n column shows finished/total) unless --include-running is given,
in which case their latest epoch stands in for "final".

    python summarize_results.py
    python summarize_results.py --dataset optc-ts --csv optc_summary.csv
    python summarize_results.py --filter cls-static --curves curves/
'''
from argparse import ArgumentParser
from collections import defaultdict
import csv
import glob
import math
import os
import re

METRICS = ('te_auc', 'te_ap', 'va_auc', 'va_ap')
NAME_RE = re.compile(r'^(?P<prefix>.+)_wl(?P<wl>\d+)_(?P<ckpt>[^_]+)_s(?P<seed>\d+)\.csv$')


def read_run(path):
    rows, finished = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                finished = finished or line.startswith('# BEST')
                continue
            if line.startswith('epoch'):
                continue
            parts = line.split(',')
            if len(parts) != 2 + len(METRICS):
                continue  # partially written line
            try:
                row = dict(epoch=int(parts[0]), updates=int(parts[1]))
                row.update({m: float(v) for m, v in zip(METRICS, parts[2:])})
            except ValueError:
                continue
            rows.append(row)
    return rows, finished


def mean_se(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float('nan'), float('nan'), 0
    mu = sum(vals) / n
    if n == 1:
        return mu, float('nan'), 1
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var / n), n


def fmt(mu, se):
    if math.isnan(mu):
        return '-'
    return f'{mu:.4f}' if math.isnan(se) else f'{mu:.4f} ± {se:.4f}'


def collect(root, dataset=None, filt=None):
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, '*', '*', '*.csv'))):
        exp_dir, ds = path.split(os.sep)[-3:-1]
        if dataset and ds != dataset:
            continue
        if filt and filt not in path:
            continue
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        key = (exp_dir, ds, m['prefix'], int(m['wl']), m['ckpt'])
        rows, finished = read_run(path)
        groups[key].append(dict(seed=int(m['seed']), rows=rows, finished=finished, path=path))
    return groups


def pick(rows, how, select_key):
    if not rows:
        return None
    if how == 'final':
        return rows[-1]
    best = rows[0]
    for r in rows[1:]:
        if r[select_key] > best[select_key]:  # strict: earliest epoch wins ties, as in ResultTracker
            best = r
    return best


def summarize(groups, include_running, select_key):
    table = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[4], k[3])):
        runs = groups[key]
        usable = [r for r in runs if r['rows'] and (r['finished'] or include_running)]
        row = dict(zip(('experiment', 'dataset', 'prefix', 'walk_len', 'pretrain_ckpt'), key))
        row['n'] = f'{len(usable)}/{len(runs)}'
        row['epochs'] = '/'.join(sorted({str(r['rows'][-1]['epoch']) for r in usable})) or '-'
        for how in ('final', 'best'):
            picked = [pick(r['rows'], how, select_key) for r in usable]
            for m in METRICS:
                mu, se, _ = mean_se([p[m] for p in picked])
                row[f'{how}_{m}'] = mu
                row[f'{how}_{m}_se'] = se
            if how == 'best':
                mu, se, _ = mean_se([p['epoch'] for p in picked])
                row['best_epoch'] = mu
        table.append(row)
    return table


def print_table(table, how, select_key):
    title = 'FINAL EPOCH' if how == 'final' else f'BEST EPOCH BY {select_key.upper()}'
    cols = ['experiment', 'dataset', 'prefix', 'pretrain_ckpt', 'walk_len', 'n', 'epochs']
    if how == 'best':
        cols.append('best_epoch')
    header = cols + [m for m in METRICS]
    lines = []
    for r in table:
        cells = [str(r[c]) if c != 'best_epoch' else ('-' if math.isnan(r[c]) else f'{r[c]:.1f}') for c in cols]
        cells += [fmt(r[f'{how}_{m}'], r[f'{how}_{m}_se']) for m in METRICS]
        lines.append(cells)
    widths = [max(len(h), *(len(l[i]) for l in lines)) if lines else len(h) for i, h in enumerate(header)]
    print(f'\n== {title} (mean ± standard error over seeds) ==')
    print('  '.join(h.ljust(w) for h, w in zip(header, widths)))
    prev = None
    for r, l in zip(table, lines):
        group = (r['experiment'], r['dataset'], r['prefix'], r['pretrain_ckpt'])
        if prev is not None and group != prev:
            print()
        prev = group
        print('  '.join(c.ljust(w) for c, w in zip(l, widths)))


def write_csv(table, path):
    fields = ['experiment', 'dataset', 'prefix', 'pretrain_ckpt', 'walk_len', 'n', 'epochs', 'best_epoch']
    for how in ('final', 'best'):
        for m in METRICS:
            fields += [f'{how}_{m}', f'{how}_{m}_se']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(table)
    print(f'\nWrote {path}')


def write_curves(groups, out_dir, include_running):
    '''Per-epoch mean ± standard error for each group, over the runs that reached that epoch.'''
    os.makedirs(out_dir, exist_ok=True)
    for (exp_dir, ds, prefix, wl, ckpt), runs in groups.items():
        usable = [r for r in runs if r['rows'] and (r['finished'] or include_running)]
        if not usable:
            continue
        by_epoch = defaultdict(list)
        for r in usable:
            for row in r['rows']:
                by_epoch[row['epoch']].append(row)
        path = os.path.join(out_dir, f'{exp_dir}_{ds}_{prefix}_wl{wl}_{ckpt}.csv')
        with open(path, 'w') as f:
            f.write('epoch,updates,n,' + ','.join(f'{m},{m}_se' for m in METRICS) + '\n')
            for e in sorted(by_epoch):
                rows = by_epoch[e]
                cells = [str(e), str(rows[0]['updates']), str(len(rows))]
                for m in METRICS:
                    mu, se, _ = mean_se([x[m] for x in rows])
                    cells += [f'{mu}', '' if math.isnan(se) else f'{se}']
                f.write(','.join(cells) + '\n')
    print(f'Wrote per-epoch curves to {out_dir}/')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--root', default='results')
    ap.add_argument('--dataset', help='only this dataset, e.g. optc-ts')
    ap.add_argument('--filter', help='only files whose path contains this string, e.g. cls-temporal')
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc', help='validation metric for "best"')
    ap.add_argument('--show', choices=['final', 'best', 'both'], default='both')
    ap.add_argument('--include-running', action='store_true',
                    help='include unfinished runs, using their latest epoch as "final"')
    ap.add_argument('--csv', help='also write the summary table to this CSV')
    ap.add_argument('--curves', help='also write per-epoch mean ± SE curves to this directory')
    args = ap.parse_args()

    select_key = f'va_{args.select_by}'
    groups = collect(args.root, args.dataset, args.filter)
    if not groups:
        raise SystemExit(f'No result CSVs found under {args.root}/')

    table = summarize(groups, args.include_running, select_key)
    for how in (('final', 'best') if args.show == 'both' else (args.show,)):
        print_table(table, how, select_key)
    if args.csv:
        write_csv(table, args.csv)
    if args.curves:
        write_curves(groups, args.curves, args.include_running)