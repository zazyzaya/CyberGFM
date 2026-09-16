'''
Data-only sanity checks for the bugs described in CODE_REVIEW.md.
Needs only torch + torch_geometric (no GPU, no torch_cluster, no model code).

Run from the repo root:
    python sanity_checks.py
    python sanity_checks.py --datasets lanl14argus lanl14argus-ts optc-ts
'''
from argparse import ArgumentParser
import math
import os

import torch


def load(ds, split):
    fn = f'data/{ds}_tgraph_{split}.pt'
    if not os.path.exists(fn):
        return None
    return torch.load(fn, weights_only=False)


def has(d, k):
    return k in d.keys() and getattr(d, k) is not None


def check_edge_feature_offset(ds, tr):
    '''Bug 2: _single_iter returns edge_attr without the + num_nodes offset used in rw().'''
    if not has(tr, 'edge_attr'):
        print('  [edge-feat offset] no edge_attr; n/a')
        return
    n = tr.x.size(0)
    lo, hi = int(tr.edge_attr.min()), int(tr.edge_attr.max())
    print(f'  [edge-feat offset] num_nodes={n}, raw edge_attr range=[{lo}, {hi}]')
    if lo < n:
        print(f'    -> CONFIRMED (if features are used): raw values {lo}..{min(hi, n-1)} alias node ids; '
              f'values >= {n} map to the wrong feature tokens (shifted by {n}).')
    else:
        print('    -> raw values all exceed num_nodes; still misaligned by num_nodes unless GNNEmbedding compensates.')


def check_temporal_context(ds, tr, te, delta):
    '''Bug 5: min_ts = ts - DELTA on the *training* graph leaves no context under temporal splits.'''
    tr_max = int(tr.ts.max())
    starved = (te.ts - delta > tr_max)
    mal = te.label.bool() if has(te, 'label') else torch.zeros_like(starved)
    print(f'  [temporal eval ctx] DELTA={delta}: tr.ts.max={tr_max}, te.ts range=[{int(te.ts.min())}, {int(te.ts.max())}]')
    print(f'    test edges whose whole walk window is after training ends: '
          f'{starved.float().mean().item():.1%} (malicious: {starved[mal].float().mean().item() if mal.any() else float("nan"):.1%})')


def check_unseen_nodes(tr, te):
    seen = torch.zeros(tr.x.size(0), dtype=torch.bool)
    seen[tr.src] = True
    seen[tr.col] = True
    unseen = ~(seen[te.src] & seen[te.col])
    mal = te.label.bool()
    print(f'  [transductive] test edges touching a node never seen in training: '
          f'{unseen.float().mean().item():.2%} overall, '
          f'{unseen[mal].float().mean().item() if mal.any() else float("nan"):.2%} of malicious')


def check_unseen_baseline(tr, te):
    '''Trivial detector: score 1 if either endpoint never appears in training, else 0 (+ tiny random tiebreak).'''
    from sklearn.metrics import roc_auc_score, average_precision_score
    seen = torch.zeros(tr.x.size(0), dtype=torch.bool)
    seen[tr.src] = True
    seen[tr.col] = True
    score = (~(seen[te.src] & seen[te.col])).float() + torch.rand(te.col.size(0)) * 1e-3
    y = te.label.numpy()
    if y.sum() == 0:
        return
    print(f'  [unseen-host baseline] AUC {roc_auc_score(y, score.numpy()):.4f}  '
          f'AP {average_precision_score(y, score.numpy()):.4f}')


def check_poison_call(ds, te, edge_features, poison_arg=5):
    '''Bug 1: poison(tr, te, has_edge_attrs, percent, fname) is called as poison(tr, te, args.poison, edge_features, DATASET).'''
    n_mal = int(te.label.sum())
    as_called_percent = edge_features  # bool lands in the `percent` slot
    as_called = int(n_mal * as_called_percent)
    intended = int(n_mal * poison_arg / 100)
    print(f'  [poison] --poison {poison_arg}: malicious test edges={n_mal}; '
          f'injected as committed={as_called}; intended={intended}; '
          f'branch taken as committed={"attr" if poison_arg else "nonattr"} '
          f'(dataset has edge_attr: {has(te, "edge_attr")})')


def check_token_accounting():
    '''Bug 4: processed_tokens only counts the final micro-batch of each accumulated update.'''
    BS = 1024
    cfgs = {
        'LANL static (MINI_BS=512)': 512,
        'LANL temporal (MINI_BS=128)': 128,
        'UNSW / tiny (MINI_BS=1024)': 1024,
        'OpTC (MINI_BS=1035)': 1035,
    }
    print('[token accounting] actual tokens seen per reported token:')
    for k, mb in cfgs.items():
        print(f'    {k}: ~{math.ceil(BS / mb)}x')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['lanl14argus', 'lanl14argus-dirty', 'unsw', 'optc'])
    args = ap.parse_args()

    check_token_accounting()
    for ds in args.datasets:
        tr, te = load(ds, 'tr'), load(ds, 'te')
        if tr is None or te is None:
            print(f'\n== {ds}: files not found, skipping')
            continue
        print(f'\n== {ds}: {tr.col.size(0)} train edges, {te.col.size(0)} test edges, {int(te.label.sum())} malicious')
        check_edge_feature_offset(ds, tr)
        # Mirrors the committed flag: only --unsw / --argus turn features on
        committed_ef = ds in ('unsw', 'lanl14argus')
        if has(tr, 'edge_attr') and not committed_ef and ds.startswith('lanl'):
            print('  [edge-feat flag] CONFIRMED: this split has edge_attr on disk, but pretrain.py / '
                  'lp_finetune.py run it with edge_features=False')
        check_poison_call(ds, te, committed_ef)
        check_unseen_nodes(tr, te)
        check_unseen_baseline(tr, te)
        if ds.startswith('lanl'):
            check_temporal_context(ds, tr, te, delta=60 * 60)       # lp_finetune DELTA
        elif ds.startswith('optc'):
            check_temporal_context(ds, tr, te, delta=60 * 60 * 24)  # lp_finetune DELTA