'''
Argus on UNSW-NB15.

The model comes from argus.py, shared with the LANL and OpTC scripts. Read its docstring for the
six fixes and especially for the SCORE CONVENTION: decode() returns a reconstruction probability
where HIGH MEANS NORMAL, so the anomaly score is its complement.

UNSW's split is RANDOM, not temporal (preprocess_unsw.py:partition_tgraph -- torch.randperm, an
80/10/10 edge partition with every malicious edge forced into test). Test edges are therefore
interleaved in time with training edges, so building all three splits on the training timestamp
axis is legitimate here and `zs[i]` IS the contemporaneous embedding for test snapshot i. This is
the same structure as argus_test_lanl.py and deliberately NOT the per-split axis plus GRU
continuation that argus_test_optc.py needs for its temporal split.

WHAT CHANGED FROM THE PREVIOUS VERSION OF THIS FILE:

0. `SPEEDTEST = True` was a module-level constant and the speedtest block ends in `exit()` after
   the first batch of the first epoch. As committed, THIS SCRIPT NEVER EVALUATED ANYTHING -- it
   wrote one timing row and quit. Any argus_results_unsw.csv on disk came from a hand-edited
   copy, so whatever produced it is not what the repository contains. It is now an opt-in
   `--speedtest` flag, default off.

1. LOSS POLARITY was inverted: `neg_pred` held the observed edges and `pos_pred` the random
   pairs, so APLoss maximised AP for random pairs. Now in argus.py, with observed edges as the
   positive class, and this file negates at eval. On LANL this one fix was worth ~50x in AP.

2. DECODE had no sigmoid, so APLoss's 0.8 margin applied to unbounded dot products -- while the
   test block separately inlined its own dot product and applied a sigmoid, so training and
   scoring were on different scales. Both now go through argus.py's decode().

3. THE AGGREGATION RAN ON EVERY SNAPSHOT, unconditionally, directly below a docstring saying
   "Skipped in UNSW". Their released code defines it and never calls it. Off unless --aggregate.
   The `# Comment out for UNSW` markers on c2/c3 were likewise never actually commented out; the
   three-GCN encoder is kept, matching every other Argus script here.

4. DIMENSIONS were 128/64; their main.py uses 32/16. --h-dim/--z-dim, defaulting to 32/16.

5. NEGATIVE SAMPLING had no true-edge rejection. UNSW has only ~50 nodes, so a uniformly drawn
   pair is a real edge a large fraction of the time -- this matters far more here than on LANL.
   argus.py rejects and redraws, as their fast_negative_sampling does.

6. `range(len(tr.edge_index) // BS)` truncated the final partial batch. Now `range(0, n, BS)`.

7. VALIDATION SCORED `zs[:42]` -- a hardcoded 42 left over from LANL's 42-hour window, which on
   UNSW just means "the first 42 snapshots". It now scores the whole validation window.

8. EDGE FEATURES WERE NORMALISED PER SPLIT: each call did
   `raw_edge_attr / raw_edge_attr.max(dim=0).values` against its OWN maximum, so train, val and
   test were on three different scales and the test scaling depended on the test data. The
   training maximum is now computed once and applied to all three.

9. `DEVICE = 1` was hardcoded, which fails outright on any host that exposes one GPU (every job
   under `bsub -gpu "num=1"`). Now --device.

10. No argparse, no seeding, and 10 replications in one process. Now one run per process by
    default with --runs to override, and --seed for reproducibility.

GAMMA. The committed code used APLoss gamma=0.1 here and on OpTC, but 0.01 on LANL (the 0.1 line
survives as a comment in argus_test_lanl.py). That per-dataset difference is preserved rather
than harmonised: --gamma defaults to 0.1 on this script. It is an explicit flag so the value is
always in the run's argparse echo.

    python argus_test_unsw.py --device 0 --runs 5
    python argus_test_unsw.py --device 0 --speedtest
'''
from argparse import ArgumentParser
from collections import defaultdict
import os
import sys
import time

import torch
from torch_geometric.data import Data
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score

# One shared model definition; a fix there lands in every Argus script at once.
from argus import Argus, SOAP, append_result, run_record

sys.path.insert(0, '..')
from metrics import detection_metrics  # noqa: E402  -- same module the CyberGFM runs use


def get_args():
    ap = ArgumentParser(description='Argus on UNSW-NB15 (random edge split). Defaults reproduce '
                                    'their released code at their published dimensions.')
    # runtime
    ap.add_argument('--device', type=int, default=0, help='CUDA index, or -1 for CPU')
    ap.add_argument('--runs', type=int, default=5, help='independent training runs to average')
    ap.add_argument('--epochs', type=int, default=1000,
                    help='with --patience 100, effectively trains to convergence. Their '
                         'main.py uses 100 epochs, patience 3')
    ap.add_argument('--patience', type=int, default=100,
                    help='epochs without validation improvement before stopping '
                         '(their main.py: 3)')
    ap.add_argument('--threads', type=int, default=0, help='torch.set_num_threads; 0 leaves it')
    ap.add_argument('--out-dir', default='../results/argus',
                    help='results go to <out-dir>/<experiment><tag>.csv, one row per '
                         'run, shared by every seed of the experiment')
    ap.add_argument('--seed', type=int, default=None, help='torch seed; omit for unseeded runs')

    # optimisation
    ap.add_argument('--lr', type=float, default=0.01, help='paper and their main.py: 0.01')

    # model -- Table 3 and the "ARGUS Hyper-parameters" paragraph
    ap.add_argument('--h-dim', type=int, default=32, help='their main.py: -H/--hidden 32')
    ap.add_argument('--z-dim', type=int, default=16,
                    help='paper: "The node embedding size is 16"')
    ap.add_argument('--gru-layers', type=int, default=2,
                    help='paper: "The RNN model is a two-layer GRU". Their recurrent.GRU leaves '
                         'hidden_units at 1 and main.py never overrides it, so 2 follows the '
                         'PAPER and 1 reproduces their CODE')
    ap.add_argument('--s', type=int, default=5,
                    help='neighbours sampled in the aggregation step (only used with --aggregate)')
    ap.add_argument('--margin', type=float, default=0.8,
                    help='APLoss margin m (paper: 0.8 for every dataset)')
    ap.add_argument('--beta', type=float, default=0.01,
                    help='weight on L_DEC in loss = L_ap + beta * L_DEC '
                         '(NOT in the paper -- Argus never evaluated UNSW. 0.01 is the smaller of their two published values, so the guessed term has the least influence). Their released code has NO L_DEC term at all, so any '
                         'nonzero value runs this repo\'s per-edge BCE reading of eq. 12, which '
                         'is a GUESS; --beta 0 reproduces their code exactly')
    ap.add_argument('--gamma', type=float, default=0.1,
                    help='APLoss moving-average factor (NOT in the paper; 0.1 is what this file hardcoded, matching OpTC)')
    ap.add_argument('--aggregate', action='store_true',
                    help='apply the neighbour aggregation before decoding. Their repo defines it '
                         '(inner_forward_sm) and never calls it, so it is OFF by default')
    ap.add_argument('--neg-sampling', choices=['uniform', 'degree'], default='uniform',
                    help="negative pairs for L_ap and the val LP metric. The paper allows "
                         "'uniform random or proportional to the node degrees'; their code does "
                         "uniform only. 'degree' is a DIAGNOSTIC for the degree shortcut, not "
                         'their method -- see Argus.sample_negatives in argus.py')
    ap.add_argument('--no-neg-filter', action='store_true',
                    help='skip rejecting sampled negatives that are real edges. Their '
                         'fast_negative_sampling rejects them; this restores the unfiltered form')

    ap.add_argument('--bs', type=int, default=64, help='snapshots per optimizer step')
    ap.add_argument('--speedtest', action='store_true',
                    help='time one batch, append to argus_speedtest.csv, exit. This used to be a '
                         'module constant set to True, so the script never evaluated.')
    ap.add_argument('--data', default='../data', help='directory holding unsw_tgraph_*_raw.pt')
    ap.add_argument('--tag', default='',
                    help='variant suffix for the results file, e.g. --tag _b0 alongside '
                         '--beta 0. Seeds of one experiment share a file; do not tag per seed')
    return ap.parse_args()


def to_snapshots(g, ts, add_csr=False, feat_max=None):
    '''
    Bucket a split's edges by timestamp onto the axis `ts`.

    feat_max is the per-column maximum used to normalise the edge features. It MUST come from the
    training split for all three calls: normalising each split by its own maximum put them on
    different scales and made the test features depend on the test data. Returns the maximum it
    used so the caller can thread the training one through.
    '''
    ei = torch.stack([g.src, g.col])
    if feat_max is None:
        feat_max = g.raw_edge_attr.max(dim=0).values
    feat_max = feat_max.clone()
    feat_max[feat_max == 0] = 1.0            # a constant-zero column must not divide by zero
    ea_all = g.raw_edge_attr / feat_max

    eis, eas, idxs, ptrs, y = [], [], [], [], []
    for t in ts:
        mask = g.ts == t
        ei_t = ei[:, mask]
        eis.append(ei_t)
        eas.append(ea_all[mask])

        if add_csr:
            csr_dict = defaultdict(list)
            for i in range(ei_t.size(1)):
                src, dst = ei_t[:, i]
                csr_dict[src.item()].append(dst.item())
            idx, ptr = [0], []
            for i in range(g.x.size(0)):
                ptr += csr_dict[i]
                idx.append(idx[-1] + len(csr_dict[i]))
            idxs.append(torch.tensor(idx))
            ptrs.append(torch.tensor(ptr))

        if 'label' in g.keys():
            y.append(g.label[mask])

    x = torch.eye(g.x.size(0))
    out = Data(x=x, edge_index=eis, label=y, eas=eas, idxs=idxs, ptrs=ptrs,
               snapshot_ts=torch.as_tensor(ts))
    return out, feat_max


def build_model(tr, args):
    return Argus(tr.x.size(0), tr.eas[0].size(1), args.h_dim, args.z_dim, args.device,
                 s=args.s, pos_samples=sum(e.size(1) for e in tr.edge_index),
                 margin=args.margin, gamma=args.gamma, aggregate=args.aggregate,
                 gru_layers=args.gru_layers, beta=args.beta,
                 neg_sampling=args.neg_sampling, neg_filter=not args.no_neg_filter)


def train(tr, va, te, args):
    model = build_model(tr, args)
    opt = SOAP(model.parameters(), lr=args.lr, mode='adam', weight_decay=0.0)

    best = (0, 0, 0, {})
    best_snooped = (0, 0)
    no_progress = 0
    n_snaps = len(tr.edge_index)
    print(f'{n_snaps} snapshots, {sum(e.size(1) for e in tr.edge_index):,} training edges')

    for e in range(args.epochs):
        fwd_time = bwd_time = loss_time = step_time = 0
        # range(0, n, BS): the old range(n // BS) silently dropped the final partial batch
        for st_i in range(0, n_snaps, args.bs):
            en_i = min(st_i + args.bs, n_snaps)
            model.train()
            opt.zero_grad()

            st = time.time()
            zs = model.forward(tr.x, tr.edge_index[st_i:en_i], tr.eas[st_i:en_i],
                               tr.idxs[st_i:en_i] if tr.idxs else [],
                               tr.ptrs[st_i:en_i] if tr.ptrs else [])
            fwd_time += time.time() - st

            st = time.time()
            loss = model.calc_loss_argus(zs, tr.edge_index[st_i:en_i])
            loss_time += time.time() - st

            st = time.time()
            loss.backward()
            bwd_time += time.time() - st

            st = time.time()
            opt.step()
            step_time += time.time() - st
            print(f'[{e}: {st_i}-{en_i}/{n_snaps}] Loss: {loss.item():0.4f}', flush=True)

        if args.speedtest:
            append_result(f'{args.out_dir}/speedtest.csv',
                          dict(experiment=f'unsw{args.tag}', fwd=fwd_time,
                               loss=loss_time, bwd=bwd_time, step=step_time))
            return None

        with torch.no_grad():
            model.eval()
            zs = model.forward(tr.x, tr.edge_index, tr.eas, tr.idxs, tr.ptrs)

            # Validation is a LINK-PREDICTION metric: observed edges vs. random pairs, scored on
            # the RAW reconstruction probability with observed edges as the positive class. It is
            # the one metric not negated, and it uses no malicious labels.
            # Scores the whole window -- the old code passed zs[:42], a LANL leftover.
            pos, neg = model.validate(zs, va.edge_index)
            labels = torch.cat([torch.ones(pos.numel()), torch.zeros(neg.numel())]).numpy()
            va_auc = auc_score(labels, torch.cat([pos, neg]).numpy())
            va_ap = ap_score(labels, torch.cat([pos, neg]).numpy())
            print(f'\tVal AUC: {va_auc:0.4f}, AP: {va_ap:0.4f}')

            # Test edges live inside the training window (random split), so zs[i] is the
            # contemporaneous embedding. Goes through model.decode so the sigmoid applies; the
            # old code inlined the dot product here and bypassed it.
            preds = []
            for i in range(len(te.edge_index)):
                if te.edge_index[i].size(1) == 0:
                    continue
                preds.append(model.decode(te.edge_index[i].to(args.device), zs[i]).cpu())

            # HIGH MEANS NORMAL, so the anomaly score is the complement -- their score_stats does
            # exactly this (`scores = 1 - scores`) before computing auc/ap.
            preds = (1 - torch.cat(preds)).numpy()
            y = torch.cat([l for l, ei in zip(te.label, te.edge_index) if ei.size(1)]) \
                     .clamp(0, 1).numpy()

            auc = auc_score(y, preds)
            ap = ap_score(y, preds)
            n_per = torch.as_tensor([ei.size(1) for ei in te.edge_index if ei.size(1)])
            keep = torch.as_tensor([i for i, ei in enumerate(te.edge_index) if ei.size(1)])
            te_ts = te.snapshot_ts[keep].repeat_interleave(n_per).numpy()
            extra = detection_metrics(y, preds, ap, ts=te_ts, recall=0.5, prefix='')
            print(f'\tTe AUC: {auc:0.4f}, AP: {ap:0.4f}', end='', flush=True)

            if va_auc + va_ap > best[0]:
                best = (va_auc + va_ap, auc, ap, extra)
                print('*')
                no_progress = 0
            else:
                print()
                no_progress += 1

            # Tracked without using validation, because validation does not track test here.
            # Data snooping: an upper bound, reported separately, never as the headline.
            if auc > best_snooped[0]:
                best_snooped = (auc, ap)

            if no_progress > args.patience:
                break

    out = {'auc': best[1], 'ap': best[2], 'auc_last': auc, 'ap_last': ap,
           'auc_snooped': best_snooped[0], 'ap_snooped': best_snooped[1], **best[3]}
    print(f'Best: AUC {best[1]:0.4f}, AP {best[2]:0.4f}')
    return out


if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.device = args.device if args.device >= 0 else 'cpu'

    raw = {n: torch.load(f'{args.data}/unsw_tgraph_{n}_raw.pt', weights_only=False)
           for n in ('tr', 'va', 'te')}
    if 'label' not in raw['te'].keys():
        raise SystemExit('the test split has no `label`, so no detection metric is computable. '
                         'Rebuild it with preprocess_unsw.py:partition_tgraph.')
    ts = raw['tr'].ts.unique()                       # legitimate here: the split is random
    tr, feat_max = to_snapshots(raw['tr'], ts, add_csr=True)
    va, _ = to_snapshots(raw['va'], ts, feat_max=feat_max)   # training scale, not their own
    te, _ = to_snapshots(raw['te'], ts, feat_max=feat_max)

    out = f'{args.out_dir}/unsw{args.tag}.csv'
    for r in range(args.runs):
        seed = None if args.seed is None else args.seed + r
        if seed is not None:
            torch.manual_seed(seed)
        print(f'\n===== run {r} (seed {seed}) =====')
        t0 = time.time()
        res = train(tr, va, te, args)
        if res is None:                              # --speedtest
            break
        append_result(out, {**run_record(args, seed, (time.time() - t0) / 60,
                                         dataset='unsw', split='random'), **res})
        print(f'appended to {out}')