'''
Argus on OpTC. Defaults to `optc-argus` -- Argus's own split and labelling -- so the number is
directly comparable to CyberGFM's on the same files.

WHAT CHANGED FROM THE PREVIOUS VERSION OF THIS FILE, and why each mattered:

1. THE TEST SET WAS BEING THROWN AWAY. The timestamp axis came from the training split
   (`ts = tr.ts.unique()`) and `to_snapshots(te, ts)` keeps only test edges whose ts is in that
   list. Under a temporal split the training window ends at the attack boundary, so nearly every
   test edge was dropped and AP/AUC were computed on roughly the single boundary bucket. Each
   split now gets snapshots over ITS OWN timestamps, and the test window is embedded by a second
   forward pass that continues the GRU state across the boundary (see `--score-mode`). This is
   the single reason the old OpTC numbers should not be trusted.

2. Dimensions were 128/64. Argus uses hidden=32, z=16 (`main.py` worker_args=[32,32], the
   reduction to 16 happens in the GRU's output projection). 128/64 is a different model.

3. The neighbour aggregation (`sample_z`) ran on every snapshot. Argus's released code defines
   `inner_forward_sm` and never calls it; `forward_once` returns `self.ac(x)`. Off by default
   now, `--aggregate` restores it.

4. `range(len(tr.edge_index) // BS)` truncates: with 1592 snapshots and BS=256 it trained on 1536
   and silently ignored 56. Now `range(0, n, BS)`.

5. The cache check looked for `tmp/optc_tr.pt` while the writes went to `tmp/optc-ts_tr.pt`, so
   the snapshots were rebuilt on every run.

6. THE LOSS POLARITY, which an earlier version of this file left inverted on the theory that
   labelling observed edges 0 and skipping the negation at test time was an equivalent
   reformulation of Argus's labelling-plus-`1 - scores`. IT IS NOT. APLoss is an
   average-precision surrogate and AP is asymmetric in the two classes, so mirroring the labels
   does not mirror the objective -- on LANL, running it inverted cost about 50x in AP (~0.002 vs
   ~0.11). The model now lives in argus.py with the observed edges as the positive class, and
   this file negates at eval. Symptoms of the old version, for the record: test AUC below 0.5,
   a "correct" sign that differed from run to run, and test AUC swinging across [0, 1] between
   epochs while validation sat at 0.98.

TWO SPLITS. `--dataset optc-argus` (default) is Argus's own temporal split; `--dataset optc` is
the original random edge split. They need different scoring, and `--split` picks it -- auto
resolves `optc` to random and everything else to temporal, then checks that against the data's
timestamps before building anything:

  temporal  test edges come after training. Their snapshots get their own time axis and the
            GRU state is carried across the boundary into them (item 1 above).
  random    test edges are interleaved with training. Their snapshots go on the training axis
            and test snapshot i is scored against the training embedding at i -- the original
            protocol, which was correct for this split all along.

OUTPUT. One row per training run, appended to <--out-dir>/<dataset><--tag>.csv, which every seed
of the experiment shares (see argus.append_result). run_argus.sh submits one job per seed.

    python argus_test_optc.py --device 0 --seed 1                    # optc-argus, one run
    python argus_test_optc.py --device 0 --seed 1 --dataset optc     # the random split
    python show_argus_results.py ../results/argus                    # mean and SE per experiment
'''
from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import random
import time

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score

# One shared model definition. argus.py's docstring carries the six fixes and the SCORE
# CONVENTION: decode() is a reconstruction probability where HIGH MEANS NORMAL, so the anomaly
# score is its negation. This file used to keep its own copy, which had drifted badly -- it was
# still training with inverted polarity and an unbounded decoder.
from argus import (Argus, SOAP, chunked_forward, append_result, build_once, run_record,
                   snap_ews, std_edge_w, use_edge_weights)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def to_snapshots(g, ts, add_csr=False):
    '''
    One snapshot per timestamp in `ts`. Pass each split ITS OWN timestamps: using the training
    split's axis for the test split silently drops every test edge past the training window.
    '''
    if 'label' in g.keys():
        all_edges = torch.stack([g.src, g.col, g.ts, g.label])
    else:
        all_edges = torch.stack([g.src, g.col, g.ts])

    all_edges, cnt = all_edges.unique(dim=1, return_counts=True)
    ei, ts_g = all_edges[:2], all_edges[2]
    labels = all_edges[3] if 'label' in g.keys() else None

    raw_edge_attr = torch.ones((ei.size(1), 1))   # OpTC has no edge features

    eis, eas, ews, idxs, ptrs, cnts, y = [], [], [], [], [], [], []
    for t in ts:
        mask = ts_g == t
        eis.append(ei[:, mask])
        eas.append(raw_edge_attr[mask])
        cnts.append(cnt[mask])
        # The snapshot is already deduplicated, with the multiplicity in cnt. Their edge weight is
        # std_edge_w of that count, computed per snapshot.
        ews.append(std_edge_w(cnt[mask]))

        if add_csr:
            csr_dict = defaultdict(list)
            ei_t = ei[:, mask]
            for i in range(ei_t.size(1)):
                src, dst = ei_t[:, i]
                csr_dict[src.item()].append(dst.item())
            idx, ptr = [0], []
            for i in range(g.x.size(0)):
                ptr += csr_dict[i]
                idx.append(idx[-1] + len(csr_dict[i]))
            idxs.append(torch.tensor(idx))
            ptrs.append(torch.tensor(ptr))

        if labels is not None:
            y.append(labels[mask])

    return Data(x=torch.eye(g.x.size(0)), edge_index=eis, label=y, eas=eas,
                idxs=idxs, ptrs=ptrs, cnts=cnts, ews=ews)


def check_split(splits, ts_tr, mode):
    '''
    Confirm the data has the shape `mode` assumes, and report how much of the test set survives.

    The code that built optc_tgraph_{tr,va,te} is not in the repository, so the split type is
    checked against the timestamps rather than trusted from the dataset name. Getting it wrong
    is not a small error: scoring a temporal split on the training axis drops nearly every test
    edge (the bug this file originally had), and continuing the recurrence into a random split's
    test snapshots embeds them as if they came AFTER a window they are actually inside.
    '''
    tr, te = splits['tr'], splits['te']
    tr_lo, tr_hi = int(tr.ts.min()), int(tr.ts.max())
    te_lo, te_hi = int(te.ts.min()), int(te.ts.max())
    looks = 'temporal' if te_lo >= tr_hi else 'random'
    on_axis = torch.isin(te.ts, ts_tr)
    mal = te.label.clamp(0, 1).bool() if 'label' in te.keys() else torch.zeros_like(on_axis)
    print(f'split check: train ts [{tr_lo}, {tr_hi}], test ts [{te_lo}, {te_hi}] -> looks {looks}; '
          f'running as {mode}')
    if looks != mode:
        raise SystemExit(
            f'--split {mode} does not match the data, which looks {looks} (test '
            f'{"starts after" if looks == "temporal" else "overlaps"} training). Pass '
            f'--split {looks}, or --split {mode} explicitly only if you know why.')
    if mode == 'random':
        print(f'  test edges on the training axis: {on_axis.float().mean():.2%} of all, '
              f'{on_axis[mal].float().mean() if mal.any() else float("nan"):.2%} of malicious '
              f'(the rest fall in buckets with no training edges and cannot be scored)')


def load_splits(args):
    '''
    The three snapshot files, built once however many seed jobs start at the same moment (see
    argus.build_once). The split mode is in the name: the test snapshots sit on a different time
    axis in each mode, so a cache built for one must never be loaded by the other.
    '''
    # _ew: snapshots carry edge weights; _sym: optc-argus test edges are undirected like training
    # (load_argus_data.py). Older caches lack one or the other.
    paths = [f'{args.cache}/{args.dataset}-{args.split}_ew_sym_{n}.pt' for n in ('tr', 'va', 'te')]

    def build():
        splits = {}
        for name in ('tr', 'va', 'te'):
            g = torch.load(f'../data/{args.dataset}_tgraph_{name}.pt', weights_only=False)
            g.ts //= args.ts_div
            splits[name] = g
        ts_tr = splits['tr'].ts.unique()
        check_split(splits, ts_tr, args.split)
        # random:   test edges are interleaved with training edges, so they go on the TRAINING
        #           axis and test snapshot i is scored against zs_tr[i].
        # temporal: test edges all come after training, so they get their OWN axis and the
        #           recurrence is continued into them (item 1 in the header).
        te_axis = ts_tr if args.split == 'random' else splits['te'].ts.unique()
        return [to_snapshots(splits['tr'], ts_tr, add_csr=args.aggregate),
                to_snapshots(splits['va'], ts_tr),         # holdout of the SAME window
                to_snapshots(splits['te'], te_axis)]

    return tuple(build_once(paths, build))

def train(tr, va, te, args):
    pos_samples = sum(tr.edge_index[i].size(1) for i in range(len(tr.edge_index)))
    model = Argus(tr.x.size(0), tr.eas[0].size(1), args.h_dim, args.z_dim, args.device,
                  s=args.s, pos_samples=pos_samples, aggregate=args.aggregate,
                  margin=args.margin, gamma=args.gamma,
                  gru_layers=args.gru_layers, beta=args.beta,
                  neg_sampling=args.neg_sampling, neg_filter=not args.no_neg_filter)
    opt = SOAP(model.parameters(), lr=args.lr, mode='adam', weight_decay=0.0)

    best = (0, 0, 0)
    best_snooped = (0, 0)
    best_epoch = -1
    n_test = n_mal = 0
    no_progress = 0
    n_snaps = len(tr.edge_index)
    print(f'{n_snaps} training snapshots, {len(te.edge_index)} test snapshots')

    for e in range(args.epochs):
        fwd_time = bwd_time = loss_time = step_time = 0
        # range(0, n, BS): the old `range(n // BS)` dropped the tail
        for st_i in range(0, n_snaps, args.bs):
            en_i = min(st_i + args.bs, n_snaps)
            model.train()
            opt.zero_grad()

            st = time.time()
            zs = model.forward(tr.x, tr.edge_index[st_i:en_i], tr.eas[st_i:en_i],
                               tr.idxs[st_i:en_i] if tr.idxs else [],
                               tr.ptrs[st_i:en_i] if tr.ptrs else [],
                               ews=snap_ews(tr, st_i, en_i))
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
                          dict(experiment=f'{args.dataset}{args.tag}', fwd=fwd_time,
                               loss=loss_time, bwd=bwd_time, step=step_time))
            return None

        with torch.no_grad():
            model.eval()
            # training window, keeping the GRU state at the boundary
            zs_tr, h = chunked_forward(model, tr.x, tr, args.chunk)

            # Validation is a LINK-PREDICTION metric: observed edges vs. random pairs, scored
            # with the RAW reconstruction probability and observed edges as the positive class.
            # It is the one metric that is not negated. (This block used to label observed edges
            # 0, matching the inverted loss that argus.py now fixes.)
            pos, neg = model.validate(zs_tr, va.edge_index)
            labels = torch.cat([torch.ones(pos.numel()), torch.zeros(neg.numel())]).numpy()
            preds = torch.cat([pos, neg]).numpy()
            va_auc = auc_score(labels, preds)
            va_ap = ap_score(labels, preds)
            print(f'\tVal AUC: {va_auc:0.4f}, AP: {va_ap:0.4f}')

            # TEST WINDOW: embed the test snapshots themselves, continuing the recurrence.
            # contemporaneous = Argus's own protocol (an edge is in the graph that embeds it);
            # predictive scores snapshot t against the embedding from t-1.
            if args.split == 'random':
                # Test snapshots share the training axis, so the training-window embeddings ARE
                # the contemporaneous ones. They are built from training edges only, so no test
                # edge shapes the embedding that scores it.
                zs_te = zs_tr
            else:
                zs_te, _ = chunked_forward(model, te.x, te, args.chunk, h0=h)

            preds, y, cnt = [], [], []
            for i in range(len(te.edge_index)):
                if te.edge_index[i].size(1) == 0:
                    continue
                j = i if args.score_mode == 'contemporaneous' else i - 1
                if j < 0:
                    continue
                preds.append(model.decode(te.edge_index[i].to(args.device), zs_te[j]))
                y.append(te.label[i])
                cnt.append(te.cnts[i])

            # decode() already applies the sigmoid, and HIGH MEANS NORMAL -- so the anomaly
            # score is the negated probability ("alarms for the ones under tau"). The old code
            # applied a second sigmoid and did not negate, which is why its AUC came out inverted.
            preds = (-torch.cat(preds)).cpu().numpy()
            y = torch.cat(y).clamp(0, 1).cpu().numpy()
            cnt = torch.cat(cnt).cpu().numpy()
            n_test, n_mal = len(y), int(y.sum())
            print(f'\tscored {n_test:,} test edges, {n_mal:,} malicious')

            auc = auc_score(y, preds, sample_weight=cnt)
            ap = ap_score(y, preds, sample_weight=cnt)
            print(f'\tTe AUC: {auc:0.4f}, AP: {ap:0.4f}', end='', flush=True)

            if va_auc + va_ap > best[0]:
                best = (va_auc + va_ap, auc, ap)
                best_epoch = e
                print('*')
                no_progress = 0
            else:
                print()
                no_progress += 1

            # Tracked without using validation, because validation does not track test here.
            # Data snooping: an upper bound, reported separately, never as the headline.
            if ap > best_snooped[1]:
                best_snooped = (auc, ap)

            if no_progress > args.patience:
                break

    out = {'auc': best[1], 'ap': best[2], 'auc_last': auc, 'ap_last': ap,
           'auc_snooped': best_snooped[0], 'ap_snooped': best_snooped[1],
           'epochs_ran': e + 1, 'best_epoch': best_epoch,
           'n_test_edges': n_test, 'n_malicious': n_mal}
    print(f"Best: AUC {best[1]:0.4f}, AP {best[2]:0.4f}")
    return out


if __name__ == '__main__':
    ap = ArgumentParser()
    # runtime
    ap.add_argument('--device', type=int, default=0, help='CUDA index, or -1 for CPU')
    ap.add_argument('--runs', type=int, default=1, help='independent training runs to average')
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
    ap.add_argument('--beta', type=float, default=0.5,
                    help='weight on L_DEC in loss = L_ap + beta * L_DEC '
                         '(paper: 0.5 for OpTC, "based on their data distributions"). Their released code has NO L_DEC term at all, so any '
                         'nonzero value runs this repo\'s per-edge BCE reading of eq. 12, which '
                         'is a GUESS; --beta 0 reproduces their code exactly')
    ap.add_argument('--gamma', type=float, default=0.1,
                    help='APLoss moving-average factor (paper: 0.1 for OpTC)')
    ap.add_argument('--no-edge-weight', action='store_true',
                    help='do not pass edge weights to the GCN layers. With them (the default) each '
                         'snapshot is deduplicated and a (src, dst) pair is weighted by std_edge_w '
                         'of its event count, as their Argus_LANL does; this is the ablation, and '
                         'what every run before the edge-weight change did')
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

    ap.add_argument('--split', choices=['auto', 'temporal', 'random'], default='auto',
                    help="auto: random for `optc`, temporal for anything else. Checked against "
                         "the data's timestamps before anything is built; a mismatch stops.")
    ap.add_argument('--dataset', default='optc-argus',
                    help="optc-argus is Argus's own split and labelling (default); optc-ts is ours")
    ap.add_argument('--bs', type=int, default=256, help='snapshots per optimizer step')
    ap.add_argument('--chunk', type=int, default=256, help='snapshots per forward pass at eval')
    ap.add_argument('--score-mode', choices=['contemporaneous', 'predictive'],
                    default='contemporaneous',
                    help="contemporaneous is Argus's protocol: the scored edge is in the graph "
                         "that embeds it. predictive scores snapshot t against t-1.")
    ap.add_argument('--ts-div', type=int, default=360,
                    help='seconds per snapshot bucket. NOTE euler_test_optc.py uses 3600 for the '
                         'same data; pick one and use it for both or the baselines differ.')
    ap.add_argument('--speedtest', action='store_true')
    ap.add_argument('--cache', default='tmp')
    ap.add_argument('--tag', default='',
                    help='variant suffix for the results file, e.g. --tag _b0 alongside '
                         '--beta 0. Seeds of one experiment share a file; do not tag per seed')
    ap.add_argument('--prepare', action='store_true',
                    help='build the snapshot cache and exit')
    args = ap.parse_args()

    if args.split == 'auto':
        args.split = 'random' if args.dataset == 'optc' else 'temporal'
    if args.threads:
        torch.set_num_threads(args.threads)

    tr, va, te = load_splits(args)
    use_edge_weights(args, tr, va, te)
    if args.prepare:
        print(f'{len(tr.edge_index)} train / {len(va.edge_index)} val / {len(te.edge_index)} test '
              f'snapshots. Cache ready.')
        raise SystemExit

    out = f'{args.out_dir}/{args.dataset}{args.tag}.csv'
    for r in range(args.runs):
        seed = None if args.seed is None else args.seed + r
        if seed is not None:
            seed_everything(seed)
        print(f'=== run {r} (seed {seed}) ===', flush=True)
        t0 = time.time()
        res = train(tr, va, te, args)
        if not res:                     # --speedtest
            continue
        append_result(out, {**run_record(args, seed, (time.time() - t0) / 60,
                                         dataset=args.dataset, split=args.split,
                                         score_mode=args.score_mode, ts_div=args.ts_div),
                            **res})
        print(f'appended to {out}', flush=True)