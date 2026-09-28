'''
Argus on the dirty / scrubbed temporal LANL splits, for the poisoning comparison.

argus_test_lanl.py builds every split's snapshots from the TRAINING timestamps, which only works
for the random split where all splits share the same hours. Here train and test hours are disjoint.

  --scrub off (default): train on days [0, cutoff), attack edges left in and unlabeled  -> dirtyts
  --scrub on:            same window with those attack edges removed                    -> cleants

Both score the identical test set (everything at/after the cutoff), so the difference between the
two runs is attributable to training on unlabeled attack traffic.

Evaluation protocol: gradients come only from the training window, but inference is one
continuous causal pass over the entire timeline (see rolling_pass). The edges at time t are
predicted from node embeddings built on the graph at t-1, wherever t-1 falls -- the split
boundary governs what the model LEARNED from, not what it may look at when scoring.

ArgusTS below overrides Argus.forward to fix three things; see its docstring. The same fixes
belong in argus_test_lanl.py, which every other Argus number comes from.

Run from prior_works/ (needs data/lanl14argus_tgraph_raw_csr.pt for Argus's edge features):

    python argus_test_lanl_ts.py --device 0 --runs 5
    python argus_test_lanl_ts.py --device 1 --runs 5 --scrub
'''
from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import sys
import time

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops
from tqdm import tqdm

from argus_test_lanl import Argus  # same layers, APLoss and hyperparameters

sys.path.insert(0, '..')
from metrics import prevalence, threshold_metrics  # noqa: E402

try:  # whatever argus_test_lanl.py already imports
    from argus_test_lanl import SOAP
except ImportError:
    from libauc.optimizers import SOAP

HOUR = 60 * 60


class ArgusTS(Argus):
    '''
    Argus with the same layers and hyperparameters, and three corrections to forward().

    1. The recurrent scan ran over nodes, not time. Argus.forward stacks the per-snapshot
       embeddings with torch.stack(zs, dim=1) -> (N, T, h), and self.rnn wraps nn.GRU with
       batch_first left at False, so the GRU reads dim 0 as the sequence: it scans the NODE
       axis, whose ordering is arbitrary, and treats each snapshot as an independent batch
       element. Nothing crossed time. Stacking dim=0 -> (T, N, h) makes the scan temporal with
       each node as a batch element; the aggregation loop below then indexes time on dim 0.

    2. The aggregation used the wrong snapshot's neighbours. The loop read idxs[i]/ptrs[i],
       where `i` is left over from the embedding loop above and is always len(eis)-1, so every
       timestep aggregated over the LAST snapshot's neighbour lists. It now uses idxs[t]/ptrs[t].

    3. The hidden state is exposed (h0 in, h out) so a window can be scored in chunks without
       restarting the GRU at every chunk boundary.

    sample_z is also vectorized (see below) -- same distribution, far less memory.
    '''

    def sample_z(self, z, idx, ptr):
        '''
        Vectorized equivalent of Argus.sample_z.

        The original loops over all N nodes in Python and appends to a list, so a single
        timestep builds ~N little autograd subgraphs; a chunk of T snapshots builds T*N of
        them. That, not the convolutions, is what makes memory climb snapshot by snapshot.

        ones(deg).multinomial(s, replacement=True) draws s neighbours uniformly with
        replacement, which is exactly randint(0, deg, (s,)) -- so this samples from the same
        distribution, using O(1) graph nodes per timestep instead of O(N).
        '''
        idx, ptr = idx.to(self.device), ptr.to(self.device)
        if ptr.numel() == 0:                       # snapshot with no edges
            return self.decode_mlp(z)

        deg = (idx[1:] - idx[:-1])                 # (N,) neighbours per node
        has = (deg > 0).unsqueeze(1)
        off = torch.randint(0, 1 << 30, (z.size(0), self.s), device=self.device)
        off = off % deg.clamp(min=1).unsqueeze(1)
        nbr = ptr[(idx[:-1].unsqueeze(1) + off).clamp(max=ptr.numel() - 1)]

        agg = (z[nbr].sum(dim=1) + z) / (self.s + 1)
        return self.decode_mlp(torch.where(has, agg, z))

    def forward(self, x, eis, eas, idxs, ptrs, h0=None, include_h=False):
        x = x.to(self.device)
        eis = [ei.to(self.device) for ei in eis]
        eas = [ea.to(self.device) for ea in eas]

        zs = []
        for ei, ea in zip(eis, eas):
            ei_self_loops = add_remaining_self_loops(ei)[0]
            z = self.c1(x, ei_self_loops)
            z = self.c2(z, ei_self_loops)
            z = self.relu(z)
            z = self.drop(z)
            z = self.c3(z, ei_self_loops)
            z = self.relu(z)
            z = self.drop(z)
            z = self.c4(z, ei, edge_attr=ea)
            z = self.ac(z)
            zs.append(z)

        out, h = self.rnn(torch.stack(zs, dim=0), h0, include_h=True)   # (T, N, z)
        out = torch.stack([self.sample_z(out[t], idxs[t], ptrs[t]) for t in range(out.size(0))])
        return (out, h) if include_h else out


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def load_windows(csr, days, scrub):
    '''Train / val / test edge masks matching temporal_split.py, plus per-edge labels.'''
    g = torch.load(csr, weights_only=False)
    non_te = HOUR * 24 * days
    tr_ts = non_te * 0.9

    is_mal = torch.zeros(g.col.size(0), dtype=torch.bool)
    is_mal[g.is_mal] = True

    tr = g.ts < tr_ts
    va = (g.ts >= tr_ts) & (g.ts < non_te)
    te = g.ts >= non_te
    n_in_window = int((is_mal & (tr | va)).sum())
    if scrub:
        tr &= ~is_mal
        va &= ~is_mal

    print(f'cutoff {non_te}s ({days}d): train {int(tr.sum()):,}, val {int(va.sum()):,}, '
          f'test {int(te.sum()):,} edges ({int((is_mal & te).sum())} malicious)')
    print(f'{n_in_window} malicious edges fall in the train/val window '
          f'({"removed (scrubbed)" if scrub else "kept, unlabeled (dirty)"})')
    return g, tr, va, te, is_mal


def snapshots(g, mask, is_mal):
    '''
    One snapshot per hour present in `mask`, with Argus's raw edge features and its own
    neighbour lists. Every split gets neighbour lists: sample_z aggregates over the neighbours
    of the snapshot being scored, so a test snapshot cannot borrow the training graph's.
    '''
    src = torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])
    ei = torch.stack([src, g.col])
    hours = (g.ts // HOUR)

    eis, eas, ys, idxs, ptrs = [], [], [], [], []
    for h in hours[mask].unique(sorted=True).tolist():
        m = mask & (hours == h)
        eis.append(ei[:, m])
        eas.append(g.raw_edge_attr[m].float())
        ys.append(is_mal[m])

        csr = defaultdict(list)
        s, d = eis[-1]
        for a, b in zip(s.tolist(), d.tolist()):
            csr[a].append(b)
        idx, ptr = [0], []
        for i in range(g.x.size(0)):
            ptr += csr[i]
            idx.append(idx[-1] + len(csr[i]))
        idxs.append(torch.tensor(idx))
        ptrs.append(torch.tensor(ptr))

    return Data(x=torch.eye(g.x.size(0)), edge_index=eis, eas=eas, label=ys, idxs=idxs, ptrs=ptrs)


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def rolling_pass(model, tr, va, te, args, collect=('va', 'te')):
    '''
    One continuous causal pass over the whole timeline: train hours, then val, then test.

    Every snapshot t is scored with the embedding built from the graph at t-1, wherever t falls
    -- the split boundaries affect only what the model LEARNED from, never what it sees at
    inference. The state is never reset and never truncated, so the embedding used for the first
    test hour is the one that follows from having watched every preceding hour, including the
    training window's (which is where the dirty/clean difference lives).

    Snapshot t is scored first and consumed afterwards, so a scored edge is never part of the
    graph that represents it. Causality relies on the GRU scanning time, i.e. on ArgusTS's fix.

    Returns {split: {'pred': real-edge scores, 'label': ..., 'neg': random-pair scores,
                     'snapshots': how many were scored}}.
    '''
    model.eval()
    out = {k: {'pred': [], 'label': [], 'neg': [], 'snapshots': 0} for k in collect}
    h, z_prev = None, None

    for name, d in (('tr', tr), ('va', va), ('te', te)):
        n = len(d.edge_index)
        for st in range(0, n, args.chunk):
            en = min(st + args.chunk, n)
            zs, h = model(tr.x, d.edge_index[st:en], d.eas[st:en],
                          d.idxs[st:en], d.ptrs[st:en], h0=h, include_h=True)
            h = h.detach()

            for j in range(st, en):
                ei = d.edge_index[j]
                if name in collect and z_prev is not None:
                    out[name]['snapshots'] += 1
                    if ei.size(1):
                        # NOT negated. calc_loss_argus labels RANDOM edges as the positive class
                        # and real edges as the negative one, so Argus is trained to score
                        # unlikely edges HIGH: decode is already an anomaly score.
                        ns = torch.randint(0, z_prev.size(0), ei.size(), device=model.device)
                        out[name]['pred'].append(model.decode(ei.to(model.device), z_prev).cpu())
                        out[name]['neg'].append(model.decode(ns, z_prev).cpu())
                        out[name]['label'].append(d.label[j])
                # consumed only after snapshot j has been scored; clone so the chunk can be freed
                z_prev = zs[j - st].clone()

    return out


def lp_metrics(part):
    '''
    Model-selection signal: real edges against random pairs, as Argus.validate does. It uses no
    malicious labels, which matters here -- the scrubbed validation window has no malicious edges
    at all, so an anomaly-labelled val metric is undefined for that run and selecting on it would
    mean selecting on the test set.
    '''
    pos, neg = torch.cat(part['pred']), torch.cat(part['neg'])
    scores = torch.cat([pos, neg]).numpy()
    y = torch.cat([torch.zeros(pos.numel()), torch.ones(neg.numel())]).numpy()  # random = positive
    return roc_auc_score(y, scores), average_precision_score(y, scores)


def detection_metrics(part):
    '''AUC/AP plus the split-comparable metrics (lift, precision and FP/day at 50% recall).'''
    y = torch.cat(part['label']).long().numpy()
    p = torch.cat(part['pred']).float().numpy()
    if y.sum() == 0:
        return dict(auc=float('nan'), ap=float('nan'), ap_lift=float('nan'),
                    prec_r50=float('nan'), fp_day_r50=float('nan'))

    ap = average_precision_score(y, p)
    prec, fp, _ = threshold_metrics(y, p, recall=0.5)
    days = part['snapshots'] / 24  # one snapshot per hour
    return dict(auc=roc_auc_score(y, p), ap=ap, ap_lift=ap / prevalence(y),
                prec_r50=prec, fp_day_r50=fp / days if days else float('nan'))


def probe_memory(tr, args):
    '''
    Train-step peak memory at a range of chunk sizes, so --chunk can be chosen by measurement.

    Memory is linear in chunk size: a chunk holds the autograd graph for every snapshot in it,
    so nothing is released until backward. Watching nvtop over the first snapshot or two
    overstates the slope, because the first forward also allocates cuBLAS/cuDNN workspaces and
    grows the caching allocator -- costs paid once, not per snapshot. This separates the two.
    '''
    model = ArgusTS(tr.x.size(0), tr.eas[0].size(1), 128, 64, args.device,
                    pos_samples=sum(e.size(1) for e in tr.edge_index))
    opt = SOAP(model.parameters(), lr=0.01, mode='adam', weight_decay=0.0)
    gib = 1 << 30
    total = torch.cuda.mem_get_info(args.device)[1] / gib if torch.cuda.is_available() else 0
    print(f'device total {total:.1f} GiB, {tr.x.size(0):,} nodes, {len(tr.edge_index)} snapshots\n')
    print(f'{"chunk":>6}  {"peak":>9}  {"% of card":>10}  {"marginal/snapshot":>18}')

    prev = None
    for c in (1, 2, 4, 8, 12, 16, 24, 32):
        if c > len(tr.edge_index):
            break
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        try:
            opt.zero_grad()
            zs, _ = model(tr.x, tr.edge_index[:c], tr.eas[:c], tr.idxs[:c], tr.ptrs[:c],
                          h0=None, include_h=True)
            model.calc_loss_argus(zs, [ei.to(args.device) for ei in tr.edge_index[:c]]).backward()
            peak = torch.cuda.max_memory_allocated(args.device) / gib
            marg = f'{(peak - prev[1]) / (c - prev[0]):.3f} GiB' if prev else '-'
            print(f'{c:>6}  {peak:>6.2f} GiB  {100 * peak / total:>9.0f}%  {marg:>18}')
            prev = (c, peak)
        except torch.cuda.OutOfMemoryError:
            print(f'{c:>6}  {"OOM":>9}')
            break
    print('\nThe largest chunk that stays comfortably under 100% is the one to use; memory is '
          'linear in chunk, so extrapolate the marginal column for sizes not listed.')


def train(tr, va, te, args):
    model = ArgusTS(tr.x.size(0), tr.eas[0].size(1), 128, 64, args.device,
                    pos_samples=sum(e.size(1) for e in tr.edge_index))
    opt = SOAP(model.parameters(), lr=0.01, mode='adam', weight_decay=0.0)

    best, best_test, no_progress = None, None, 0
    for e in range(args.epochs):
        model.train()
        st = time.time()
        h = None  # each epoch restarts from a zero state; within it, chunks carry state forward
        for i in tqdm(range(0, len(tr.edge_index), args.chunk)):
            eis = tr.edge_index[i:i + args.chunk]
            eas = tr.eas[i:i + args.chunk]
            if not eis:
                continue
            opt.zero_grad()
            zs, h = model(tr.x, eis, eas, tr.idxs[i:i + args.chunk], tr.ptrs[i:i + args.chunk],
                          h0=h, include_h=True)
            loss = model.calc_loss_argus(zs, [ei.to(args.device) for ei in eis])
            loss.backward()
            opt.step()
            # truncated BPTT: the state crosses the chunk boundary, the gradient does not
            h = h.detach()

        scored = rolling_pass(model, tr, va, te, args)
        va_auc, va_ap = lp_metrics(scored['va'])
        te_m = detection_metrics(scored['te'])
        key = va_auc + va_ap  # the criterion argus_test_lanl.py selects on

        marker = ''
        if best is None or key > best[0]:
            best, no_progress, marker = (key, te_m, e), 0, ' *'
        else:
            no_progress += 1
        if best_test is None or te_m['ap'] > best_test[0]:
            best_test = (te_m['ap'], te_m, e)

        print(f'[epoch {e}] {time.time() - st:.0f}s | VAL AUC {va_auc:.4f} AP {va_ap:.4f} | '
              f'TEST AUC {te_m["auc"]:.4f} AP {te_m["ap"]:.4f} lift {te_m["ap_lift"]:.1f}x '
              f'FP/day {te_m["fp_day_r50"]:.0f}{marker}')

        if no_progress > args.patience:
            break

    out = dict(best[1])
    out['epoch'] = best[2]
    # Reported separately and never used for selection: the best test AP any epoch reached.
    # argus_test_lanl.py tracks the same thing and calls it snooping, because it is.
    out['ap_snooped'] = best_test[0]
    out['epoch_snooped'] = best_test[2]
    return out


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--csr', default='../data/lanl14argus_tgraph_raw_csr.pt')
    ap.add_argument('--days', type=float, default=7, help='train/val window; must match the CyberGFM split')
    ap.add_argument('--scrub', action='store_true', help='remove in-window malicious edges (cleants control)')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--chunk', type=int, default=24,
                    help='snapshots per forward pass; 0 = the whole window in one pass, which is '
                         'what argus_test_lanl.py does (BS = len(tr.edge_index), so one optimizer '
                         'step per epoch over the full sequence)')
    ap.add_argument('--probe', action='store_true',
                    help='report train-step peak memory at several chunk sizes, then exit')
    ap.add_argument('--cache', default='tmp')
    args = ap.parse_args()
    args.device = args.device if args.device >= 0 else 'cpu'
    tag = 'cleants' if args.scrub else 'dirtyts'
    tag += f'-{args.days}' if args.days != 7 else ''

    os.makedirs(args.cache, exist_ok=True)
    # v2: every split now carries its own neighbour lists, so older caches are not compatible
    cache = f'{args.cache}/argus_lanl14argus-{tag}_{args.days:g}d_v2.pt'
    if os.path.exists(cache):
        tr, va, te = torch.load(cache, weights_only=False)
        print('loaded snapshots from', cache)
    else:
        g, tr_m, va_m, te_m, is_mal = load_windows(args.csr, args.days, args.scrub)
        tr, va, te = snapshots(g, tr_m, is_mal), snapshots(g, va_m, is_mal), snapshots(g, te_m, is_mal)
        torch.save((tr, va, te), cache)
        print(f'snapshots: train {len(tr.edge_index)}, val {len(va.edge_index)}, test {len(te.edge_index)} hours')

    if args.chunk <= 0:  # one pass over everything, as in argus_test_lanl.py
        args.chunk = max(len(tr.edge_index), len(va.edge_index), len(te.edge_index))
        print(f'--chunk 0: using {args.chunk} (whole window, one optimizer step per epoch)')

    torch.set_num_threads(64)
    if args.probe:
        probe_memory(tr, args)
        raise SystemExit

    results = []
    for r in range(args.runs):
        print(f'\n===== run {r} ({tag}) =====')
        results.append(train(tr, va, te, args))
        with open(f'argus_lanl14argus-{tag}_log.csv', 'a') as f:
            f.write(json.dumps(results[-1]) + '\n')

    df = pd.DataFrame(results)
    df.loc['mean'] = df.mean()
    df.loc['sem'] = df.sem()
    print(df)
    df.to_csv(f'argus_results_lanl14argus-{tag}.csv')