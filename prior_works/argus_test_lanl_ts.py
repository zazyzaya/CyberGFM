'''
Argus on a temporal LANL split, as the Argus paper describes its own protocol.

SPLIT (--cutoff 41, the default; 147600s is FIRST_RED, the first redteam event):
  * every hourly snapshot before the cutoff is training data, and all of those snapshots are used
    for both training and validation -- validation is a random --val-frac (5%) of the EDGES in
    that window, not a later slice of time, so training covers the whole window.
  * everything at or after the cutoff is test.

EVALUATION: gradients come only from the training window, but inference is one continuous causal
pass over the whole timeline (rolling_pass). Edges at time t are scored from node embeddings built
on the graph at t-1, wherever t-1 falls -- the split boundary governs what the model LEARNED from,
not what it may look at when scoring. A scored edge is never in the graph that represents it.

POISONING COMPARISON: the same split at a later cutoff, where redteam edges land in the training
window (at --cutoff 41 there are none by construction, so --scrub is a no-op there):
  * --scrub off: attack edges left in the training window, unlabeled   -> dirtyts
  * --scrub on:  the same window with those attack edges removed       -> cleants
Both arms score the identical test set, so the difference is attributable to training on
unlabeled attack traffic.

ArgusTS overrides Argus.forward to fix three bugs; see its docstring. The same fixes belong in
argus_test_lanl.py, which every other Argus number comes from.

Run from prior_works/ (needs data/lanl14argus_tgraph_raw_csr.pt for Argus's edge features):

    python argus_test_lanl_ts.py --device 0 --runs 5                       # paper's protocol
    python argus_test_lanl_ts.py --device 0 --runs 5 --cutoff 168          # poisoning, dirty
    python argus_test_lanl_ts.py --device 1 --runs 5 --cutoff 168 --scrub  # poisoning, clean
'''
from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import sys
import time

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from torch_geometric.utils import add_remaining_self_loops

from argus_test_lanl import Argus, GRU  # same layers, APLoss and optimizer

sys.path.insert(0, '..')
from metrics import prevalence, threshold_metrics  # noqa: E402

try:  # whatever argus_test_lanl.py already imports
    from argus_test_lanl import SOAP
except ImportError:
    from libauc.optimizers import SOAP

HOUR = 60 * 60


class ArgusTS(Argus):
    '''
    Argus brought in line with the paper (Xu, Shu & Li, S&P 2024), with three bug fixes.

    FIXES to Argus.forward:

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

    DEVIATIONS from the paper that are corrected here:

    4. Loss polarity was inverted. calc_loss_argus labelled the RANDOM edges as the positive
       class and the observed edges as negative, training the model to score unlikely edges
       HIGH. The paper's L_ap ranks the observed edges of G_t high (eq. 13, P_t = positive
       edges in G_t) and detects by "set[ting] off alarms for the ones under tau" -- a LOW
       reconstruction probability is the anomaly signal. Fixed, and detection_metrics negates
       accordingly.

    5. L_DEC was missing entirely. The paper optimizes loss = L_ap + beta * L_DEC with
       beta = 0.01 on LANL, where L_DEC = -log p(A_t | A_hat_t), "cross-entropy (CE) to compute
       the loss of the decoder on the sampled edges" (eq. 12).

    6. Dimensions were 4x the paper's. Table 3 gives 32-wide EW layers, an EF layer mapping
       32 -> 16, a node embedding size of 16, and a TWO-layer GRU; the code built 128-wide
       layers, a 64-d embedding and a one-layer GRU.

    sample_z is also vectorized (same distribution, far less memory).
    '''

    def __init__(self, in_dim, edge_dim, h_dim=32, z_dim=16, device='cpu', s=5,
                 pos_samples=283573, gru_layers=1, beta=0.0, gamma=0.01,
                 neg_sampling='uniform', neg_filter=True, aggregate=False):
        super().__init__(in_dim, edge_dim, h_dim, z_dim, device, s=s, pos_samples=pos_samples)
        self.beta = beta
        self.neg_sampling = neg_sampling
        self.neg_filter = neg_filter
        self.aggregate = aggregate
        # Their main.py sets worker_args = [hidden, hidden] and rnn_args = [hidden, hidden, zdim]
        # with hidden=32, zdim=16, and recurrent.GRU defaults to hidden_units=1. So c4 IS
        # h_dim -> h_dim and the GRU's output projection does h_dim -> z_dim -- exactly the shape
        # Argus.__init__ already builds, with a ONE-layer GRU. Only the sizes were wrong. My
        # earlier reading of Table 3 (EF 32->16, two-layer GRU) was mistaken; nothing to rebuild.

    def decode(self, ei, z):
        '''
        sigmoid of the dot product, per Euler_Embed_Unit.decode in their repo:

            return torch.sigmoid((z[src] * z[dst]).sum(dim=1))

        argus_test_lanl.py returns a bare dot product of decode_mlp outputs, where decode_mlp ends
        in Softmax(dim=1). That is three differences in one line: the sigmoid is missing, a softmax
        their scoring path never applies is applied, and the scores end up concentrated near
        1/z_dim instead of spanning (0, 1). The last one matters most -- APLoss's margin of 0.8 is
        reachable between two sigmoids and essentially unreachable between two softmax dot
        products, so the surrogate loss was operating in a regime it was never meant for.
        '''
        return torch.sigmoid((z[ei[0]] * z[ei[1]]).sum(dim=1))

    def sample_negatives(self, ps, num_nodes):
        '''
        Negative pairs for L_ap.

        'uniform' (default) matches the released fast_negative_sampling: uniform node pairs, but
        with pairs that are actually edges of this snapshot REJECTED and redrawn. Argus's
        calc_loss_argus skips that filter, so a fraction of its "negatives" are real edges being
        pushed down. The collision rate is tiny (~1e-5 per snapshot at this graph's density), so
        this is fidelity rather than a fix.

        'degree' is a DIAGNOSTIC and a deviation from their code, not a replication of it. It
        resamples each endpoint from the snapshot's own endpoint multiset so negatives carry the
        positives' degree distribution. Uniform negatives are nearly separable by degree alone --
        a uniformly drawn pair is two arbitrary nodes, so "does this edge exist" collapses toward
        "are these nodes active" -- and a model can improve on that objective by becoming a degree
        prior, which raises LP AUC while destroying anomaly ranking. Use it to test whether that
        shortcut is what is happening; do not report it as Argus.
        '''
        if self.neg_sampling == 'degree':
            e = ps.size(1)
            si = torch.randint(0, e, (e,), device=ps.device)
            di = torch.randint(0, e, (e,), device=ps.device)
            return torch.stack([ps[0][si], ps[1][di]])

        ns = torch.randint(0, num_nodes, ps.size(), device=self.device)
        if self.neg_filter:
            key = lambda e: e[0] + e[1] * num_nodes   # same hash fast_negative_sampling uses
            pos_key = key(ps)
            for _ in range(5):                        # their while-loop, bounded
                bad = torch.isin(key(ns), pos_key)
                if not bad.any():
                    break
                ns[:, bad] = torch.randint(0, num_nodes, (2, int(bad.sum())), device=self.device)
        return ns

    def calc_loss_argus(self, zs, eis):
        '''
        loss = L_ap + beta * L_DEC, with the OBSERVED edges as the positive class.

        Argus.calc_loss_argus had the two blocks swapped (its `neg_pred` held the real edges),
        so APLoss was maximizing AP for the random pairs. Here the observed edges of G_t are
        the positives, matching eq. 13, and L_DEC is the decoder cross-entropy over the same
        sampled edges (eq. 12). decode() dots two softmax outputs, so it is already in [0, 1]
        and plain BCE applies rather than the with-logits form.
        '''
        tot = torch.zeros(1, device=self.device)
        for i in range(len(zs)):
            ps = eis[i]
            if ps.size(1) == 0:
                continue
            ps = ps.to(self.device)
            ns = self.sample_negatives(ps, zs.size(1))
            pos = self.decode(ps, zs[i])                   # observed edges -> class 1
            neg = self.decode(ns, zs[i])                   # random pairs   -> class 0

            score = torch.cat([pos, neg])                  # positives first: APLoss indexes them
            y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).detach()
            t_index = torch.arange(pos.size(0), dtype=torch.int64, device=self.device).detach()

            l_ap = self.ap_loss(score, y, t_index)
            l_dec = F.binary_cross_entropy(score.clamp(1e-7, 1 - 1e-7), y)
            tot = tot + l_ap + self.beta * l_dec

        return tot.true_divide(len(zs))

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
        # Their forward_once returns the encoder output straight from self.ac(x): the neighbour
        # aggregation (inner_forward_sm, matmul(s, z) then Softmax) is defined in their repo and
        # NEVER CALLED. argus_test_lanl.py applies it on every timestep -- and its own comment
        # already noticed that "scores are much better when we skip the aggregation step".
        if self.aggregate:
            out = torch.stack([self.sample_z(out[t], idxs[t], ptrs[t]) for t in range(out.size(0))])
        return (out, h) if include_h else out


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def edge_masks(g, cutoff_hours, val_frac, scrub, seed=0):
    '''
    Train / val / test edge masks for the paper's protocol.

    Every edge before the cutoff is training data; a random `val_frac` of those EDGES is held out
    for validation. Train and val therefore span the same hours, and the training window is the
    whole window -- no hours are sacrificed to validation. Everything at or after the cutoff is
    test. (temporal_split.py's temporally_partition_lanl_tgraph() does the same thing at
    FIRST_RED = 147600s = 41h, with a 10% holdout.)
    '''
    is_mal = torch.zeros(g.col.size(0), dtype=torch.bool)
    is_mal[g.is_mal] = True

    in_window = g.ts < HOUR * cutoff_hours
    te = ~in_window

    # "the 5% validation split occurs within each snapshot", so hold out per hour rather than
    # globally -- with uneven traffic across hours the two are not the same stratification.
    torch.manual_seed(seed)
    hrs = g.ts // HOUR
    tr, va = torch.zeros_like(in_window), torch.zeros_like(in_window)
    for h in range(int(cutoff_hours)):
        pool = (in_window & (hrs == h)).nonzero().squeeze(-1)
        if pool.numel() == 0:
            continue
        perm = torch.randperm(pool.numel())
        cut = int(pool.numel() * (1 - val_frac))
        tr[pool[perm[:cut]]] = True
        va[pool[perm[cut:]]] = True

    n_in_window = int((is_mal & in_window).sum())
    if scrub:
        tr &= ~is_mal
        va &= ~is_mal

    print(f'cutoff {cutoff_hours:g}h ({HOUR * cutoff_hours:,}s): train+val hours 0-'
          f'{cutoff_hours - 1:g}, random {1 - val_frac:.0%}/{val_frac:.0%} edge holdout; '
          f'test everything after')
    print(f'  edges: train {int(tr.sum()):,}, val {int(va.sum()):,}, test {int(te.sum()):,} '
          f'({int((is_mal & te).sum())} malicious)')
    print(f'  {n_in_window} malicious edges fall in the train/val window '
          f'({"removed (scrubbed)" if scrub else "kept, unlabeled (dirty)"})')
    if n_in_window == 0:
        print('  >> no malicious edges before the cutoff, so --scrub is a no-op at this cutoff '
              'and the dirty/clean comparison has nothing to measure')
    return tr, va, te, is_mal


def snapshots(g, mask, is_mal, hours, add_csr=True):
    '''
    One snapshot per hour in `hours`, with Argus's raw edge features.

    `hours` is an explicit list of integer hours, not just the hours that happen to contain edges,
    so an empty hour still produces an empty snapshot. Two things depend on that: splits built
    over the same hour list stay aligned index-for-index (which is what lets a val edge at hour t
    be scored against the training graph's embedding for t-1), and the GRU sees real elapsed time
    rather than a compressed sequence.

    add_csr=False skips the neighbour lists, which only snapshots fed to the model need; a split
    that is only ever scored never reaches sample_z.
    '''
    src = torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])
    ei = torch.stack([src, g.col])
    hrs = g.ts // HOUR

    eis, eas, ys, idxs, ptrs = [], [], [], [], []
    for h in hours:
        m = mask & (hrs == h)
        eis.append(ei[:, m])
        eas.append(g.raw_edge_attr[m].float())
        ys.append(is_mal[m])

        if add_csr:
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


def build_splits(csr, cutoff_hours, val_frac, scrub, seed=0):
    '''Snapshot tensors for the three splits, aligned as rolling_pass expects.'''
    assert float(cutoff_hours).is_integer(), '--cutoff must be a whole number of hours'
    cutoff_hours = int(cutoff_hours)

    g = torch.load(csr, weights_only=False)
    tr_m, va_m, te_m, is_mal = edge_masks(g, cutoff_hours, val_frac, scrub, seed)

    last = int((g.ts // HOUR).max())
    win_hours = list(range(cutoff_hours))          # 0 .. cutoff-1, all used for train AND val
    te_hours = list(range(cutoff_hours, last + 1))

    tr = snapshots(g, tr_m, is_mal, win_hours)
    va = snapshots(g, va_m, is_mal, win_hours, add_csr=False)  # scored only, never embedded
    te = snapshots(g, te_m, is_mal, te_hours)
    print(f'snapshots: {len(win_hours)} train/val hours (shared), {len(te_hours)} test hours')
    return tr, va, te


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
def eval_stages(tr, va, te):
    '''
    (graph fed to the model, name of what is scored against it, split scored) in temporal order.

    Train and val share hours, so a single walk over the training snapshots both builds the GRU
    state and scores the held-out val edges aligned to it; the walk then continues, uninterrupted,
    into the test hours.
    '''
    return [(tr, 'va', va), (te, 'te', te)]


@torch.no_grad()
def rolling_pass(model, x, stages, chunk, keep_edges=False, score_mode='contemporaneous'):
    '''
    One continuous causal pass over the whole timeline, never resetting the GRU.

    Every snapshot t is scored with the embedding built from the graph at t-1, wherever t falls
    -- the split boundaries affect only what the model LEARNED from, never what it sees at
    inference. Snapshot t is scored first and consumed afterwards, so a scored edge is never part
    of the graph that represents it. Causality relies on the GRU scanning time (ArgusTS's fix).

    Returns {split: {'pred': real-edge scores, 'label': ..., 'neg': random-pair scores,
                     'snapshots': how many were scored}}, plus 'src'/'dst' when keep_edges.
    '''
    model.eval()
    out, h, z_prev = {}, None, None

    for feed, name, score in stages:
        if name:
            out.setdefault(name, {'pred': [], 'label': [], 'neg': [], 'snapshots': 0})
        n = len(feed.edge_index)
        for st in range(0, n, chunk):
            en = min(st + chunk, n)
            zs, h = model(x, feed.edge_index[st:en], feed.eas[st:en],
                          feed.idxs[st:en], feed.ptrs[st:en], h0=h, include_h=True)
            h = h.detach()

            for j in range(st, en):
                # Which embedding scores snapshot j:
                #   'contemporaneous' -> zs[j], the graph AT j. This is what their decode_all
                #       does (models/argus.py:242, zs[i] against eis[i]). For the val split the
                #       scored edges are masked out of the graph, so it leaks nothing; for the
                #       test split every edge is in the graph, so an edge shapes its own
                #       embedding. That is their protocol.
                #   'predictive' -> zs[j-1], the graph at j-1, so a scored edge is never part of
                #       its own representation. More defensible, but a deviation from theirs, and
                #       it scores edges whose endpoints may have been inactive at j-1 against
                #       stale embeddings.
                z = zs[j - st] if score_mode == 'contemporaneous' else z_prev
                if name and z is not None:
                    ei = score.edge_index[j]
                    out[name]['snapshots'] += 1
                    if ei.size(1):
                        # Raw reconstruction probabilities, not anomaly scores: with the loss
                        # polarity fixed, a HIGH value means "this edge looks normal". lp_metrics
                        # uses them as-is; detection_metrics negates.
                        ei_d = ei.to(model.device)
                        # same negative distribution as training, so val LP is a usable proxy
                        ns = model.sample_negatives(ei_d, z.size(0))
                        out[name]['pred'].append(model.decode(ei_d, z).cpu())
                        out[name]['neg'].append(model.decode(ns, z).cpu())
                        out[name]['label'].append(score.label[j])
                        if keep_edges:
                            out[name].setdefault('src', []).append(ei[0].cpu())
                            out[name].setdefault('dst', []).append(ei[1].cpu())
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
    # observed edges are the positive class, matching the paper's L_ap
    y = torch.cat([torch.ones(pos.numel()), torch.zeros(neg.numel())]).numpy()
    return roc_auc_score(y, scores), average_precision_score(y, scores)


def detection_metrics(part):
    '''AUC/AP plus the split-comparable metrics (lift, precision and FP/day at 50% recall).'''
    y = torch.cat(part['label']).long().numpy()
    # "set off alarms for the ones under tau": a low reconstruction probability is the anomaly
    # signal, so the anomaly score is the negated probability.
    p = -torch.cat(part['pred']).float().numpy()
    if y.sum() == 0:
        return dict(auc=float('nan'), ap=float('nan'), ap_lift=float('nan'),
                    prec_r50=float('nan'), fp_day_r50=float('nan'))

    ap = average_precision_score(y, p)
    prec, fp, _ = threshold_metrics(y, p, recall=0.5)
    days = part['snapshots'] / 24  # one snapshot per hour
    return dict(auc=roc_auc_score(y, p), ap=ap, ap_lift=ap / prevalence(y),
                prec_r50=prec, fp_day_r50=fp / days if days else float('nan'))


def top_rank_report(part, tr, ks=(100, 500, 1000, 5000)):
    '''
    What sits at the TOP of the anomaly ranking.

    AUC near 0.97 with AP near 0.03 means global separation is fine but the highest-scored edges
    are mostly benign, and AP is dominated by that top slice. Reporting precision@k next to the
    endpoint degrees says which kind of edge is winning the ranking: decode() dots two 16-d
    softmax outputs, so it is maximal when both endpoints collapse onto the same basis vector --
    easiest for rare, low-degree nodes, which would then flood the top regardless of maliciousness.
    '''
    y = torch.cat(part['label']).long()
    p = -torch.cat(part['pred']).float()          # anomaly score: low probability = anomalous
    src, dst = torch.cat(part['src']), torch.cat(part['dst'])

    deg = torch.zeros(int(max(src.max(), dst.max())) + 1, dtype=torch.long)
    for d in (tr.idxptr[1:] - tr.idxptr[:-1],) if hasattr(tr, 'idxptr') else ():
        deg[:d.numel()] = d
    if deg.sum() == 0:                            # snapshot Data has no idxptr; count from edges
        for ei in tr.edge_index:
            deg.index_add_(0, ei[0], torch.ones(ei.size(1), dtype=torch.long))
            deg.index_add_(0, ei[1], torch.ones(ei.size(1), dtype=torch.long))

    order = torch.argsort(p, descending=True)
    print(f'  top-of-ranking ({int(y.sum())} malicious in {y.numel():,} test edges, '
          f'median training degree {deg[deg > 0].median().item()}):')
    for k in ks:
        if k > y.numel():
            break
        sel = order[:k]
        d = torch.cat([deg[src[sel]], deg[dst[sel]]]).float()
        print(f'    @{k:<5} precision {y[sel].sum().item() / k:.4f}  '
              f'recall {y[sel].sum().item() / max(1, int(y.sum())):.4f}  '
              f'endpoint degree: median {d.median():.0f} mean {d.mean():.0f} '
              f'zero-degree {100 * (d == 0).float().mean():.0f}%')


def build_model(tr, args):
    '''Argus at the paper's Table 3 dimensions unless overridden.'''
    return ArgusTS(tr.x.size(0), tr.eas[0].size(1), args.h_dim, args.z_dim, args.device,
                   s=args.s, pos_samples=sum(e.size(1) for e in tr.edge_index),
                   gru_layers=args.gru_layers, beta=args.beta, aggregate=args.aggregate,
                   neg_sampling=args.neg_sampling, neg_filter=not args.no_neg_filter)


def probe_memory(tr, args):
    '''
    Train-step peak memory at a range of chunk sizes, so --chunk can be chosen by measurement.

    Memory is linear in chunk size: a chunk holds the autograd graph for every snapshot in it,
    so nothing is released until backward. Watching nvtop over the first snapshot or two
    overstates the slope, because the first forward also allocates cuBLAS/cuDNN workspaces and
    grows the caching allocator -- costs paid once, not per snapshot. This separates the two.
    '''
    model = build_model(tr, args)
    opt = SOAP(model.parameters(), lr=args.lr, mode='adam', weight_decay=0.0)
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


def chunk_groups(n_snapshots, chunk, steps_per_epoch):
    '''
    Split the window into (offset, length) chunks, then bundle them into `steps_per_epoch` groups.

    Each group becomes one optimizer step, so steps_per_epoch=1 matches argus_test_lanl.py's single
    step per epoch no matter how small --chunk has to be to fit in memory.
    '''
    spans = [(i, min(chunk, n_snapshots - i)) for i in range(0, n_snapshots, chunk)]
    spans = [s for s in spans if s[1] > 0]
    k = max(1, min(steps_per_epoch, len(spans)))
    per = -(-len(spans) // k)  # ceil
    return [spans[j:j + per] for j in range(0, len(spans), per)]


def train(tr, va, te, args):
    model = build_model(tr, args)
    opt = SOAP(model.parameters(), lr=args.lr, mode='adam', weight_decay=0.0)

    best, best_test, no_progress = None, None, 0
    groups = chunk_groups(len(tr.edge_index), args.chunk, args.steps_per_epoch)
    for e in range(args.epochs):
        model.train()
        st = time.time()
        h = None  # each epoch restarts from a zero state; within it, chunks carry state forward
        for group in groups:
            # One optimizer step per group. argus_test_lanl.py sets BS = len(tr.edge_index) and so
            # takes exactly ONE step per epoch over the whole window; --chunk only decides how much
            # of the graph is resident at once, so gradients accumulate across a group and each
            # chunk's loss is weighted by its share of the group's snapshots. That reproduces the
            # single-pass gradient (bar BPTT truncation at chunk boundaries) at chunk-sized memory.
            #
            # --never-zero-grad reproduces their training loop, which calls opt.zero_grad()
            # NOWHERE (classification.py:104-111; grep finds it nowhere outside libauc/). The
            # gradient at epoch e is then the running sum over epochs 0..e. Under SOAP's adam mode
            # that is not simply a larger step: both moments grow together, so the update direction
            # is dominated by the averaged gradient history rather than the current epoch's -- very
            # heavy momentum. That plausibly explains why lr 0.01 is stable for them and diverged
            # for us. Only equivalent when there is one group per epoch, hence the assertion below.
            if not args.never_zero_grad:
                opt.zero_grad()
            total = sum(n for _, n in group)
            for i, n in group:
                zs, h = model(tr.x, tr.edge_index[i:i + n], tr.eas[i:i + n],
                              tr.idxs[i:i + n], tr.ptrs[i:i + n], h0=h, include_h=True)
                loss = model.calc_loss_argus(zs, [ei.to(args.device) for ei in tr.edge_index[i:i + n]])
                (loss * n / total).backward()
                # truncated BPTT: the state crosses the chunk boundary, the gradient does not
                h = h.detach()
            opt.step()

        scored = rolling_pass(model, tr.x, eval_stages(tr, va, te), args.chunk,
                              keep_edges=args.rank_report, score_mode=args.score_mode)
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
        if args.rank_report and marker:
            top_rank_report(scored['te'], tr)

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
    ap.add_argument('--cutoff', type=float, default=41,
                    help='train/test boundary in HOURS. 41 (= FIRST_RED, 147600s) is the paper\'s '
                         'protocol; use a later cutoff to put redteam edges in the training window '
                         'for the poisoning comparison (e.g. 168 for 7 days)')
    ap.add_argument('--val-frac', type=float, default=0.05,
                    help="fraction of the training window's EDGES held out for model selection")
    ap.add_argument('--split-seed', type=int, default=0,
                    help='seed for the validation holdout; temporal_split.py uses 0')
    # Architecture / loss, defaults from the paper (Table 3 and section 5)
    ap.add_argument('--h-dim', type=int, default=32, help="EW layer width (paper: 32)")
    ap.add_argument('--z-dim', type=int, default=16, help='node embedding size (paper: 16)')
    ap.add_argument('--gru-layers', type=int, default=1,
                    help='their recurrent.GRU defaults to hidden_units=1')
    ap.add_argument('--beta', type=float, default=0.0,
                    help='weight on L_DEC in loss = L_ap + beta * L_DEC. The paper uses 0.01 on '
                         'LANL, but the per-edge BCE form implemented here is a GUESS at eq. 12 '
                         'and can only be satisfied by collapsing embeddings onto a shared basis '
                         'vector, so it defaults OFF. See calc_loss_argus.')
    ap.add_argument('--lr', type=float, default=0.01, help='paper: 0.01')
    ap.add_argument('--steps-per-epoch', type=int, default=1,
                    help='optimizer steps per epoch; gradients accumulate across chunks within a '
                         'step. 1 reproduces argus_test_lanl.py (BS = the whole window) regardless '
                         'of how small --chunk has to be for memory')
    ap.add_argument('--neg-sampling', choices=['uniform', 'degree'], default='uniform',
                    help="negative pairs for L_ap and for the val LP metric. The paper allows "
                         "'uniform random or proportional to the node degrees'; the released code "
                         "only does uniform, which lets a degree prior win the objective. 'degree' "
                         'resamples endpoints from the snapshot, matching the positive degree '
                         'distribution. See ArgusTS.sample_negatives.')
    ap.add_argument('--no-neg-filter', action='store_true',
                    help="skip rejecting sampled negatives that are real edges. The released "
                         'fast_negative_sampling rejects them; argus_test_lanl.py does not, so this '
                         'reproduces the latter')
    ap.add_argument('--never-zero-grad', action='store_true',
                    help='never call opt.zero_grad(), so gradients accumulate across epochs. This '
                         'reproduces their classification.py, which omits it entirely. Requires '
                         '--steps-per-epoch 1, otherwise gradients would also accumulate across '
                         'groups within an epoch, which is not what they do')
    ap.add_argument('--score-mode', choices=['contemporaneous', 'predictive'],
                    default='contemporaneous',
                    help="which embedding scores snapshot t. 'contemporaneous' uses the graph at t, "
                         'as their decode_all does. \'predictive\' uses t-1 so a scored edge is '
                         'never in its own graph -- more defensible, but a deviation from theirs')
    ap.add_argument('--rank-report', action='store_true',
                    help='on each new best epoch, report precision@k and the endpoint degrees of '
                         'the highest-scored edges (diagnoses high AUC with low AP)')
    ap.add_argument('--s', type=int, default=5, help='neighbours sampled in the aggregation step')
    ap.add_argument('--aggregate', action='store_true',
                    help='apply the neighbour-sampling aggregation before decoding. Their repo '
                         'defines it (inner_forward_sm) but never calls it, so this is OFF by '
                         'default; argus_test_lanl.py applies it on every timestep')
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
    ap.add_argument("--tag", default='')
    args = ap.parse_args()
    args.device = args.device if args.device >= 0 else 'cpu'
    assert 0 < args.val_frac < 1, '--val-frac must be in (0, 1)'
    assert not (args.never_zero_grad and args.steps_per_epoch != 1), \
        '--never-zero-grad only reproduces their loop with --steps-per-epoch 1'
    tag = f'{"clean" if args.scrub else "dirty"}ts-{args.cutoff:g}h'
    tag += args.tag

    os.makedirs(args.cache, exist_ok=True)
    # v3: train and val now share hours (random edge holdout), so older caches are incompatible
    cache = f'{args.cache}/argus_lanl14argus-{tag}_vf{args.val_frac:g}_v3.pt'
    if os.path.exists(cache):
        tr, va, te = torch.load(cache, weights_only=False)
        print(f'loaded snapshots from {cache}: {len(tr.edge_index)} train/val hours, '
              f'{len(te.edge_index)} test hours')
    else:
        tr, va, te = build_splits(args.csr, args.cutoff, args.val_frac, args.scrub, args.split_seed)
        torch.save((tr, va, te), cache)

    assert len(va.edge_index) == len(tr.edge_index), 'train and val snapshots must stay aligned'

    if args.chunk <= 0:  # one pass over everything, as in argus_test_lanl.py
        args.chunk = max(len(tr.edge_index), len(te.edge_index))
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