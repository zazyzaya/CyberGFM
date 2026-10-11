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

The model comes from argus.py, which all three Argus scripts now share -- see its docstring for
the six fixes and, in particular, for the score convention: decode() returns a reconstruction
probability where HIGH MEANS NORMAL, so detection_metrics negates it.

Run from prior_works/ (needs data/lanl14argus_tgraph_raw_csr.pt for Argus's edge features):

    python argus_test_lanl_ts.py --device 0 --runs 5                       # paper's protocol
    python argus_test_lanl_ts.py --device 0 --runs 5 --cutoff 168          # poisoning, dirty
    python argus_test_lanl_ts.py --device 1 --runs 5 --cutoff 168 --scrub  # poisoning, clean
'''
from argparse import ArgumentParser
from collections import defaultdict
import os
import sys
import time

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

# One shared model definition; see argus.py for the six fixes and the score convention
# (decode() is a reconstruction probability -- HIGH MEANS NORMAL, so detection negates).
from argus import Argus, SOAP, append_result, build_once, run_record

sys.path.insert(0, '..')
from metrics import prevalence, threshold_metrics  # noqa: E402

HOUR = 60 * 60


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
    of the graph that represents it. Causality relies on the GRU scanning time (argus.py, fix 2).

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
    '''Argus at their main.py's dimensions unless overridden; the class is in argus.py.'''
    return Argus(tr.x.size(0), tr.eas[0].size(1), args.h_dim, args.z_dim, args.device,
                   s=args.s, pos_samples=sum(e.size(1) for e in tr.edge_index),
                   gru_layers=args.gru_layers, beta=args.beta, aggregate=args.aggregate,
                   margin=args.margin, gamma=args.gamma,
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
                         '(paper: 0.01 for LANL). Their released code has NO L_DEC term at all, so any '
                         'nonzero value runs this repo\'s per-edge BCE reading of eq. 12, which '
                         'is a GUESS; --beta 0 reproduces their code exactly')
    ap.add_argument('--gamma', type=float, default=0.01,
                    help='APLoss moving-average factor (paper: 0.01 for LANL)')
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
    ap.add_argument('--steps-per-epoch', type=int, default=1,
                    help='optimizer steps per epoch; gradients accumulate across chunks within a '
                         'step. 1 reproduces argus_test_lanl.py (BS = the whole window) regardless '
                         'of how small --chunk has to be for memory')
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
    ap.add_argument('--scrub', action='store_true', help='remove in-window malicious edges (cleants control)')
    ap.add_argument('--tag', default='',
                    help='variant suffix for the results file, after the automatic '
                         '{clean|dirty}ts-{cutoff}h part, e.g. --tag _b0 alongside --beta 0. '
                         'Seeds of one experiment share a file; do not tag per seed')
    ap.add_argument('--chunk', type=int, default=24,
                    help='snapshots per forward pass; 0 = the whole window in one pass, which is '
                         'what argus_test_lanl.py does (BS = len(tr.edge_index), so one optimizer '
                         'step per epoch over the full sequence)')
    ap.add_argument('--probe', action='store_true',
                    help='report train-step peak memory at several chunk sizes, then exit')
    ap.add_argument('--cache', default='tmp')
    args = ap.parse_args()
    args.device = args.device if args.device >= 0 else 'cpu'
    assert 0 < args.val_frac < 1, '--val-frac must be in (0, 1)'
    assert not (args.never_zero_grad and args.steps_per_epoch != 1), \
        '--never-zero-grad only reproduces their loop with --steps-per-epoch 1'
    # The split is named without --tag: the tag only distinguishes runs, and putting it in the
    # cache name made every tagged job rebuild an identical copy of the same split.
    split_name = f'lanl14argus-{"clean" if args.scrub else "dirty"}ts-{args.cutoff:g}h'

    # v3: train and val now share hours (random edge holdout), so older caches are incompatible
    cache = f'{args.cache}/argus_{split_name}_vf{args.val_frac:g}_v3.pt'
    (tr, va, te), = build_once([cache], lambda: [build_splits(
        args.csr, args.cutoff, args.val_frac, args.scrub, args.split_seed)])
    print(f'{len(tr.edge_index)} train/val hours, {len(te.edge_index)} test hours')

    assert len(va.edge_index) == len(tr.edge_index), 'train and val snapshots must stay aligned'

    if args.chunk <= 0:  # one pass over everything, as in argus_test_lanl.py
        args.chunk = max(len(tr.edge_index), len(te.edge_index))
        print(f'--chunk 0: using {args.chunk} (whole window, one optimizer step per epoch)')

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.probe:
        probe_memory(tr, args)
        raise SystemExit

    out = f'{args.out_dir}/{split_name}{args.tag}.csv'
    for r in range(args.runs):
        seed = None if args.seed is None else args.seed + r
        if seed is not None:
            torch.manual_seed(seed)
        print(f'\n===== run {r} ({split_name}, seed {seed}) =====')
        t0 = time.time()
        res = train(tr, va, te, args)
        append_result(out, {**run_record(args, seed, (time.time() - t0) / 60,
                                         dataset=split_name, cutoff=args.cutoff,
                                         scrub=int(args.scrub), score_mode=args.score_mode),
                            **res})
        print(f'appended to {out}')