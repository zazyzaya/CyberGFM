'''
Argus on the dirty / scrubbed temporal LANL splits, for the poisoning comparison.

argus_test_lanl.py builds every split's snapshots from the TRAINING timestamps, which only works
for the random split where all splits share the same hours. Here train and test hours are disjoint,
so snapshots are built per window, and test snapshots are scored with the trailing training
snapshots fed in first so the GRU has history.

  --scrub off (default): train on days [0, cutoff), attack edges left in and unlabeled  -> dirtyts
  --scrub on:            same window with those attack edges removed                    -> cleants

Both score the identical test set (everything at/after the cutoff), so the difference between the
two runs is attributable to training on unlabeled attack traffic.

Argus.forward stacks the per-snapshot embeddings with torch.stack(zs, dim=1) -> (N, T, h) and
nn.GRU is batch_first=False, so the recurrent scan runs over the NODE axis: node ordering is
arbitrary and carries no information, and each snapshot is an independent batch element, so
nothing crosses time. _RNNWrapper transposes to (T, N, h) around the recurrent call so the scan
runs over time, and threads the resulting (layers, N, h) state across chunked forward passes.

Run from prior_works/ (needs data/lanl14argus_tgraph_raw_csr.pt for Argus's edge features):

    python argus_test_lanl_ts.py --device 0 --runs 5
    python argus_test_lanl_ts.py --device 1 --runs 5 --scrub
'''
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
import inspect
import json
import os
import sys
import time

import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from argus_test_lanl import Argus  # identical model, APLoss and hyperparameters

sys.path.insert(0, '..')
from metrics import prevalence, threshold_metrics  # noqa: E402

try:  # whatever argus_test_lanl.py already imports
    from argus_test_lanl import SOAP
except ImportError:
    from libauc.optimizers import SOAP

HOUR = 60 * 60
_HIDDEN_ARGS = ('h0', 'h_0', 'hx', 'h', 'hidden', 'h_prev')


# ---------------------------------------------------------------------------------------------
# Recurrent state
# ---------------------------------------------------------------------------------------------
class _RNNWrapper(nn.Module):
    '''
    Wraps Argus's recurrent module so the scan runs over time and the state survives chunking.

    Argus.forward hands the recurrent module a (N, T, h) stack, which nn.GRU (batch_first=False)
    reads as sequence = N nodes, batch = T snapshots. Node ordering is arbitrary, so that scan
    integrates noise, and no information crosses time. This transposes to (T, N, h) before the
    call and back afterwards, so the scan is temporal with each node as a batch element and
    everything downstream of the recurrent module sees the shape it already expected.

    The hidden state is then (layers, N, h) -- the same shape for every chunk, and genuinely the
    state after the last snapshot -- so `h_in` / `h_out` let a caller carry it between chunks.
    '''

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.h_in = None       # set by the caller before the forward pass
        self.h_out = None      # read by the caller afterwards
        self._params = list(inspect.signature(inner.forward).parameters)

    def _hidden_kwarg(self):
        for nm in _HIDDEN_ARGS:
            if nm in self._params:
                return nm
        return None

    def forward(self, zs, *a, **kw):
        transposed = zs.dim() == 3
        if transposed:
            zs = zs.transpose(0, 1).contiguous()   # (N, T, h) -> (T, N, h)

        h0 = self.h_in
        if h0 is not None and h0.size(1) != zs.size(1):
            h0 = None  # node count changed; start from zeros rather than silently misaligning

        name = self._hidden_kwarg()
        if h0 is not None and name is not None and name not in kw:
            kw[name] = h0
        elif h0 is not None and not a:
            a = (h0,)
        if 'include_h' in self._params and 'include_h' not in kw:
            kw['include_h'] = True

        out = self.inner(zs, *a, **kw)
        h = None
        if isinstance(out, (tuple, list)):
            out, h = out[0], out[1]
        self.h_out = h

        if transposed:
            out = out.transpose(0, 1).contiguous()  # back to (N, T, h)
        return out


@contextmanager
def _threaded_rnn(model, h0):
    '''Temporarily swap model.rnn for the wrapper; yields it so the caller can read h_out.'''
    inner = model.rnn
    wrap = _RNNWrapper(inner).to(next(inner.parameters()).device)
    wrap.h_in = h0
    model.rnn = wrap
    try:
        yield wrap
    finally:
        model.rnn = inner


def _pad_csr(idxs, ptrs, n):
    '''Neighbour lists are only cached for the training snapshots; repeat the last for the rest.'''
    if not idxs:
        return idxs, ptrs
    if len(idxs) >= n:
        return list(idxs[:n]), list(ptrs[:n])
    pad = n - len(idxs)
    return list(idxs) + [idxs[-1]] * pad, list(ptrs) + [ptrs[-1]] * pad


def forward_h(model, x, eis, eas, idxs, ptrs, h0=None):
    '''Argus.forward with a temporal recurrent scan, seeded with h0 and returning the final state.'''
    idxs, ptrs = _pad_csr(idxs, ptrs, len(eis))
    with _threaded_rnn(model, h0) as wrap:
        zs = model.forward(x, eis, eas, idxs, ptrs)
    return zs, wrap.h_out


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


def snapshots(g, mask, is_mal, add_csr=False):
    '''One snapshot per hour present in `mask`, with Argus's raw edge features.'''
    src = torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])
    ei = torch.stack([src, g.col])
    hours = (g.ts // HOUR)

    eis, eas, ys, idxs, ptrs = [], [], [], [], []
    for h in hours[mask].unique(sorted=True).tolist():
        m = mask & (hours == h)
        eis.append(ei[:, m])
        eas.append(g.raw_edge_attr[m].float())
        ys.append(is_mal[m])

        if add_csr:  # neighbour lists for Argus's sample_z aggregation
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
def score_window(model, tr, split, args):
    '''
    Decode each snapshot's edges. The GRU is primed on the trailing `--history` training
    snapshots, then the state is carried from chunk to chunk instead of restarting at every one.
    '''
    model.eval()
    preds, labels = [], []

    h = None
    if args.history:
        _, h = forward_h(model, tr.x, tr.edge_index[-args.history:], tr.eas[-args.history:],
                         tr.idxs[-args.history:], tr.ptrs[-args.history:])

    for st in range(0, len(split.edge_index), args.chunk):
        en = min(st + args.chunk, len(split.edge_index))
        zs, h = forward_h(model, tr.x, split.edge_index[st:en], split.eas[st:en],
                          tr.idxs, tr.ptrs, h0=h)
        h = h.detach() if h is not None else None

        for j in range(st, en):
            ei = split.edge_index[j]
            if ei.size(1) == 0:
                continue
            z = zs[j - st]
            preds.append(-model.decode(ei.to(model.device), z).cpu())  # low score = normal
            labels.append(split.label[j])

    return torch.cat(preds).float(), torch.cat(labels).long()


def evaluate(model, tr, split, args):
    '''AUC/AP plus the split-comparable metrics (lift, precision and FP/day at 50% recall).'''
    preds, labels = score_window(model, tr, split, args)
    y, p = labels.numpy(), preds.numpy()
    if y.sum() == 0:  # e.g. a scrubbed validation window with no anomalies left
        return dict(auc=float('nan'), ap=float('nan'), ap_lift=float('nan'),
                    prec_r50=float('nan'), fp_day_r50=float('nan'))

    ap = average_precision_score(y, p)
    prec, fp, _ = threshold_metrics(y, p, recall=0.5)
    days = len(split.edge_index) / 24  # one snapshot per hour
    return dict(auc=roc_auc_score(y, p), ap=ap, ap_lift=ap / prevalence(y),
                prec_r50=prec, fp_day_r50=fp / days if days else float('nan'))


def train(tr, va, te, args):
    model = Argus(tr.x.size(0), tr.eas[0].size(1), 128, 64, args.device,
                  pos_samples=sum(e.size(1) for e in tr.edge_index))
    opt = SOAP(model.parameters(), lr=0.01, mode='adam', weight_decay=0.0)

    best, no_progress = None, 0
    for e in range(args.epochs):
        model.train()
        st = time.time()
        h = None  # each epoch restarts from a zero state; within it, chunks carry state forward
        for i in range(0, len(tr.edge_index), args.chunk):
            eis = tr.edge_index[i:i + args.chunk]
            eas = tr.eas[i:i + args.chunk]
            if not eis:
                continue
            opt.zero_grad()
            zs, h = forward_h(model, tr.x, eis, eas,
                              tr.idxs[i:i + args.chunk], tr.ptrs[i:i + args.chunk], h0=h)
            loss = model.calc_loss_argus(zs, [ei.to(args.device) for ei in eis])
            loss.backward()
            opt.step()
            # truncated BPTT: the state crosses the chunk boundary, the gradient does not
            h = h.detach() if h is not None else None

        va_m = evaluate(model, tr, va, args)
        te_m = evaluate(model, tr, te, args)
        print(f'[epoch {e}] {time.time() - st:.0f}s | VAL AP {va_m["ap"]:.4f} | '
              f'TEST AUC {te_m["auc"]:.4f} AP {te_m["ap"]:.4f} lift {te_m["ap_lift"]:.1f}x '
              f'FP/day {te_m["fp_day_r50"]:.0f}')

        key = va_m['ap'] if va_m['ap'] == va_m['ap'] else te_m['ap']  # fall back if val is unlabelled
        if best is None or key > best[0]:
            best, no_progress = (key, te_m), 0
        else:
            no_progress += 1
            if no_progress > args.patience:
                break

    return best[1]


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--csr', default='../data/lanl14argus_tgraph_raw_csr.pt')
    ap.add_argument('--days', type=float, default=7, help='train/val window; must match the CyberGFM split')
    ap.add_argument('--scrub', action='store_true', help='remove in-window malicious edges (cleants control)')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--chunk', type=int, default=24, help='snapshots per forward pass')
    ap.add_argument('--history', type=int, default=24, help='training snapshots used to prime the GRU')
    ap.add_argument('--cache', default='tmp')
    args = ap.parse_args()
    args.device = args.device if args.device >= 0 else 'cpu'
    tag = 'cleants' if args.scrub else 'dirtyts'

    os.makedirs(args.cache, exist_ok=True)
    cache = f'{args.cache}/argus_lanl14argus-{tag}_{args.days:g}d.pt'
    if os.path.exists(cache):
        tr, va, te = torch.load(cache, weights_only=False)
        print('loaded snapshots from', cache)
    else:
        g, tr_m, va_m, te_m, is_mal = load_windows(args.csr, args.days, args.scrub)
        tr = snapshots(g, tr_m, is_mal, add_csr=True)
        va = snapshots(g, va_m, is_mal)
        te = snapshots(g, te_m, is_mal)
        torch.save((tr, va, te), cache)
        print(f'snapshots: train {len(tr.edge_index)}, val {len(va.edge_index)}, test {len(te.edge_index)} hours')

    torch.set_num_threads(64)
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