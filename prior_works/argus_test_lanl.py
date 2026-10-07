from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import time

import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.utils import add_remaining_self_loops
from tqdm import tqdm
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score

from argus_opt import SOAP
from fast_auc import fast_auc, fast_ap

def get_args():
    ap = ArgumentParser(description='Argus on the random (non-temporal) LANL split. Defaults '
                                    'reproduce the released code (C0ldstudy/Argus @ 0bd70e6).')
    # runtime
    ap.add_argument('--device', type=int, default=0, help='CUDA index, or -1 for CPU')
    ap.add_argument('--runs', type=int, default=10, help='independent training runs to average')
    ap.add_argument('--epochs', type=int, default=100, help='their main.py: 100')
    ap.add_argument('--patience', type=int, default=3, help='their main.py: 3')
    ap.add_argument('--threads', type=int, default=64, help='torch.set_num_threads')
    ap.add_argument('--seed', type=int, default=None, help='torch seed; omit for unseeded runs')

    # optimisation
    ap.add_argument('--lr', type=float, default=0.01, help='their main.py: 0.01')
    ap.add_argument('--bs', type=int, default=0,
                    help='snapshots per forward pass; 0 = the whole window, as they use. '
                         'Gradients accumulate and ONE step is taken per epoch whatever this is, '
                         'so it trades memory against BPTT length, not the optimizer regime')

    # model -- their main.py uses -H/--hidden 32 and -z/--zdim 16
    ap.add_argument('--h-dim', type=int, default=32)
    ap.add_argument('--z-dim', type=int, default=16)
    ap.add_argument('--s', type=int, default=5,
                    help='neighbours sampled in the aggregation step (only used with --aggregate)')
    ap.add_argument('--margin', type=float, default=0.8, help='APLoss margin')
    ap.add_argument('--gamma', type=float, default=0.01, help='APLoss moving-average factor')
    ap.add_argument('--aggregate', action='store_true',
                    help='apply the neighbour aggregation before decoding. Their repo defines it '
                         '(inner_forward_sm) and never calls it, so it is OFF by default')

    # data and output
    ap.add_argument('--csr', default='../data/lanl14argus_tgraph_raw_csr.pt',
                    help='the full graph; the tr/va/te partitions are built from it')
    ap.add_argument('--dataset', default='lanl14argus', help='label used for cache and output names')
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--test-frac', type=float, default=0.10,
                    help='edge_tvt_split in lanl_preprocess/load_utils.py, "From Argus src", '
                         'uses 85/5/10')
    ap.add_argument('--split-seed', type=int, default=0)
    ap.add_argument('--cache', default='tmp', help='where the built snapshots are cached')
    ap.add_argument('--tag', default='',
                    help='suffix for output filenames so parallel configs do not overwrite '
                         "each other, e.g. --tag _agg alongside --aggregate")
    ap.add_argument('--max-pos', type=int, default=0,
                    help="cap positives per snapshot in the loss (0 = no cap). APLoss allocates "
                         'n_pos x 2*n_pos matrices, so its memory is QUADRATIC in a snapshot\'s '
                         'edge count; capping bounds the busiest hours. A deviation from their '
                         'code, which has no cap -- but their default device is CPU')
    ap.add_argument('--probe', action='store_true',
                    help='report snapshot sizes and the implied APLoss memory, then exit')
    ap.add_argument('--speedtest', action='store_true',
                    help='write one timing row after the first epoch, then exit')
    return ap.parse_args()

def squared_loss(margin, t): return (margin - t)** 2
def squared_hinge_loss(margin, t): return torch.max(margin - t, torch.zeros_like(t)) ** 2
def logistic_loss(margin, t): return torch.log(1+torch.log(-margin*t))

def _check_tensor_shape(inputs, shape=(-1, 1)):
    input_shape = inputs.shape
    target_shape = shape
    if len(input_shape) != len(target_shape):
        inputs = inputs.reshape(target_shape)
    return inputs

def _get_surrogate_loss(backend='squared_hinge'):
    if backend == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif backend == 'squared':
       surr_loss = squared_loss
    elif backend == 'logistic':
       surr_loss = logistic_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss


class APLoss(torch.nn.Module):
    """AP Loss with squared-hinge function: a novel loss function to directly optimize AUPRC.

        Args:
            margin: margin for squred hinge loss, e.g., m in [0, 1]
            gamma: factors for moving average

        Return:
            loss value (scalar)

        Reference:
                Stochastic Optimization of Areas Under Precision-Recall Curves with Provable Convergence},
                Qi, Qi and Luo, Youzhi and Xu, Zhao and Ji, Shuiwang and Yang, Tianbao,
                Advances in Neural Information Processing Systems 2021.

        Notes: This version of AP loss reduces redundant computation for the original implementation by Qi Qi.
        In addition, it fixed a potential memory leaking issue related to 'u_all' and 'u_pos'. This version is contributed by Gang Li.
    """

    def __init__(self, pos_len=None, margin=1.0, gamma=0.99, surrogate_loss='squared_hinge', device=None):
        super(APLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # self.device = torch.device('cpu')

        else:
            self.device = device
        self.u_all = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)#.cpu()
        self.u_pos = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)#.cpu()
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def forward(self, y_pred, y_true, index_p):
        CHECKMEORY=False
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))
        index_p = index_p[index_p>=0] # only need indices from positive samples

        pos_mask = (y_true == 1).flatten()
        f_ps = y_pred[pos_mask]
        mat_data = y_pred.flatten().repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))#.detach() # memory leak here
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        pos_sur_loss = sur_loss * pos_mask
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_all[index_p] = (1 - self.gamma) * self.u_all[index_p] + self.gamma * (sur_loss.mean(1, keepdim=True)).detach() # memory leak here
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_pos[index_p] = (1 - self.gamma) * self.u_pos[index_p] + self.gamma * (pos_sur_loss.mean(1, keepdim=True)).detach()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        p = (self.u_pos[index_p] - (self.u_all[index_p]) * pos_mask) / (self.u_all[index_p] ** 2) # size of p: len(f_ps)* len(y_pred)
        # p.detach_()
        loss = torch.mean(p * sur_loss)
        # loss = (p * sur_loss).mean()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        # del sur_loss, pos_sur_loss, p
        return loss

class GRU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(
            x_dim, h_dim, num_layers=hidden_units
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)

        self.z_dim = z_dim

    def forward(self, xs, h0, include_h=False):
        xs = self.drop(xs)

        if isinstance(h0, type(None)):
            xs, h = self.rnn(xs)
        else:
            xs, h = self.rnn(xs, h0)

        if not include_h:
            return self.lin(xs)

        return self.lin(xs), h

class Argus(nn.Module):
    def __init__(self, in_dim, edge_dim, h_dim, z_dim, device, s=5, pos_samples=315163,
                 margin=0.8, gamma=0.01, aggregate=False, max_pos=0):
        super().__init__()

        self.c1 = GCNConv(in_dim, h_dim).to(device)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, h_dim).to(device)
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        self.c3 = GCNConv(h_dim, h_dim).to(device)
        nn4 = nn.Sequential(nn.Linear(edge_dim, 8, device=device), nn.ReLU(), # lanl: 3 or 10; optc: 5
                            nn.Linear(8, h_dim * h_dim, device=device))

        self.c4 = NNConv(h_dim, h_dim, nn4, aggr='mean').to(device)
        self.rnn = GRU(h_dim, h_dim, z_dim).to(device)

        self.decode_mlp = nn.Sequential(
            nn.Linear(z_dim, z_dim, device=device),
            nn.Softmax(dim=1)
        )

        self.device = device
        # UNSW
        # self.ap_loss = APLoss(pos_len=1775035, margin=0.8, gamma=0.1, surrogate_loss='squared', device=device)
        self.ap_loss = APLoss(pos_len=pos_samples, margin=margin, gamma=gamma, surrogate_loss='squared', device=device)
        self.s = s
        # Their repo never applies the neighbour aggregation: DetectorEncoder.inner_forward_sm
        # exists and is called nowhere, and Argus_LANL.forward_once returns self.ac(x) straight
        # out. The comment in forward() below already noticed the scores are better without it.
        self.aggregate = aggregate
        self.max_pos = max_pos


    def forward(self, x, eis, eas, idxs, ptrs, h0=None, include_h=False):
        eis = [ei.to(self.device) for ei in eis]
        eas = [ea.to(self.device) for ea in eas]
        zs = []

        for i in range(len(eis)):
            ei = eis[i]; ea = eas[i]
            ei_self_loops = add_remaining_self_loops(ei)[0]

            z = self.c1(x.to(self.device), ei_self_loops)
            z = self.c2(z, ei_self_loops) # Comment out for UNSW
            z = self.relu(z)
            z = self.drop(z)
            z = self.c3(z, ei_self_loops) # Comment out for UNSW
            z = self.relu(z) # Comment out for UNSW
            z = self.drop(z) # Comment out for UNSW
            z = self.c4(z, ei, edge_attr=ea)
            z = self.ac(z)

            zs.append(z)

        zs = torch.stack(zs, dim=0)
        out, h = self.rnn(zs, h0, include_h=True)

        '''
        Scores are much better when we skip the aggregation step
        AUC 0.05 -> what's reported (at least 0.6, but it's still training)
            Skipped in UNSW
        '''
        if self.aggregate:
            zs = []
            for t in range(out.size(0)):
                z = out[t]
                z = self.sample_z(z, idxs[t], ptrs[t])
                zs.append(z)
            out = torch.stack(zs)

        # out is already (T, N, z) from the GRU, so no transpose is needed either way.
        return (out, h) if include_h else out

    def sample_z(self, z, idx,ptr):
        z_agg = []
        idx = idx.to(self.device)
        ptr = ptr.to(self.device)

        for i in range(z.size(0)):
            n_neighbors = idx[i+1]-idx[i]

            if n_neighbors:
                neighbors = ptr[idx[i] + torch.ones(n_neighbors, device=self.device).multinomial(self.s, replacement=True)]
                z_agg.append((z[neighbors].sum(dim=0) + z[i]) / (self.s+1))
            else:
                z_agg.append(z[i])

        z_agg = torch.stack(z_agg)
        return self.decode_mlp(z_agg)

    def decode(self, ei, z):
        # Their Euler_Embed_Unit.decode is sigmoid(dot). Without the sigmoid the scores are
        # unbounded, and APLoss's margin of 0.8 means something completely different.
        return torch.sigmoid((z[ei[0]] * z[ei[1]]).sum(dim=1))

    def sample_negatives(self, ps, num_nodes):
        '''Uniform pairs with real edges rejected, as their fast_negative_sampling does.'''
        ns = torch.randint(0, num_nodes, ps.size(), device=ps.device)
        key = lambda e: e[0] + e[1] * num_nodes      # the hash they use
        pos_key = key(ps)
        for _ in range(5):
            bad = torch.isin(key(ns), pos_key)
            if not bad.any():
                break
            ns[:, bad] = torch.randint(0, num_nodes, (2, int(bad.sum())), device=ps.device)
        return ns

    def calc_loss_argus(self, zs, eis):
        tot_loss = torch.zeros(1).to(self.device)
        for i in range(len(zs)):
            ps = eis[i]
            if ps.size(1) == 0:
                continue

            ps = ps.to(self.device)
            if self.max_pos and ps.size(1) > self.max_pos:
                # APLoss is O(n_pos^2); the busiest LANL hours are many times the median, so an
                # uncapped snapshot can OOM on its own whatever the batch size.
                keep = torch.randperm(ps.size(1), device=ps.device)[:self.max_pos]
                ps = ps[:, keep]
            ns = self.sample_negatives(ps, zs.size(1))

            t_index = torch.arange(0, ps.size(1), dtype=torch.int64, device=self.device).detach()
            # OBSERVED edges are the positive class. This was inverted: pos_pred held the random
            # pairs, so APLoss was maximising AP for random pairs rather than for real edges.
            # Their calc_loss_argus labels self.decode(ps, ...) as 1.
            pos_pred = self.decode(ps, zs[i])
            neg_pred = self.decode(ns, zs[i])

            tot_loss += self.ap_loss(
                torch.cat((pos_pred, neg_pred), 0),
                torch.cat((torch.ones(pos_pred.size(0)),torch.zeros(neg_pred.size(0))), 0).to(self.device).detach(),
                t_index)

        return tot_loss.true_divide(len(zs))

    def validate(self, zs, eis):
        pos, neg = [],[]
        for i in range(len(zs)):
            ps = eis[i]
            if ps.size(1) == 0:
                continue

            ps = ps.to(self.device)
            ns = self.sample_negatives(ps, zs.size(1))
            pos_pred = self.decode(ps, zs[i])
            neg_pred = self.decode(ns, zs[i])

            pos.append(pos_pred)
            neg.append(neg_pred)

        pos = torch.cat(pos)
        neg = torch.cat(neg)
        return pos, neg


def random_masks(g, val_frac, test_frac, seed):
    """
    Random edge partition of the full graph, with every malicious edge forced into test.

    Proportions follow edge_tvt_split in lanl_preprocess/load_utils.py (85/5/10), which is
    labelled "From Argus src". Anomalies go to test wholesale, as temporal_split.py does for the
    -ts split ("Mask out anomalies so they're all in test set") -- otherwise ~90% of the 439
    redteam edges would sit unlabelled in training and the test set would be missing most of
    what it is supposed to find.
    """
    n = g.col.size(0)
    is_mal = torch.zeros(n, dtype=torch.bool)
    is_mal[g.is_mal] = True

    gen = torch.Generator().manual_seed(seed)   # local, so it does not perturb training randomness
    rnd = torch.randperm(n, generator=gen)
    n_va, n_te = int(n * val_frac), int(n * test_frac)

    va = torch.zeros(n, dtype=torch.bool); va[rnd[:n_va]] = True
    te = torch.zeros(n, dtype=torch.bool); te[rnd[n_va:n_va + n_te]] = True
    tr = torch.zeros(n, dtype=torch.bool); tr[rnd[n_va + n_te:]] = True

    tr &= ~is_mal
    va &= ~is_mal
    te |= is_mal

    print(f'edges: train {int(tr.sum()):,}, val {int(va.sum()):,}, test {int(te.sum()):,} '
          f'({int(is_mal.sum())} malicious, all in test)')
    print(f'test prevalence: {float(is_mal.sum()) / max(1, int(te.sum())):.3e}')
    return tr, va, te, is_mal


def masked_graph(g, mask, is_mal):
    """One split's edges as a flat Data with the fields to_snapshots() needs."""
    deg = g.idxptr[1:] - g.idxptr[:-1]
    src = torch.arange(g.x.size(0)).repeat_interleave(deg)
    return Data(
        x=g.x,
        src=src[mask],
        col=g.col[mask],
        ts=g.ts[mask] // (60 * 60),          # to_snapshots buckets by hour
        raw_edge_attr=g.raw_edge_attr[mask],
        label=is_mal[mask].long(),
    )


def build_splits(args):
    """tr/va/te snapshot tensors, built from the full graph rather than pre-split files."""
    g = torch.load(args.csr, weights_only=False)
    tr_m, va_m, te_m, is_mal = random_masks(g, args.val_frac, args.test_frac, args.split_seed)

    flat = {k: masked_graph(g, m, is_mal) for k, m in (('tr', tr_m), ('va', va_m), ('te', te_m))}
    # All three are bucketed over the TRAINING timestamps, as the original main block did, so the
    # three splits stay aligned index-for-index and zs[i] lines up with te.edge_index[i].
    ts = flat['tr'].ts.unique()
    print(f'{ts.size(0)} snapshots')

    # neighbour lists are only needed by the aggregation, and only for the graph fed to the model
    tr = to_snapshots(flat['tr'], ts, add_csr=args.aggregate)
    va = to_snapshots(flat['va'], ts)
    te = to_snapshots(flat['te'], ts)
    return tr, va, te


def to_snapshots(g, ts=None, add_csr=False, last_ts=None):
    # Assumes graph has src and col already
    ei = torch.stack([g.src, g.col])

    # Normalize (already done for LANL)
    #g.raw_edge_attr = g.raw_edge_attr / g.raw_edge_attr.max(dim=0).values

    eis = []
    eas = []
    idxs = []
    ptrs = []
    y = []
    for t in ts:
        if last_ts and t > last_ts:
            continue

        mask = g.ts == t
        ei_t = ei[:, mask]
        ea_t = g.raw_edge_attr[mask]

        eis.append(ei_t)
        eas.append(ea_t)

        idx = [0]
        ptr = []

        if add_csr:
            csr_dict = defaultdict(list)
            for i in range(ei_t.size(1)):
                src,dst = ei_t[:, i]
                csr_dict[src.item()].append(dst.item())

            for i in range(g.x.size(0)):
                ptr += csr_dict[i]
                idx.append(idx[-1] + len(csr_dict[i]))

            idxs.append(torch.tensor(idx))
            ptrs.append(torch.tensor(ptr))

        if 'label' in g.keys():
            label = g.label[mask]
            y.append(label)

    x = torch.eye(g.x.size(0))
    return Data(x=x, edge_index=eis, label=y, eas=eas, idxs=idxs, ptrs=ptrs)

def probe_memory(tr, args):
    """
    Snapshot size distribution and the APLoss memory it implies.

    APLoss.forward builds mat_data = y_pred.repeat(n_pos, 1), an n_pos x 2*n_pos float32 matrix,
    plus (f_ps - mat_data), sur_loss, pos_sur_loss and p of the same shape. sur_loss and p stay
    resident until backward, so roughly 2 matrices per snapshot are RETAINED and about 6 exist at
    the peak of a single call. Memory is therefore quadratic in a snapshot's edge count, and LANL
    is bursty, so the worst hour matters more than the mean.
    """
    n = torch.tensor([ei.size(1) for ei in tr.edge_index], dtype=torch.float)
    qs = torch.quantile(n, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
    lo, med, p90, p99, hi = [float(q) for q in qs]
    mat = lambda e: e * (2 * e) * 4 / 2 ** 30     # one n_pos x 2*n_pos float32 matrix, GiB

    print(f'{len(n)} training snapshots, {int(n.sum()):,} edges')
    print(f'edges/snapshot: min {lo:.0f}  median {med:.0f}  p90 {p90:.0f}  p99 {p99:.0f}  max {hi:.0f}')
    print(f'\nAPLoss, one matrix: median {mat(med):.2f} GiB, max {mat(hi):.2f} GiB')
    print(f'  retained per snapshot (~2):  median {2*mat(med):.2f} GiB, max {2*mat(hi):.2f} GiB')
    print(f'  transient peak, one call (~6): median {6*mat(med):.2f} GiB, max {6*mat(hi):.2f} GiB')

    free = 0.0
    if torch.cuda.is_available() and args.device != 'cpu':
        free = torch.cuda.mem_get_info(args.device)[1] / 2 ** 30
        print(f'\ndevice total {free:.1f} GiB, of which ~1.0 GiB goes to torch.eye({tr.x.size(0)})')

    print('\nA chunk of B snapshots retains about 2*B matrices plus the peak of the call in '
          'flight. Sorted worst-first, the largest snapshots are:')
    top = torch.topk(n, min(8, len(n))).values
    print('  ' + ', '.join(f'{int(v):,} ({2*mat(float(v)):.2f} GiB)' for v in top))

    if free:
        budget = free - 1.5
        est = 2 * mat(med)
        print(f'\nWith ~{budget:.0f} GiB usable, --bs of roughly {max(1, int(budget / est))} fits '
              'the MEDIAN snapshot, but a single worst-case hour needs '
              f'{6*mat(hi):.1f} GiB on its own.')
        if 6 * mat(hi) > budget:
            cap = int((budget / 6 / 4 * 2 ** 30) ** 0.5 / 2 ** 0.5)
            print(f'  That does not fit, so --bs alone cannot fix this. Either run on CPU '
                  f'(--device -1, which is their default) or cap with --max-pos {cap // 1000 * 1000}.')


def train(tr, va, te, args):
    # was tr.edge_index[:42], a leftover from the 42-hour temporal window. This only sizes
    # APLoss's moving-average buffer, which must exceed the largest snapshot's edge count.
    pos_samples = sum(ei.size(1) for ei in tr.edge_index)
    # their main.py: -H/--hidden 32, -z/--zdim 16, and worker_args = [hidden, hidden] so c4 is
    # h_dim -> h_dim with the reduction to z_dim in the GRU's output projection -- which is the
    # shape Argus.__init__ already builds. Only the sizes were wrong. At 128/64 the NNConv
    # per-edge weight matrix is 32x larger, which is also what made a full-window pass OOM.
    model = Argus(tr.x.size(0), tr.eas[0].size(1), args.h_dim, args.z_dim, args.device,
                  s=args.s, pos_samples=pos_samples, margin=args.margin, gamma=args.gamma,
                  aggregate=args.aggregate, max_pos=args.max_pos)
    opt = SOAP(model.parameters(), lr=args.lr, mode='adam', weight_decay=0.0)

    best = (0,0,0)
    best_cheating = (0,0)
    BS = args.bs if args.bs > 0 else len(tr.edge_index)
    no_progress = 0
    n_snaps = len(tr.edge_index)
    for e in range(args.epochs):
        model.train()
        opt.zero_grad()
        h = None
        # ONE optimizer step per epoch whatever BS is: their classification.py sets BS to the
        # whole window and steps once, so gradients accumulate across BS-sized pieces, each
        # weighted by its share of the window, and the GRU state is threaded between them.
        # Lowering BS for memory then costs only BPTT length, not the optimizer regime.
        # (The old `range(len // BS)` silently dropped the tail whenever BS did not divide the
        # window evenly -- with BS=24 over 352 snapshots it trained on 336 and ignored 16.)
        for st_i in tqdm(range(0, n_snaps, BS)):
            en_i = min(st_i + BS, n_snaps)
            w = (en_i - st_i) / n_snaps
            zs, h = model.forward(tr.x, tr.edge_index[st_i:en_i], tr.eas[st_i:en_i], tr.idxs[st_i:en_i], tr.ptrs[st_i:en_i], h0=h, include_h=True)
            loss = model.calc_loss_argus(zs, tr.edge_index[st_i:en_i])
            (loss * w).backward()
            h = h.detach()   # state crosses the boundary, gradient does not

        opt.step()

        with torch.no_grad():
            model.eval()
            zs = model.forward(tr.x, tr.edge_index, tr.eas, tr.idxs, tr.ptrs)
            # was zs[:42] -- a leftover from the 42-hour temporal window that validated on 42 of
            # the 352 snapshots of this random split, for no reason and with a weaker signal.
            pos,neg = model.validate(zs, va.edge_index)
            # Observed edges are the positive class, matching the loss above and their
            # DetectorEncoder.score_edges. Was labels[pos.size(0):] = 1, i.e. random pairs
            # labelled positive, which inverted validation along with the loss.
            labels = torch.zeros(pos.size(0)+neg.size(0))
            labels[:pos.size(0)] = 1

            preds = torch.cat([pos,neg]).cpu().numpy()
            labels = labels.numpy()

            va_auc = fast_auc(labels, preds)
            va_ap = fast_ap(labels, preds)

            print(f'\tVal AUC: {va_auc:0.4f}, AP: {va_ap:0.4f}')

            preds = []
            for i in range(len(te.edge_index)):
                # go through model.decode so the sigmoid fix applies here too; this inlined the
                # dot product and so bypassed it
                preds.append(model.decode(te.edge_index[i].to(args.device), zs[i]).cpu())

            # decode() is sigmoid(dot): HIGH means "looks like a real edge", so the ANOMALY score
            # is its complement. Their score_stats does exactly this (`scores = 1-scores`) before
            # computing auc/ap.
            preds = (1 - torch.cat(preds)).numpy()
            y = torch.cat(te.label).clamp(0,1).numpy()

            auc = fast_auc(y, preds)
            ap = fast_ap(y, preds)
            print(f'\tTe AUC: {auc:0.4f}, AP: {ap:0.4f}', end='', flush=True)

            if va_auc+va_ap > best[0]:
                best = (va_auc+va_ap, auc, ap)
                print('*')
                no_progress = 0
            else:
                print()
                if e >= 1:      # their classification.py only counts misses after the first epoch
                    no_progress += 1

            # Validation doesn't want to work. I want to give Argus a fair
            # run, so let's just keep track of the best scores without
            # using the val set (this is data snooping, but even with
            # snooping, it doesn't seem like it will perform well)
            # NOTE: this now tracks the best AP, not the best AUC. 'ap_snooped' previously meant
            # "AP at the best-AUC epoch", which is not the number you want to compare against a
            # reported AP of 0.3227.
            if ap > best_cheating[1]:
                best_cheating = (auc, ap)

            if no_progress > args.patience:
                break

            print(json.dumps({'auc': best[1], 'ap': best[2], 'auc_last': auc, 'ap_last': ap, 'auc_snooped': best_cheating[0], 'ap_snooped': best_cheating[1]}, indent=1))

    print(f"Best: AUC {best[1]:0.4f}, AP {best[2]:0.4f}")
    return {'auc': best[1], 'ap': best[2], 'auc_last': auc, 'ap_last': ap, 'auc_snooped': best_cheating[0], 'ap_snooped': best_cheating[1]}

if __name__ == '__main__':
    args = get_args()
    args.device = args.device if args.device >= 0 else 'cpu'
    if args.seed is not None:
        torch.manual_seed(args.seed)
    os.makedirs(args.cache, exist_ok=True)
    print(args)

    # cache key carries everything that changes the partition, so switching --val-frac,
    # --test-frac or --split-seed rebuilds instead of silently reusing the wrong split
    cache = (f'{args.cache}/argus_{args.dataset}_v{args.val_frac:g}_t{args.test_frac:g}'
             f'_s{args.split_seed}{"_agg" if args.aggregate else ""}.pt')
    if os.path.exists(cache):
        tr, va, te = torch.load(cache, weights_only=False)
        print(f'loaded snapshots from {cache}')
    else:
        tr, va, te = build_splits(args)
        torch.save((tr, va, te), cache)


    torch.set_num_threads(args.threads)
    if args.probe:
        probe_memory(tr, args)
        raise SystemExit

    best = []
    for r in range(args.runs):
        print(f'\n===== run {r} =====')
        best.append(train(tr, va, te, args))
        with open(f'argus_{args.dataset}{args.tag}_log.csv', 'a') as f:
            f.write(json.dumps(best[-1]) + '\n')

    df = pd.DataFrame(best)
    df.loc['mean'] = df.mean()
    df.loc['sem'] = df.sem()

    print(df)
    df.to_csv(f'argus_results_{args.dataset}{args.tag}.csv')