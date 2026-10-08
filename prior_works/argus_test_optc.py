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

DELIBERATELY NOT CHANGED: the loss polarity. `calc_loss_argus` labels observed edges 0 and random
pairs 1, so the model learns an anomaly score directly and no negation is needed at test time.
Argus's released code labels observed edges 1 and then applies `scores = 1 - scores` before
scoring. The two are equivalent; do not "fix" one without removing the other.

PARALLELISM. One process = one replication (`--runs 1`, the default). Each finished run appends a
row to a shared CSV under an exclusive advisory lock, so N jobs on N GPUs can write to one file.
Build the snapshot cache once before fanning out -- `--prepare` -- or let the first job build it
while the others block on the same lock.

    python argus_test_optc.py --prepare                         # build tmp/optc-argus_{tr,va,te}.pt
    python argus_test_optc.py --device 0 --run-id 1              # one replication, appends one row
    python argus_test_optc.py --summarize                        # mean/sem over whatever is there

    # five in parallel as an LSF job array; LSB_JOBINDEX becomes --run-id automatically
    bsub -J "argus[1-5]" -q h100 -gpu "num=1:mode=exclusive_process" -n 4 -W 8:00 \
         -o logs/lsf/argus_%I.out ./argus_one.sh
'''
from argparse import ArgumentParser
from collections import defaultdict
import csv
import fcntl
import json
import os
import random
import socket
import time
import zlib
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score

from argus_opt import SOAP


# ------------------------------------------------------------------------------------------------
# Parallel-safe result collection
# ------------------------------------------------------------------------------------------------
FIELDS = ['run_id', 'seed', 'dataset', 'score_mode', 'h_dim', 'z_dim', 'lr', 's', 'aggregate',
          'epochs_ran', 'best_epoch', 'n_test_edges', 'n_malicious',
          'auc', 'ap', 'auc_last', 'ap_last', 'auc_snooped', 'ap_snooped',
          'host', 'finished', 'minutes']


class FileLock:
    '''
    Exclusive advisory lock on `<path>.lock`, held for the duration of the `with` block.

    The lock lives in its own file rather than on the CSV itself: the holder can then open,
    truncate or replace the real file freely while the lock is held, and the lock's existence
    never depends on the data file existing yet.

    LOCK_NB in a retry loop, not a blocking LOCK_EX, deliberately. If the filesystem's lock
    manager wedges -- which on a shared cluster is a thing that happens -- a blocking flock never
    returns and the job sits burning wall time with no output. This raises instead.

    On NFSv4 flock works. On NFSv3 without a running lockd it can silently no-op, in which case
    two simultaneous appends can interleave. If you cannot confirm the mount, use --per-run-files
    and skip locking altogether.
    '''

    def __init__(self, path, timeout=3600):
        self.lockpath = f'{path}.lock'
        self.timeout = timeout
        self.fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.lockpath)), exist_ok=True)
        self.fh = open(self.lockpath, 'a+')
        start = time.time()
        waited = False
        while True:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if waited:
                    print(f'[lock] acquired after {time.time() - start:.0f}s', flush=True)
                return self
            except OSError:
                if time.time() - start > self.timeout:
                    self.fh.close()
                    raise TimeoutError(
                        f'could not lock {self.lockpath} after {self.timeout}s. Another run is '
                        f'holding it, or a stale lock survived a kill -- check with '
                        f'`fuser {self.lockpath}` and delete the file if nothing holds it.')
                if not waited:
                    print(f'[lock] waiting on {self.lockpath} ...', flush=True)
                    waited = True
                time.sleep(0.5 + random.random())    # jitter: 5 jobs released together

    def __exit__(self, *exc):
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        self.fh.close()
        return False


def append_row(csv_path, row, per_run_files=False):
    '''Append one run's results. Serialised across processes, header written exactly once.'''
    if per_run_files:
        # No lock needed: every process writes a distinct path. summarize() globs them.
        d = f'{csv_path}.d'
        os.makedirs(d, exist_ok=True)
        p = f'{d}/run_{row["run_id"]}.csv'
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            w.writeheader()
            w.writerow(row)
        return p

    with FileLock(csv_path):
        # Inside the lock: whether a header is needed and the write itself have to be one
        # operation, or two processes that both saw an empty file both write a header.
        fresh = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            if fresh:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())    # so a reader on another node sees the row, not a stale page
    return csv_path


def summarize(csv_path):
    '''Mean/sem over every row on disk, whichever layout was used.'''
    import glob
    frames = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        frames.append(pd.read_csv(csv_path))
    frames += [pd.read_csv(p) for p in sorted(glob.glob(f'{csv_path}.d/run_*.csv'))]
    if not frames:
        print(f'no results yet at {csv_path}')
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[df['ap'].notna()]
    if df.empty:
        print(f'{csv_path}: rows present but no completed runs')
        return

    metrics = ['auc', 'ap', 'auc_last', 'ap_last', 'auc_snooped', 'ap_snooped']
    print(f'{csv_path}: {len(df)} completed run(s)')
    dupes = df['run_id'].duplicated().sum()
    if dupes:
        print(f'  WARNING {dupes} duplicate run_id(s) -- two jobs used the same id, so these are '
              f'not {len(df)} independent replications')
    for col in ('dataset', 'score_mode', 'h_dim', 'z_dim', 'lr', 'aggregate'):
        if col in df and df[col].nunique() > 1:
            print(f'  WARNING mixed {col}: {sorted(df[col].unique().tolist())} -- these rows are '
                  f'not the same experiment and should not be pooled')
    print()
    print(df[metrics].agg(['mean', 'sem', 'std', 'min', 'max']).to_string(float_format='%.4f'))
    if len(df) < 2:
        print('\n(sem is undefined for a single run)')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def squared_loss(margin, t): return (margin - t) ** 2
def squared_hinge_loss(margin, t): return torch.max(margin - t, torch.zeros_like(t)) ** 2
def logistic_loss(margin, t): return torch.log(1 + torch.log(-margin * t))


def _check_tensor_shape(inputs, shape=(-1, 1)):
    if len(inputs.shape) != len(shape):
        inputs = inputs.reshape(shape)
    return inputs


def _get_surrogate_loss(backend='squared_hinge'):
    return {'squared_hinge': squared_hinge_loss, 'squared': squared_loss,
            'logistic': logistic_loss}[backend]


class APLoss(torch.nn.Module):
    '''
    AP loss with a squared-hinge surrogate (Qi et al., NeurIPS 2021), as vendored by Argus.
    Unchanged.
    '''

    def __init__(self, pos_len=None, margin=1.0, gamma=0.99, surrogate_loss='squared_hinge',
                 device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.u_all = torch.tensor([0.0] * pos_len).reshape(-1, 1).to(self.device)
        self.u_pos = torch.tensor([0.0] * pos_len).reshape(-1, 1).to(self.device)
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def forward(self, y_pred, y_true, index_p):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))
        index_p = index_p[index_p >= 0]

        pos_mask = (y_true == 1).flatten()
        f_ps = y_pred[pos_mask]
        mat_data = y_pred.flatten().repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))
        pos_sur_loss = sur_loss * pos_mask

        self.u_all[index_p] = ((1 - self.gamma) * self.u_all[index_p]
                               + self.gamma * sur_loss.mean(1, keepdim=True).detach())
        self.u_pos[index_p] = ((1 - self.gamma) * self.u_pos[index_p]
                               + self.gamma * pos_sur_loss.mean(1, keepdim=True).detach())

        p = ((self.u_pos[index_p] - self.u_all[index_p] * pos_mask)
             / (self.u_all[index_p] ** 2))
        return torch.mean(p * sur_loss)


class GRU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super().__init__()
        self.rnn = nn.GRU(x_dim, h_dim, num_layers=hidden_units)
        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)
        self.z_dim = z_dim

    def forward(self, xs, h0, include_h=False):
        xs = self.drop(xs)
        xs, h = self.rnn(xs) if h0 is None else self.rnn(xs, h0)
        return (self.lin(xs), h) if include_h else self.lin(xs)


class Argus(nn.Module):
    def __init__(self, in_dim, edge_dim, h_dim, z_dim, device, s=5, pos_samples=784665,
                 aggregate=False):
        super().__init__()

        self.c1 = GCNConv(in_dim, h_dim, add_self_loops=True).to(device)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True).to(device)
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        self.c3 = GCNConv(h_dim, h_dim, add_self_loops=True).to(device)
        nn4 = nn.Sequential(nn.Linear(edge_dim, 8, device=device), nn.ReLU(),
                            nn.Linear(8, h_dim * h_dim, device=device))
        self.c4 = NNConv(h_dim, h_dim, nn4, aggr='mean').to(device)
        self.rnn = GRU(h_dim, h_dim, z_dim).to(device)
        self.decode_mlp = nn.Sequential(nn.Linear(z_dim, z_dim, device=device), nn.Softmax(dim=1))

        self.device = device
        self.ap_loss = APLoss(pos_len=pos_samples, margin=0.8, gamma=0.1,
                              surrogate_loss='squared', device=device)
        self.s = s
        self.aggregate = aggregate

    def forward(self, x, eis, eas, idxs, ptrs, h0=None, include_h=False):
        '''
        h0/include_h let the GRU state be threaded across calls, which is what makes it possible
        to embed the test window as a continuation of the training window instead of restarting
        the recurrence (and instead of scoring test edges against training-time embeddings).
        '''
        eis = [ei.to(self.device) for ei in eis]
        eas = [ea.to(self.device) for ea in eas]
        zs = []

        for i in range(len(eis)):
            ei, ea = eis[i], eas[i]
            ei_self_loops = add_remaining_self_loops(ei)[0]

            z = self.c1(x.to(self.device), ei_self_loops)
            z = self.c2(z, ei_self_loops)
            z = self.relu(z)
            z = self.drop(z)
            z = self.c3(z, ei_self_loops)
            z = self.relu(z)
            z = self.drop(z)
            z = self.c4(z, ei, ea)
            z = self.ac(z)
            zs.append(z)

        # time-major: (T, N, h), so the GRU scans time and not the node axis
        out, h = self.rnn(torch.stack(zs, dim=0), h0, include_h=True)

        if self.aggregate:
            out = torch.stack([self.sample_z(out[t], idxs[t], ptrs[t]) for t in range(out.size(0))])

        return (out, h) if include_h else out

    def sample_z(self, z, idx, ptr):
        z_agg = []
        for i in range(z.size(0)):
            n_neighbors = idx[i + 1] - idx[i]
            if n_neighbors:
                neighbors = ptr[idx[i] + torch.ones(n_neighbors).multinomial(self.s, replacement=True)]
                z_agg.append((z[neighbors].sum(dim=0) + z[i]) / (self.s + 1))
            else:
                z_agg.append(z[i])
        return self.decode_mlp(torch.stack(z_agg))

    def decode(self, ei, z):
        return (z[ei[0]] * z[ei[1]]).sum(dim=1)

    def calc_loss_argus(self, zs, eis):
        '''
        Observed edges are the 0 class and random pairs the 1 class, so the decoder output IS an
        anomaly score and nothing is negated at test time. Argus's released code uses the
        opposite labelling plus `scores = 1 - scores`; the two are equivalent.
        '''
        tot_loss = torch.zeros(1).to(self.device)
        for i in range(len(zs)):
            ps = eis[i]
            if ps.size(1) == 0:
                continue
            ns = torch.randint(0, zs.size(1), ps.size())
            t_index = torch.arange(0, ps.size(1), dtype=torch.int64, device=self.device).detach()
            neg_pred = self.decode(ps, zs[i])
            pos_pred = self.decode(ns, zs[i])
            tot_loss += self.ap_loss(
                torch.cat((pos_pred, neg_pred), 0),
                torch.cat((torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))), 0)
                    .to(self.device).detach(),
                t_index)
        return tot_loss.true_divide(len(zs))

    def validate(self, zs, eis):
        pos, neg = [], []
        for i in range(min(len(zs), len(eis))):
            ps = eis[i]
            if ps.size(1) == 0:
                continue
            ns = torch.randint(0, zs.size(1), ps.size())
            pos.append(self.decode(ps, zs[i]))
            neg.append(self.decode(ns, zs[i]))
        return torch.cat(pos), torch.cat(neg)


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

    eis, eas, idxs, ptrs, cnts, y = [], [], [], [], [], []
    for t in ts:
        mask = ts_g == t
        eis.append(ei[:, mask])
        eas.append(raw_edge_attr[mask])
        cnts.append(cnt[mask])

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
                idxs=idxs, ptrs=ptrs, cnts=cnts)


def chunked_forward(model, x, split, chunk, h0=None):
    '''Embed a split's snapshots in chunks of `chunk`, threading the GRU state. Returns (zs, h).'''
    zs, h = [], h0
    n = len(split.edge_index)
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        out, h = model.forward(x, split.edge_index[st:en], split.eas[st:en],
                               split.idxs[st:en] if split.idxs else [],
                               split.ptrs[st:en] if split.ptrs else [],
                               h0=h, include_h=True)
        zs.append(out)
    return torch.cat(zs, dim=0), h


def load_splits(args):
    '''
    Snapshot cache, safe to call from several processes at once.

    Without the lock, five jobs launched together all find the cache missing, all spend the same
    ten minutes bucketing edges, and all write the same three files concurrently -- torch.save is
    not atomic, so a reader can get a truncated archive and the failure looks like data
    corruption. One process builds; the rest block, then find the files and load them.

    Writes go to a per-pid temporary and are moved into place with os.replace, which is atomic
    within a filesystem. So even a job killed mid-build leaves no half-written cache.
    '''
    paths = {n: f'{args.cache}/{args.dataset}_{n}.pt' for n in ('tr', 'va', 'te')}

    def have_all():
        return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths.values())

    if not have_all():
        with FileLock(f'{args.cache}/{args.dataset}_snapshots'):
            if not have_all():      # re-check: another job may have built it while we waited
                print('building snapshots', flush=True)
                splits = {}
                for name in ('tr', 'va', 'te'):
                    g = torch.load(f'../data/{args.dataset}_tgraph_{name}.pt', weights_only=False)
                    g.ts //= args.ts_div
                    splits[name] = g

                ts_tr = splits['tr'].ts.unique()
                built = {
                    'tr': to_snapshots(splits['tr'], ts_tr, add_csr=args.aggregate),
                    'va': to_snapshots(splits['va'], ts_tr),        # holdout of the SAME window
                    'te': to_snapshots(splits['te'], splits['te'].ts.unique()),   # own axis, item 1
                }
                for name, obj in built.items():
                    tmp = f'{paths[name]}.tmp{os.getpid()}'
                    torch.save(obj, tmp)
                    os.replace(tmp, paths[name])
                print(f'wrote {", ".join(paths.values())}', flush=True)
                return built['tr'], built['va'], built['te']

    return tuple(torch.load(paths[n], weights_only=False) for n in ('tr', 'va', 'te'))


def train(tr, va, te, args):
    pos_samples = sum(tr.edge_index[i].size(1) for i in range(len(tr.edge_index)))
    model = Argus(tr.x.size(0), tr.eas[0].size(1), args.h_dim, args.z_dim, args.device,
                  s=args.s, pos_samples=pos_samples, aggregate=args.aggregate)
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
            with open('argus_speedtest.csv', 'a') as f:
                f.write(f'{args.dataset},{fwd_time},{loss_time},{bwd_time},{step_time}\n')
            return None

        with torch.no_grad():
            model.eval()
            # training window, keeping the GRU state at the boundary
            zs_tr, h = chunked_forward(model, tr.x, tr, args.chunk)

            pos, neg = model.validate(zs_tr, va.edge_index)
            labels = torch.zeros(pos.size(0) + neg.size(0))
            labels[pos.size(0):] = 1            # observed 0, random 1: matches calc_loss_argus
            preds = torch.cat([pos, neg]).cpu().numpy()
            va_auc = auc_score(labels.numpy(), preds)
            va_ap = ap_score(labels.numpy(), preds)
            print(f'\tVal AUC: {va_auc:0.4f}, AP: {va_ap:0.4f}')

            # TEST WINDOW: embed the test snapshots themselves, continuing the recurrence.
            # contemporaneous = Argus's own protocol (an edge is in the graph that embeds it);
            # predictive scores snapshot t against the embedding from t-1.
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

            preds = torch.sigmoid(torch.cat(preds)).cpu().numpy()
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
            if auc > best_snooped[0]:
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
    ap.add_argument('--dataset', default='optc-argus',
                    help="optc-argus is Argus's own split and labelling (default); optc-ts is ours")
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--runs', type=int, default=1,
                    help='replications IN THIS PROCESS. Default 1: fan out across GPUs instead, '
                         'one job per run, and let the shared CSV collect them.')
    ap.add_argument('--run-id', default=os.environ.get('LSB_JOBINDEX') or '0',
                    help='identifies this replication in the shared CSV. Defaults to '
                         'LSB_JOBINDEX, so `bsub -J "argus[1-5]"` needs nothing else. MUST be '
                         'distinct per job or the rows are not independent replications.')
    ap.add_argument('--seed', type=int, default=None,
                    help='default: derived from --run-id, so parallel jobs differ and each is '
                         'reproducible on its own')
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--patience', type=int, default=100)
    ap.add_argument('--bs', type=int, default=256, help='snapshots per optimizer step')
    ap.add_argument('--chunk', type=int, default=256, help='snapshots per forward pass at eval')
    ap.add_argument('--lr', type=float, default=0.01, help='Argus: 0.01')
    ap.add_argument('--h-dim', type=int, default=32, help='Argus: 32 (was 128 here)')
    ap.add_argument('--z-dim', type=int, default=16, help='Argus: 16 (was 64 here)')
    ap.add_argument('--s', type=int, default=5)
    ap.add_argument('--aggregate', action='store_true',
                    help="restore the neighbour aggregation Argus's code never calls")
    ap.add_argument('--score-mode', choices=['contemporaneous', 'predictive'],
                    default='contemporaneous',
                    help="contemporaneous is Argus's protocol: the scored edge is in the graph "
                         "that embeds it. predictive scores snapshot t against t-1.")
    ap.add_argument('--ts-div', type=int, default=360,
                    help='seconds per snapshot bucket. NOTE euler_test_optc.py uses 3600 for the '
                         'same data; pick one and use it for both or the baselines differ.')
    ap.add_argument('--speedtest', action='store_true')
    ap.add_argument('--cache', default='tmp')
    ap.add_argument('--tag', default='')
    ap.add_argument('--results', default=None,
                    help='shared CSV every run appends to. '
                         'Default argus_results_{dataset}{tag}.csv')
    ap.add_argument('--per-run-files', action='store_true',
                    help='write <results>.d/run_<id>.csv instead of appending to one file. Use '
                         'this if the results directory is on NFSv3, where flock can no-op; '
                         '--summarize reads either layout.')
    ap.add_argument('--prepare', action='store_true',
                    help='build the snapshot cache and exit. Run this once before fanning out.')
    ap.add_argument('--summarize', action='store_true',
                    help='print mean/sem over the rows already on disk and exit')
    args = ap.parse_args()

    results_csv = args.results or f'argus_results_{args.dataset}{args.tag}.csv'

    if args.summarize:
        summarize(results_csv)
        raise SystemExit

    os.makedirs(args.cache, exist_ok=True)
    tr, va, te = load_splits(args)

    if args.prepare:
        print(f'{len(tr.edge_index)} train / {len(va.edge_index)} val / {len(te.edge_index)} test '
              f'snapshots. Cache ready; fan out now.')
        raise SystemExit

    for i in range(args.runs):
        run_id = args.run_id if args.runs == 1 else f'{args.run_id}.{i}'
        # zlib.crc32, not hash(): str hashing is salted per process (PYTHONHASHSEED), so
        # hash(run_id) would hand the same run a different seed on every invocation.
        seed = args.seed if args.seed is not None else zlib.crc32(run_id.encode()) % 100_000
        seed_everything(seed)
        print(f'=== run {run_id} (seed {seed}) on {socket.gethostname()} ===', flush=True)

        t0 = time.time()
        res = train(tr, va, te, args)
        if not res:                     # --speedtest
            continue

        res.update(run_id=run_id, seed=seed, dataset=args.dataset, score_mode=args.score_mode,
                   h_dim=args.h_dim, z_dim=args.z_dim, lr=args.lr, s=args.s,
                   aggregate=int(args.aggregate), host=socket.gethostname(),
                   finished=datetime.now().isoformat(timespec='seconds'),
                   minutes=round((time.time() - t0) / 60, 1))

        # Appended as soon as the run ends, not after all --runs finish, so a job killed partway
        # through keeps whatever it already completed.
        where = append_row(results_csv, res, per_run_files=args.per_run_files)
        print(f'appended run {run_id} to {where}', flush=True)

    summarize(results_csv)