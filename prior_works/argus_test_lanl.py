from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import sys
import time

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score


# The model, APLoss, the GRU wrapper and SOAP all live in argus.py, so every Argus script
# shares one definition and a fix lands everywhere at once.
from argus import Argus, SOAP

sys.path.insert(0, '..')
from metrics import detection_metrics  # noqa: E402  -- same module the CyberGFM runs use

def get_args():
    ap = ArgumentParser(description='Argus on the random (non-temporal) LANL split. Defaults '
                                    'reproduce the released code (C0ldstudy/Argus @ 0bd70e6).')
    # runtime
    # runtime
    ap.add_argument('--device', type=int, default=0, help='CUDA index, or -1 for CPU')
    ap.add_argument('--runs', type=int, default=10, help='independent training runs to average')
    ap.add_argument('--epochs', type=int, default=1000, help='their main.py: 100')
    ap.add_argument('--patience', type=int, default=100, help='their main.py: 3')
    ap.add_argument('--threads', type=int, default=64, help='torch.set_num_threads; 0 leaves it')
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

    ap.add_argument('--bs', type=int, default=0,
                    help='snapshots per forward pass; 0 = the whole window, as they use. '
                         'Gradients accumulate and ONE step is taken per epoch whatever this is, '
                         'so it trades memory against BPTT length, not the optimizer regime')

    # model -- their main.py uses -H/--hidden 32 and -z/--zdim 16
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
    # snapshot_ts: the hour each snapshot covers, kept so fp_day can be normalised by the span of
    # test traffic. to_snapshots emits one entry per t in `ts`, so index i <-> snapshot_ts[i].
    kept = torch.as_tensor([int(t) for t in ts if not (last_ts and t > last_ts)])
    return Data(x=x, edge_index=eis, label=y, eas=eas, idxs=idxs, ptrs=ptrs, snapshot_ts=kept)

def probe_memory(tr, args):
    """
    Snapshot sizes and the largest --bs that fits, computed over actual contiguous windows.

    APLoss.forward builds mat_data = y_pred.repeat(n_pos, 1), an n_pos x 2*n_pos float32 matrix.
    Three of that shape are saved for backward per snapshot -- pow saves (margin - t), and
    p * sur_loss saves both p and sur_loss -- and about six exist at the peak of one call. Memory
    is therefore QUADRATIC in a snapshot's edge count, and since consecutive hours correlate
    (business hours cluster), a contiguous chunk can be entirely above median. So the limit is the
    worst contiguous window, not the average, which is what this scans for.
    """
    n = torch.tensor([ei.size(1) for ei in tr.edge_index], dtype=torch.float)
    qs = torch.quantile(n, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
    lo, med, p90, p99, hi = [float(q) for q in qs]
    mat = lambda e: e * (2 * e) * 4 / 2 ** 30
    RETAINED, PEAK = 3, 6

    print(f'{len(n)} training snapshots, {int(n.sum()):,} edges')
    print(f'edges/snapshot: min {lo:.0f}  median {med:.0f}  p90 {p90:.0f}  p99 {p99:.0f}  max {hi:.0f}')
    print(f'APLoss matrix: median {mat(med):.2f} GiB, max {mat(hi):.2f} GiB '
          f'({RETAINED} retained per snapshot, ~{PEAK} live during a call)')

    sizes = torch.tensor([mat(float(e)) for e in n])
    cum = torch.cat([torch.zeros(1), sizes.cumsum(0)])
    worst_window = lambda b: float((cum[b:] - cum[:-b]).max())
    need = lambda b: RETAINED * worst_window(b) + (PEAK - RETAINED) * float(sizes.max())

    total = (torch.cuda.mem_get_info(args.device)[1] / 2 ** 30
             if torch.cuda.is_available() and args.device != 'cpu' else 0.0)
    if not total:
        print('\nCPU: no device budget to check against. Full window needs about '
              f'{RETAINED * float(cum[-1]):.0f} GiB of RAM for APLoss alone.')
        return

    eye = tr.x.size(0) ** 2 * 4 / 2 ** 30
    budget = total - eye - 1.5
    print(f'\ndevice {total:.1f} GiB - {eye:.1f} GiB for torch.eye({tr.x.size(0)}) - 1.5 GiB '
          f'overhead = {budget:.1f} GiB usable\n')
    print(f'{"--bs":>5}  {"worst window":>14}  {"peak need":>10}  fits?')
    for b in (1, 2, 4, 6, 8, 12, 16, 24, 32, len(n)):
        if b > len(n):
            break
        print(f'{b:>5}  {worst_window(b):>11.1f} GiB  {need(b):>7.1f} GiB  '
              f'{"yes" if need(b) <= budget else "NO"}')

    safe = [b for b in range(1, len(n) + 1) if need(b) <= budget]
    safe = max(safe) if safe else 0
    print(f'\nlargest safe --bs: {safe}')
    print(f'full window (--bs 0) would need {RETAINED * float(cum[-1]):.0f} GiB for APLoss alone, '
          'so it is a CPU-only configuration')
    if safe:
        print(f'\nRun:  python {os.path.basename(__file__)} --bs {safe}')
        print(f'One optimizer step per epoch is preserved by gradient accumulation; only BPTT '
              f'truncates, to {safe} snapshots instead of {len(n)}.')


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
                  gru_layers=args.gru_layers, beta=args.beta, aggregate=args.aggregate,
                  neg_sampling=args.neg_sampling, neg_filter=not args.no_neg_filter,
                  max_pos=args.max_pos)
    opt = SOAP(model.parameters(), lr=args.lr, mode='adam', weight_decay=0.0)

    best = (0, 0, 0, {})
    best_cheating = (0,0)
    BS = args.bs if args.bs > 0 else len(tr.edge_index)
    no_progress = 0
    n_snaps = len(tr.edge_index)
    for e in range(args.epochs):
        fwd_time=bwd_time=loss_time=step_time = 0
        model.train()
        opt.zero_grad()
        h = None
        # ONE optimizer step per epoch whatever BS is: their classification.py sets BS to the
        # whole window and steps once, so gradients accumulate across BS-sized pieces, each
        # weighted by its share of the window, and the GRU state is threaded between them.
        # Lowering BS for memory then costs only BPTT length, not the optimizer regime.
        # (The old `range(len // BS)` silently dropped the tail whenever BS did not divide the
        # window evenly -- with BS=24 over 352 snapshots it trained on 336 and ignored 16.)
        for st_i in range(0, n_snaps, BS):
            en_i = min(st_i + BS, n_snaps)
            w = (en_i - st_i) / n_snaps

            st = time.time()
            print("Fwd", end='', flush=True)
            zs, h = model.forward(tr.x, tr.edge_index[st_i:en_i], tr.eas[st_i:en_i], tr.idxs[st_i:en_i], tr.ptrs[st_i:en_i], h0=h, include_h=True)
            fwd_time += time.time() - st
            print(f' ({((fwd_time) / 60):0.2f} mins)')

            st = time.time()
            print("Loss", end='', flush=True)
            loss = model.calc_loss_argus(zs, tr.edge_index[st_i:en_i])
            loss_time += time.time() - st
            print(f' ({((time.time() - st) / 60):0.2f} mins)')

            st = time.time()
            print("Bwd", end='', flush=True)
            (loss * w).backward()
            bwd_time += time.time() - st
            print(f' ({((time.time() - st) / 60):0.2f} mins)')

            h = h.detach()   # state crosses the boundary, gradient does not

        st = time.time()
        opt.step()
        step_time += time.time() - st
        print(f'[{e}] Loss: {loss.item():0.4f}')

        if args.speedtest:
            with open(f'argus_speedtest{args.tag}.csv', 'a') as f:
                f.write(f'LANL,{fwd_time},{loss_time},{bwd_time},{step_time}\n')
            exit()


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

            preds = torch.cat([pos,neg]).numpy()
            labels = labels.numpy()

            # sklearn rather than fast_auc/fast_ap: the latter agree exactly when no scores tie,
            # but diverge by up to ~20% (in either direction) when they do, because they step
            # through the sorted array element-by-element instead of by distinct threshold.
            va_auc = auc_score(labels, preds)
            va_ap = ap_score(labels, preds)

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

            auc = auc_score(y, preds)
            ap = ap_score(y, preds)

            # AP lift, precision and FP/day at 50% recall, via the same metrics.py the CyberGFM
            # runs use, so the rows are directly comparable. Per-edge timestamps are rebuilt from
            # the snapshot hours; test_days() only needs their span.
            n_per = torch.as_tensor([ei.size(1) for ei in te.edge_index])
            te_ts = (te.snapshot_ts.repeat_interleave(n_per) * (60 * 60)).numpy()
            extra = detection_metrics(y, preds, ap, ts=te_ts, recall=0.5, prefix='')
            print(f'\tTe AUC: {auc:0.4f}, AP: {ap:0.4f}, lift: {extra["ap_lift"]:0.1f}x, '
                  f'P@r50: {extra["prec_r50"]:0.4f}, FP/day: {extra["fp_day_r50"]:0.0f}',
                  end='', flush=True)

            if va_auc+va_ap > best[0]:
                best = (va_auc+va_ap, auc, ap, extra)
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
    return {'auc': best[1], 'ap': best[2], **best[3],
            'auc_last': auc, 'ap_last': ap,
            'auc_snooped': best_cheating[0], 'ap_snooped': best_cheating[1]}

if __name__ == '__main__':
    args = get_args()
    args.device = args.device if args.device >= 0 else 'cpu'

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
        if args.seed is not None:
            torch.manual_seed(args.seed + r)   # per replication, so each is reproducible alone
        print(f'\n===== run {r} =====')
        best.append(train(tr, va, te, args))
        with open(f'argus_{args.dataset}{args.tag}_log.csv', 'a') as f:
            f.write(json.dumps(best[-1]) + '\n')

    df = pd.DataFrame(best)
    df.loc['mean'] = df.mean()
    df.loc['sem'] = df.sem()

    print(df)
    df.to_csv(f'argus_results_{args.dataset}{args.tag}.csv')