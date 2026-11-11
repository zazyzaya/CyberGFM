'''
Checks for the --undirected-walks experiment, and the numbers that say whether it can matter.

Per dataset, from the split files alone (no model, no walks -- fine on the login node):
  * the undirected walk graph is what it should be: every training edge once in each direction
    with its timestamp and features, rows sorted by time;
  * how far a directed walk gets: the share of training edges that lead to a node with no
    out-edges (a walk that takes one stops there), and the share of validation / test edges whose
    source has no out-edge in the training graph, i.e. edges scored with no context at all --
    against the same shares on the undirected graph.

With --walks it also samples context walks with the real walk kernels (run_undirected.sh --smoke
does this inside a job): every hop must be a training edge with that edge's features, the
directed sampler must never move against an edge, and the undirected one must give context to
nodes that have only in-edges.

    python check_undirected.py                                   # the four LANL splits
    python check_undirected.py --datasets lanl14argus-dirtyts8d
    python check_undirected.py --datasets lanl14argus --walks --device 0
'''
import argparse

import torch

from common import PAD, context, load_split, make_sampler, undirected_walk_graph, uses_edge_features

LANL = ['lanl14argus', 'lanl14argus-ts', 'lanl14argus-dirtyts', 'optc', 'optc-argus', 'unsw']


def rows(g):
    return torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])


def arc_hash(s, d, ea=None, ts=None):
    '''One int64 per (src, dst[, features][, ts]); wraps around, which is fine for comparing.'''
    h = s.long() * 1000003 + d.long() * 7919
    if ts is not None:
        h = h * 31 + ts.long()
    if ea is not None:
        for j in range(ea.size(1)):
            h = h * 1000033 + ea[:, j].long() + 1
    return h


def check_graph(name, tr, und, edge_features):
    n = tr.x.size(0)
    s, d = rows(tr), tr.col
    ea = tr.edge_attr if edge_features else None
    us, ud = rows(und), und.col
    uea = und.edge_attr if edge_features else None
    rev = s != d

    assert torch.equal(us, und.src), 'src does not match idxptr'
    want = torch.cat([arc_hash(s, d, ea, tr.ts),
                      arc_hash(d[rev], s[rev], ea[rev] if ea is not None else None, tr.ts[rev])])
    got = arc_hash(us, ud, uea, und.ts)
    assert torch.equal(want.sort().values, got.sort().values), 'arcs differ from edges + reversed edges'
    same_row = us[1:] == us[:-1]
    assert bool((und.ts[1:][same_row] >= und.ts[:-1][same_row]).all()), 'a row is not sorted by time'
    print(f'  walk graph ok: {s.numel():,} edges -> {us.numel():,} arcs '
          f'({int((~rev).sum()):,} self-loops kept once), rows sorted by time')

    deg = tr.idxptr[1:] - tr.idxptr[:-1]
    udeg = und.idxptr[1:] - und.idxptr[:-1]
    print(f'  nodes: {n:,} in the vocabulary, {int((udeg > 0).sum()):,} in the training graph, '
          f'{int((deg > 0).sum()):,} with an out-edge ({int(((udeg > 0) & (deg == 0)).sum()):,} have in-edges only)')
    print(f'  training edges that lead to a node with no out-edge:  directed {(deg[d] == 0).float().mean():.1%}'
          f'   undirected {(udeg[d] == 0).float().mean():.1%}')
    return deg, udeg


def check_context(name, deg, udeg):
    for split in ('va', 'te'):
        g = load_split(name, split)
        src = g.src.long() if 'src' in g.keys() else rows(g)
        no_d, no_u = deg[src] == 0, udeg[src] == 0
        line = (f'  {split} edges whose source has no context:  directed {no_d.float().mean():.1%}'
                f'   undirected {no_u.float().mean():.1%}')
        if split == 'te' and 'label' in g.keys():
            mal = g.label.bool()
            if mal.any():
                line += (f'   | attack edges ({int(mal.sum())}): directed {no_d[mal].float().mean():.1%}'
                         f'   undirected {no_u[mal].float().mean():.1%}')
        print(line)


def check_walks(name, tr_data, edge_features, device, walk_len=4, n=4096):
    F = tr_data.edge_attr.size(1) if edge_features else 0
    num_nodes = tr_data.x.size(0)
    s, d = rows(tr_data), tr_data.col
    ea = tr_data.edge_attr if edge_features else None
    fwd = arc_hash(s, d, ea).unique().to(device)       # (u, v, features) of real edges

    deg = tr_data.idxptr[1:] - tr_data.idxptr[:-1]
    indeg = torch.bincount(d, minlength=num_nodes)
    sinks = ((deg == 0) & (indeg > 0)).nonzero().squeeze(-1)
    sinks = sinks[torch.randperm(sinks.numel())[:n]]
    busy = (deg > 0).nonzero().squeeze(-1)
    busy = busy[torch.randperm(busy.numel())[:n]]

    for temporal in (False, True):
        for undirected in (False, True):
            smp = make_sampler(tr_data, temporal, edge_features, walk_len, n, device,
                               undirected_walks=undirected)
            against = hops = 0
            for kind, start in (('busy', busy), ('sinks', sinks)):
                if start.numel() == 0:
                    continue
                start = start.to(device)
                ts = torch.full_like(start, int(tr_data.ts.max()))
                rw = context(smp, start, ts, delta=10 ** 12, use_walk=True)
                assert rw.size(1) == (walk_len + 1) + walk_len * F, f'unexpected context width {rw.size(1)}'
                assert torch.equal(rw[:, -1], start), 'context does not end at the source'
                nodes = rw[:, ::1 + F]                                   # left-to-right; source last
                for i in range(walk_len):
                    a, b = nodes[:, i], nodes[:, i + 1]                  # the walk stepped b -> a
                    real = a != PAD
                    if not real.any():
                        continue
                    assert bool((b[real] != PAD).all()), 'a hop starts from PAD'
                    f = None
                    if F:
                        f = rw[:, i * (1 + F) + 1:(i + 1) * (1 + F)][real] - num_nodes
                        assert bool((f >= 0).all()), 'PAD features on a real hop'
                    along = torch.isin(arc_hash(b[real], a[real], f), fwd)
                    back = torch.isin(arc_hash(a[real], b[real], f), fwd)
                    assert bool((along | back).all()), 'a hop is not a training edge with its features'
                    against += int((~along).sum())
                    hops += int(real.sum())
                if kind == 'sinks':
                    got = (nodes[:, :-1] != PAD).any(1).float().mean().item()
                    if undirected:
                        assert got == 1.0, f'only {got:.1%} of in-edge-only nodes got context'
                    else:
                        assert got == 0.0, 'a node with no out-edges got directed context'
            if not undirected:
                assert against == 0, 'the directed sampler moved against an edge'
            else:
                assert against > 0, 'the undirected sampler never moved against an edge'
            print(f'  walks ok: {"temporal" if temporal else "static":<8} '
                  f'{"undirected" if undirected else "directed":<10} {hops:,} hops checked, '
                  f'{against / max(1, hops):.1%} against the edge direction')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--datasets', nargs='*', default=LANL)
    ap.add_argument('--walks', action='store_true', help='also sample walks (do this inside a job)')
    ap.add_argument('--device', type=int, default=0, help='for --walks; -1 for CPU')
    args = ap.parse_args()
    torch.manual_seed(0)

    for name in args.datasets:
        print(name)
        edge_features = uses_edge_features(name)
        tr = load_split(name, 'tr')
        und = undirected_walk_graph(tr)
        deg, udeg = check_graph(name, tr, und, edge_features)
        check_context(name, deg, udeg)
        if args.walks:
            check_walks(name, tr, edge_features, args.device if args.device >= 0 else 'cpu')
    print('all checks passed')