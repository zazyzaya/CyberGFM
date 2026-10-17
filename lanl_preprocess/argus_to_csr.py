from collections import defaultdict
from math import log10

from joblib import Parallel, delayed
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from load_utils import id_edge_a
from load_lanl import load_partial_lanl, TIMES

# Same config as reported in paper
END = TIMES['test']
DELTA = 60**2

def build_csr():
    '''Builds the tokenized and the raw file in one pass; both share the same edge order.'''
    t = 0
    jobs = []
    ea_map = dict()

    while t < END:
        jobs.append((t,t+DELTA))
        t += DELTA

    def get_or_add(e):
        if (eid := ea_map.get(e)) is None:
            eid = len(ea_map)
            ea_map[e] = eid
        return eid

    [get_or_add(i) for i in range(100)]

    def ea_to_token(ef):
        if ef > 1:
            ef = int(ef)
            #
            if ef < 100:
                return get_or_add(ef)
            else:
                return get_or_add(f'10^{int(log10(ef))}')
            #
        else:
            ef = int(ef*100)
            return get_or_add(f'0.{ef}')

    def fmt_ea_row(row):
        return [ea_to_token(ea) for ea in row]

    num_nodes = 0
    # dst, tokenized attr, ts, mal idx, edge weight, raw attr
    csr = defaultdict(lambda : [[],[],[],[],[],[]])
    test = 0
    for j in tqdm(jobs, desc='Reading raw'):
        tgraph = load_partial_lanl(start=j[0], end=j[1], delta=DELTA, is_test=True, use_flows=True, ea_fn=id_edge_a)

        if not num_nodes:
            num_nodes = tgraph.num_nodes

        # Took 25 mins of runtime to hit this error and realize
        # this check was needed :)
        if len(tgraph.eis) == 0:
            continue

        ei = tgraph.eis[0]
        ea = tgraph.eas[0]
        ew = tgraph.ews[0]
        y = tgraph.ys[0]

        for i in range(ei.size(1)):
            src,dst = ei[:,i]
            src = src.item()
            dst = dst.item()

            csr[src][0].append(dst)
            # edge weight goes on the end as one more token column
            csr[src][1].append(fmt_ea_row(ea[:,i]) + [ea_to_token(ew[i].item())])
            csr[src][5].append(ea[:,i].tolist())
            csr[src][2].append(j[0])
            csr[src][4].append(ew[i].item())

            if y[i]:
                csr[src][3].append(len(csr[src][0])-1)

        test += 1
        #if test >= 10:
        #    break

    idxptr = [0]
    col = []
    attr = []
    raw_attr = []
    label = []
    ts = []
    ew = []

    for i in tqdm(range(num_nodes), desc='Building torch'):
        neigh,ef,t,y,w,raw_ef = csr[i]
        col += neigh
        attr += ef
        raw_attr += raw_ef
        ts += t
        ew += w

        if y:
            label += [y_ + idxptr[-1] for y_ in y]

        idxptr.append(idxptr[-1] + len(neigh))

    idxptr = torch.tensor(idxptr)
    col = torch.tensor(col)
    ew = torch.tensor(ew)
    label = torch.tensor(label)
    ts = torch.tensor(ts)

    for suffix, ea in (('', attr), ('_raw', raw_attr)):
        torch.save(
            Data(
                x = torch.zeros((num_nodes,1)),
                idxptr = idxptr,
                col = col,
                edge_attr = torch.tensor(ea),
                edge_weight = ew,
                is_mal = label,
                ts = ts
            ),
            f'../data/lanl14argus_tgraph{suffix}_csr.pt'
        )

build_csr()