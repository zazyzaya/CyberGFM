import torch
from torch_geometric.data import Data
from collections import defaultdict
from tqdm import tqdm

TS = 0
SRC = 1
DST = 2
LABEL = 3
# Unclear what the remaining columns are

# Using dataset linked in Argus repo
# https://drive.google.com/drive/folders/1pTU-ZcyJbzoB1FuvujXe-ynaUy8O-PVD?usp=sharing

DATASET = '/mnt/raid10/cyber_datasets/OpTC/argus_optc/auth_optc.txt'
N_LINES = 10109338 # wc -l auth_optc.txt
OPTC_FIRST_RED = 573290

def build_dataset():
    f = open(DATASET)

    nmap = dict()
    def get_or_add(n):
        if (nid := nmap.get(int(n))) is None:
            nid = len(nmap)
            nmap[int(n)] = nid
        return nid

    edges = defaultdict(list)

    prog = tqdm(total=N_LINES, desc='Reading file')
    line = f.readline()
    while line:
        ts,src,dst,label,_ = line.split(',', 4)

        src = get_or_add(src)
        dst = get_or_add(dst)
        ts = int(ts)
        label = int(label)

        # Bidirectional or all walks are 1 step..
        edges[src].append([dst,ts,label])

        if ts < OPTC_FIRST_RED:
            edges[dst].append([src,ts,label])

        prog.update()
        line = f.readline()

    f.close()
    prog.close()

    idx = [0]
    col = []
    ts = []
    labels = []

    for i in tqdm(range(len(nmap)), desc='torchifying'):
        edge = edges[i]

        if edge:
            d, t, l = zip(*edge)
            col.extend(d)
            ts.extend(t)
            labels.extend(l)
            num_edges = len(d)
        else:
            num_edges = 0

        idx.append(idx[-1] + num_edges)

    idx = torch.tensor(idx)
    deg = idx[1:] - idx[:-1]
    src = torch.arange(idx.size(0)-1)
    src = src.repeat_interleave(deg)

    g = Data(
        x = torch.zeros((len(nmap),1)),
        idxptr = idx,
        col = torch.tensor(col),
        src = src,
        edge_attr = torch.zeros((len(col),1)),
        ts = torch.tensor(ts),
        label = torch.tensor(labels)
    )

    print(g)
    torch.save(g, '../data/optc-argus_tgraph_csr.pt')

if __name__ == '__main__':
    build_dataset()