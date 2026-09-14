import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import BertConfig
import torch
from torch.optim import Adam
from sklearn.metrics import accuracy_score

from models.gnn_bert import RWBertFT, GNNEmbedding
from sampler import TRWSampler

DEVICE = 0

with open('data/unsw_node_map.pkl', 'rb') as f:
    nm = pickle.load(f)

id_to_node = {v:k for k,v in nm.items()}
labels = [id_to_node[i].split('.')[0] for i in range(1,50)]
codes, categories = pd.factorize(labels)

tr = torch.load('data/unsw_tgraph_csr.pt')
tr = TRWSampler(tr, device=DEVICE, walk_len=1, batch_size=512, edge_features=True)
params = SimpleNamespace(H=128, L=2, MINI_BS=1024)
config = BertConfig(
    228,
    hidden_size=         params.H,
    num_hidden_layers=   params.L,
    num_attention_heads= params.H // 64,
    intermediate_size=   params.H * 4,
    num_nodes = tr.num_tokens,
    max_position_embeddings = 512
)

categories = np.array(codes)
should_eval = (categories == 0 ) + (categories == 1) + (categories == 3)
x_idx = torch.arange(1,50)[should_eval]
y = categories[should_eval]
y[y==3] = 2
y = torch.tensor(y)

for _ in range(10):
    sd = torch.load('pretrained/static/unsw/rw_bert_unsw_tiny-best.pt')
    model = RWBertFT(config, sd, device=DEVICE, out_dim=3)
    model.fm.requires_grad = False

    tr_size = int(0.8*x_idx.size(0))
    tr_idx = torch.randperm(x_idx.size(0))[:tr_size]
    te_idx = torch.randperm(x_idx.size(0))[tr_size:]

    tr = x_idx[tr_idx].to(DEVICE)
    te = x_idx[te_idx].to(DEVICE)

    tr_y = y[tr_idx].to(DEVICE)
    te_y = y[te_idx].to(DEVICE)

    opt = Adam(model.parameters(), lr=0.001)
    for _ in range(100):
        loss = model.forward(
            tr[:,None],
            torch.ones((tr.size(0),1,128), dtype=torch.bool, device=DEVICE),
            torch.ones((tr.size(0),1,1), dtype=torch.bool, device=DEVICE),
            tr_y
        )
        loss.backward()
        opt.step()

        print(loss)

        preds = model.predict(
            te[:,None,None],
            torch.ones((tr.size(0),1,1), device=DEVICE),
            torch.ones((tr.size(0),1,1), device=DEVICE)
        ).argmax(dim=1)

        print(accuracy_score(te_y, preds.detach().cpu()))