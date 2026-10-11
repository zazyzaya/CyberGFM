'''
Argus (Xu, Shu & Li, IEEE S&P 2024) -- the model, in one place.

Every Argus experiment imports from here, so a fix lands in all of them at once. Previously
argus_test_lanl.py, argus_test_lanl_ts.py and argus_test_optc.py each carried their own copy of
APLoss, the GRU wrapper and the Argus module, and they had drifted: the OpTC copy was still
training with inverted loss polarity and an unbounded decoder long after both were fixed in the
LANL files. That is the whole reason this module exists.

  Paper: "Understanding and Bridging the Gap Between Unsupervised Network Representation Learning
         and Security Analytics", IEEE S&P 2024
         <https://c0ldstudy.github.io/commons/papers/SP2024_paper118.pdf>
  Code:  github.com/C0ldstudy/Argus @ 0bd70e6

=================================================================================================
SCORE CONVENTION -- read this before using decode() anywhere
=================================================================================================

`decode()` returns a RECONSTRUCTION PROBABILITY in (0, 1). High means "this edge looks like it
belongs in the graph". It is NOT an anomaly score.

  * In the loss, OBSERVED edges are class 1 and random pairs are class 0.
  * Detection therefore alarms on LOW values -- the paper "sets off alarms for the ones under
    tau". Every evaluation must NEGATE: `anomaly_score = -decode(...)`, or equivalently
    `1 - decode(...)`. AUC and AP are identical under either, both being monotone decreasing.
  * A link-prediction metric (real edges vs. random pairs, used for model selection) uses the
    RAW probability with observed edges as the positive class. That is the one place the score
    is not negated, and it is not an anomaly metric.

This matches Argus's released code, which labels observed edges 1 and applies `scores = 1-scores`
before scoring. Flipping the labels in the loss and dropping the negation is NOT an equivalent
reformulation: APLoss is an average-precision surrogate, and AP is asymmetric in the two classes
-- its u_all/u_pos buffers and margin are defined over the positive class. Measured on LANL,
running it inverted cost roughly 50x in AP (~0.002 vs ~0.11). See
claude/argus-replication-notes.md section 1.

=================================================================================================
THE SIX FIXES CARRIED HERE, relative to a naive transcription of their repo
=================================================================================================

1. LOSS POLARITY. Observed edges are the positive class (eq. 13). An earlier reimplementation had
   the two blocks swapped, so APLoss maximised AP for the random pairs.

2. TIME-MAJOR GRU. Per-snapshot embeddings stack with `dim=0` -> (T, N, h). nn.GRU leaves
   batch_first False, so it reads dim 0 as the sequence; stacking `dim=1` made it scan the NODE
   axis, whose order is arbitrary, and treat each snapshot as an independent batch element.
   Nothing crossed time at all.

3. AGGREGATION OFF BY DEFAULT, and indexed by t when on. Their repo defines the neighbour
   aggregation (`inner_forward_sm`) and never calls it; `forward_once` returns `self.ac(x)`.
   The loop that applied it also read `idxs[i]` with `i` left over from the embedding loop, so
   every timestep aggregated over the LAST snapshot's neighbour lists.

4. SIGMOID DECODER. `sigmoid((z[src] * z[dst]).sum(1))`, per their Euler_Embed_Unit.decode.
   Without the sigmoid the scores are unbounded and APLoss's margin of 0.8 means something
   entirely different; with a softmax instead, the margin is essentially unreachable.

5. DIMENSIONS 32/16. Their main.py uses -H 32 -z 16 with worker_args = [hidden, hidden], so c4 is
   h_dim -> h_dim and the reduction to z_dim happens in the GRU's output projection. An earlier
   copy built 128/64, where the NNConv per-edge weight matrix is 32x larger.

6. TRUE-EDGE REJECTION in negative sampling, as their fast_negative_sampling does. The collision
   rate is small on sparse graphs, so this is fidelity rather than a fix -- but it is not
   negligible on the small dense ones (OpTC has ~1k nodes).

=================================================================================================
PAPER vs. CODE, for the hyperparameters the scripts set
=================================================================================================

Their "ARGUS Hyper-parameters" paragraph says: a TWO-LAYER GRU, lr 0.01, node embedding size 16,
margin m = 0.8, and -- per dataset, "based on their data distributions" --

    beta  = 0.01 (LANL)   0.5 (OpTC)        the weight on L_DEC
    gamma = 0.01 (LANL)   0.1 (OpTC)        APLoss's moving-average factor

Three of those do not match the released code, so each script states which it follows:

  gru_layers  paper 2; their recurrent.GRU leaves hidden_units=1 and main.py never overrides it.
              DEFAULT HERE IS 2, following the paper. `--gru-layers 1` reproduces their code.
  beta        the paper's loss is L_ap + beta*L_DEC, but THEIR CODE HAS NO L_DEC TERM ANYWHERE.
              So a nonzero beta cannot reproduce their code -- it runs this repo's per-edge BCE
              reading of eq. 12, which is a GUESS at "cross-entropy on the sampled edges".
              `--beta 0` is their code. Note also that eq. 12 is written over the eq. 11
              aggregated decoder, so beta>0 with --aggregate off applies L_DEC to a different
              decoder than the paper describes.
  gamma       the per-dataset split is in the paper and also in their code, where LANL hardcodes
              0.01 and the OpTC/UNSW scripts hardcode 0.1. These agree; each script sets its own.

UNSW appears in neither, because Argus never evaluated it; argus_test_unsw.py documents what it
picked and why.
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.utils import add_remaining_self_loops

# Their optimizer. argus_opt is the vendored copy; libauc is the upstream package.
try:
    from argus_opt import SOAP
except ImportError:                                  # pragma: no cover
    from libauc.optimizers import SOAP

__all__ = ['APLoss', 'GRU', 'Argus', 'ArgusTS', 'SOAP',
           'squared_loss', 'squared_hinge_loss', 'logistic_loss']


# -----------------------------------------------------------------------------------------------
# APLoss, transcribed from libauc with the debug scaffolding removed. The arithmetic is unchanged.
# -----------------------------------------------------------------------------------------------
def squared_loss(margin, t): return (margin - t) ** 2
def squared_hinge_loss(margin, t): return torch.max(margin - t, torch.zeros_like(t)) ** 2
def logistic_loss(margin, t): return torch.log(1 + torch.log(-margin * t))


def _check_tensor_shape(inputs, shape=(-1, 1)):
    if len(inputs.shape) != len(shape):
        inputs = inputs.reshape(shape)
    return inputs


def _get_surrogate_loss(backend='squared_hinge'):
    try:
        return {'squared_hinge': squared_hinge_loss,
                'squared': squared_loss,
                'logistic': logistic_loss}[backend]
    except KeyError:
        raise ValueError(f'unknown surrogate_loss {backend!r}') from None


class APLoss(nn.Module):
    '''
    Stochastic AP maximization (Qi et al., NeurIPS 2021), as vendored in libauc and used by Argus.

    pos_len sizes the u_all/u_pos moving-average buffers, so it must be at least the number of
    positive samples the largest single call will index. Callers pass the total positive count
    over the training window.
    '''

    def __init__(self, pos_len=None, margin=1.0, gamma=0.99, surrogate_loss='squared_hinge',
                 device=None):
        super().__init__()
        # `is None`, not the original `if not device`: that treated device=0 (a valid CUDA index,
        # and what every caller here passes) as unset and silently re-derived it.
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.u_all = torch.zeros(pos_len, 1, device=self.device)
        self.u_pos = torch.zeros(pos_len, 1, device=self.device)
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def forward(self, y_pred, y_true, index_p):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))
        index_p = index_p[index_p >= 0]          # only positives index the buffers

        pos_mask = (y_true == 1).flatten()
        f_ps = y_pred[pos_mask]
        mat_data = y_pred.flatten().repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))
        pos_sur_loss = sur_loss * pos_mask

        self.u_all[index_p] = ((1 - self.gamma) * self.u_all[index_p]
                               + self.gamma * sur_loss.mean(1, keepdim=True).detach())
        self.u_pos[index_p] = ((1 - self.gamma) * self.u_pos[index_p]
                               + self.gamma * pos_sur_loss.mean(1, keepdim=True).detach())

        p = (self.u_pos[index_p] - self.u_all[index_p] * pos_mask) / (self.u_all[index_p] ** 2)
        return torch.mean(p * sur_loss)


# -----------------------------------------------------------------------------------------------
# Recurrence
# -----------------------------------------------------------------------------------------------
class GRU(nn.Module):
    '''
    nn.GRU plus dropout and an output projection to z_dim.

    batch_first is left at False, so the input must be (T, N, h) -- see fix 2. h0/include_h let
    the hidden state be threaded across calls, which is what makes it possible to embed a window
    in chunks, or to continue the recurrence across a train/test boundary, without restarting
    from zeros at every chunk edge.
    '''

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super().__init__()
        self.rnn = nn.GRU(x_dim, h_dim, num_layers=hidden_units)
        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)
        self.z_dim = z_dim

    def forward(self, xs, h0=None, include_h=False):
        xs = self.drop(xs)
        xs, h = self.rnn(xs) if h0 is None else self.rnn(xs, h0)
        return (self.lin(xs), h) if include_h else self.lin(xs)


# -----------------------------------------------------------------------------------------------
# The model
# -----------------------------------------------------------------------------------------------
class Argus(nn.Module):
    '''
    Three GCNConv layers, an edge-conditioned NNConv, and a GRU over snapshots.

    Defaults reproduce their released code. Every deviation is opt-in:

      aggregate=True   the neighbour aggregation of eq. 11, which their repo never calls
      beta>0           the L_DEC decoder cross-entropy of eq. 12, absent from their repo
      neg_sampling     'degree' is a DIAGNOSTIC, not their method -- it resamples endpoints from
                       the snapshot's own endpoint multiset so negatives carry the positives'
                       degree distribution. Uniform negatives are nearly separable by degree
                       alone, so a model can score well by becoming a degree prior; use 'degree'
                       to test whether that is what is happening, and never report it as Argus.
      neg_filter=False drops the true-edge rejection (fix 6)
      max_pos>0        caps positives per snapshot. APLoss is O(n_pos^2), so one busy hour can
                       OOM on its own whatever the batch size.
    '''

    def __init__(self, in_dim, edge_dim, h_dim=32, z_dim=16, device='cpu', s=5,
                 pos_samples=315163, margin=0.8, gamma=0.01, gru_layers=2, beta=0.0,
                 neg_sampling='uniform', neg_filter=True, aggregate=False, max_pos=0):
        super().__init__()

        self.c1 = GCNConv(in_dim, h_dim, add_self_loops=True).to(device)
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True).to(device)
        self.c3 = GCNConv(h_dim, h_dim, add_self_loops=True).to(device)
        nn4 = nn.Sequential(nn.Linear(edge_dim, 8, device=device), nn.ReLU(),
                            nn.Linear(8, h_dim * h_dim, device=device))
        self.c4 = NNConv(h_dim, h_dim, nn4, aggr='mean').to(device)
        self.rnn = GRU(h_dim, h_dim, z_dim, hidden_units=gru_layers).to(device)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        # Only reachable with aggregate=True; their forward_once never applies it.
        self.decode_mlp = nn.Sequential(nn.Linear(z_dim, z_dim, device=device),
                                        nn.Softmax(dim=1))

        self.device = device
        self.ap_loss = APLoss(pos_len=pos_samples, margin=margin, gamma=gamma,
                              surrogate_loss='squared', device=device)
        self.s = s
        self.beta = beta
        self.neg_sampling = neg_sampling
        self.neg_filter = neg_filter
        self.aggregate = aggregate
        self.max_pos = max_pos

    # -- embedding ------------------------------------------------------------------------------
    def forward(self, x, eis, eas, idxs, ptrs, h0=None, include_h=False):
        '''
        eis/eas are per-snapshot edge indices and edge attributes. Returns (T, N, z), and with
        include_h also the final hidden state so the caller can continue the recurrence.
        '''
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

        # dim=0 -> (T, N, h): the GRU scans time, not the node axis (fix 2).
        out, h = self.rnn(torch.stack(zs, dim=0), h0, include_h=True)

        if self.aggregate:
            # idxs[t]/ptrs[t], not idxs[i] (fix 3).
            out = torch.stack([self.sample_z(out[t], idxs[t], ptrs[t])
                               for t in range(out.size(0))])

        return (out, h) if include_h else out

    def sample_z(self, z, idx, ptr):
        '''
        Neighbour aggregation of eq. 11, vectorized.

        The original loops over all N nodes in Python and appends to a list, so one timestep
        builds ~N little autograd subgraphs and a chunk of T snapshots builds T*N of them -- that,
        not the convolutions, is what made memory climb snapshot by snapshot.
        `ones(deg).multinomial(s, replacement=True)` draws s neighbours uniformly with
        replacement, which is exactly `randint(0, deg, (s,))`, so the distribution is unchanged.
        '''
        idx, ptr = idx.to(self.device), ptr.to(self.device)
        if ptr.numel() == 0:                       # snapshot with no edges
            return self.decode_mlp(z)

        deg = idx[1:] - idx[:-1]                   # (N,) neighbours per node
        has = (deg > 0).unsqueeze(1)
        off = torch.randint(0, 1 << 30, (z.size(0), self.s), device=self.device)
        off = off % deg.clamp(min=1).unsqueeze(1)
        nbr = ptr[(idx[:-1].unsqueeze(1) + off).clamp(max=ptr.numel() - 1)]

        agg = (z[nbr].sum(dim=1) + z) / (self.s + 1)
        return self.decode_mlp(torch.where(has, agg, z))

    # -- scoring --------------------------------------------------------------------------------
    def decode(self, ei, z):
        '''
        Reconstruction probability in (0, 1). HIGH MEANS NORMAL -- negate for an anomaly score.
        See the module docstring. Matches their Euler_Embed_Unit.decode (fix 4).
        '''
        return torch.sigmoid((z[ei[0]] * z[ei[1]]).sum(dim=1))

    def sample_negatives(self, ps, num_nodes):
        '''Negative pairs for L_ap and for the validation LP metric.'''
        if self.neg_sampling == 'degree':
            e = ps.size(1)
            si = torch.randint(0, e, (e,), device=ps.device)
            di = torch.randint(0, e, (e,), device=ps.device)
            return torch.stack([ps[0][si], ps[1][di]])

        ns = torch.randint(0, num_nodes, ps.size(), device=ps.device)
        if self.neg_filter:
            key = lambda e: e[0] + e[1] * num_nodes      # the hash fast_negative_sampling uses
            pos_key = key(ps)
            for _ in range(5):                           # their while-loop, bounded
                bad = torch.isin(key(ns), pos_key)
                if not bad.any():
                    break
                ns[:, bad] = torch.randint(0, num_nodes, (2, int(bad.sum())), device=ps.device)
        return ns

    def calc_loss_argus(self, zs, eis):
        '''
        loss = L_ap + beta * L_DEC, with the OBSERVED edges as the positive class (fix 1).

        beta defaults to 0, so L_DEC is off and this is their code's loss. decode() is already
        in (0, 1), so plain BCE applies rather than the with-logits form.
        '''
        tot = torch.zeros(1, device=self.device)
        for i in range(len(zs)):
            ps = eis[i]
            if ps.size(1) == 0:
                continue
            ps = ps.to(self.device)
            if self.max_pos and ps.size(1) > self.max_pos:
                keep = torch.randperm(ps.size(1), device=ps.device)[:self.max_pos]
                ps = ps[:, keep]

            ns = self.sample_negatives(ps, zs.size(1))
            pos = self.decode(ps, zs[i])                  # observed edges -> class 1
            neg = self.decode(ns, zs[i])                  # random pairs   -> class 0

            score = torch.cat([pos, neg])                 # positives first: APLoss indexes them
            y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).detach()
            t_index = torch.arange(pos.size(0), dtype=torch.int64, device=self.device).detach()

            tot = tot + self.ap_loss(score, y, t_index)
            if self.beta:
                tot = tot + self.beta * F.binary_cross_entropy(score.clamp(1e-7, 1 - 1e-7), y)

        return tot.true_divide(len(zs))

    def validate(self, zs, eis):
        '''
        Reconstruction probabilities for observed edges and for random pairs, as a (pos, neg)
        pair of CPU tensors. The caller builds the LP metric from them with OBSERVED AS CLASS 1
        and the RAW probabilities -- this is the one metric that is not negated.

        It uses no malicious labels, which matters: a scrubbed validation window has no malicious
        edges at all, so an anomaly-labelled validation metric would be undefined there and
        selecting on it would mean selecting on the test set.
        '''
        pos, neg = [], []
        for i in range(min(len(zs), len(eis))):
            ps = eis[i]
            if ps.size(1) == 0:
                continue
            ps = ps.to(self.device)
            ns = self.sample_negatives(ps, zs.size(1))
            pos.append(self.decode(ps, zs[i]))
            neg.append(self.decode(ns, zs[i]))
        # .cpu(): callers hand these straight to sklearn.
        return torch.cat(pos).cpu(), torch.cat(neg).cpu()


# argus_test_lanl_ts.py referred to the fixed model as ArgusTS while the unfixed base class still
# existed. There is now one class; the alias keeps older call sites working.
ArgusTS = Argus


def chunked_forward(model, x, split, chunk, h0=None):
    '''
    Embed a split's snapshots in chunks of `chunk`, threading the GRU state across the seams.

    Returns (T, N, z) and the final hidden state, so a later window can continue the recurrence
    rather than restarting it from zeros. One chunk holds the autograd graph for every snapshot
    in it, which is why chunk size and not batch size governs peak memory.
    '''
    n = len(split.edge_index)
    zs, h = [], h0
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        out, h = model.forward(x, split.edge_index[st:en], split.eas[st:en],
                               split.idxs[st:en] if split.idxs else [],
                               split.ptrs[st:en] if split.ptrs else [],
                               h0=h, include_h=True)
        zs.append(out)
    return torch.cat(zs, dim=0), h