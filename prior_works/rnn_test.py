'''
Self-test for the recurrent-state handling in argus_test_lanl_ts.py.

Uses a stand-in model with the same shape contract as Argus (per-snapshot embeddings stacked
dim=1, recurrent module, transposed back so zs[t] is (N, H)), so it runs without the LANL data.

    python test_rnn_threading.py

Checks:
  1. the scan is temporal, not over nodes: permuting the node order must not change any node's
     embedding (it does under the (N, T, h) stacking this replaces)
  2. the output shape the rest of Argus expects is preserved, and h is (layers, N, H)
  3. chunked + threaded == one unchunked pass          <- the bug this was written to fix
  4. chunked without threading differs                 <- so (3) is not vacuous
  5. model.rnn is restored, including after an exception
  6. neighbour-list padding covers a longer eis list
'''
import os
import sys

import torch
from torch import nn


def _load_helpers():
    '''Load just the wrapper section of argus_test_lanl_ts.py (importing it needs argus_test_lanl).'''
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'argus_test_lanlts.py')
    src = open(path).read()
    start = src.index('class _RNNWrapper')
    end = src.rindex('# ---', 0, src.index('# Data'))
    ns = {'__name__': 'argus_rnn_helpers'}
    exec(compile(
        'import inspect\nfrom contextlib import contextmanager\nimport torch\nfrom torch import nn\n'
        "_HIDDEN_ARGS = ('h0', 'h_0', 'hx', 'h', 'hidden', 'h_prev')\n" + src[start:end],
        'argus_test_lanl_ts.py', 'exec'), ns)
    return ns['forward_h'], ns['_pad_csr']


forward_h, _pad_csr = _load_helpers()
N, T, H = 7, 6, 4


class FakeArgus(nn.Module):
    '''Same shape contract as Argus.forward: stack(dim=1) -> rnn -> transpose to (T, N, H).'''

    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(H, H)
        self.rnn = nn.GRU(H, H, 1)

    def forward(self, x, eis, eas, idxs, ptrs):
        assert len(idxs) == len(eis), f'_pad_csr should have padded: {len(idxs)} vs {len(eis)}'
        zs = torch.stack([self.emb(x) for _ in eis], dim=1)   # (N, T, H)
        out, _ = self.rnn(zs)
        return out.transpose(0, 1)                            # (T, N, H)


def main():
    torch.manual_seed(0)
    m = FakeArgus().eval()
    x = torch.randn(N, H)
    eis = [torch.zeros(2, 1, dtype=torch.long)] * T
    eas = [torch.zeros(1, 1)] * T
    idxs, ptrs = [torch.zeros(1)] * 2, [torch.zeros(1)] * 2    # deliberately shorter than eis
    ok = True

    def check(label, cond, extra=''):
        nonlocal ok
        ok &= bool(cond)
        print(f'{"PASS" if cond else "FAIL"}  {label}{("  " + extra) if extra else ""}')

    with torch.no_grad():
        z, h = forward_h(m, x, eis, eas, idxs, ptrs)

        # 1. node ordering must be irrelevant: permute the rows, and each node's embedding follows
        perm = torch.randperm(N)
        zp, _ = forward_h(m, x[perm], eis, eas, idxs, ptrs)
        check('scan is temporal (node order irrelevant)',
              torch.allclose(z[:, perm], zp, atol=1e-6))

        # the released (N, T, h) stacking fails that same test, which is why it was a bug
        base = m.forward(x, eis, eas, [idxs[0]] * T, [ptrs[0]] * T)
        basep = m.forward(x[perm], eis, eas, [idxs[0]] * T, [ptrs[0]] * T)
        check('node-major stacking would fail it', not torch.allclose(base[:, perm], basep, atol=1e-6))

        # 2. shapes
        check('output shape preserved', z.shape == base.shape, f'{tuple(z.shape)}')
        check('h is (layers, N, H)', tuple(h.shape) == (1, N, H), f'{tuple(h.shape)}')

        # 3/4. chunking
        parts, hc = [], None
        for st in range(0, T, 2):
            zc, hc = forward_h(m, x, eis[st:st + 2], eas[st:st + 2], idxs, ptrs, hc)
            parts.append(zc)
        check('chunked + threaded == unchunked', torch.allclose(z, torch.cat(parts, 0), atol=1e-6))

        naive = torch.cat([forward_h(m, x, eis[s:s + 2], eas[s:s + 2], idxs, ptrs)[0]
                           for s in range(0, T, 2)], 0)
        check('chunked without threading differs', not torch.allclose(z, naive, atol=1e-6))

    # 5. the swap is undone
    check('model.rnn restored', isinstance(m.rnn, nn.GRU))

    class Boom(FakeArgus):
        def forward(self, *a):
            raise RuntimeError('boom')

    b = Boom()
    try:
        forward_h(b, x, eis, eas, idxs, ptrs)
    except RuntimeError:
        pass
    check('model.rnn restored after exception', isinstance(b.rnn, nn.GRU))

    # 6. padding
    pi, pp = _pad_csr(idxs, ptrs, T)
    check('_pad_csr pads to len(eis)', len(pi) == T and len(pp) == T)
    check('_pad_csr truncates when longer', len(_pad_csr([0] * 9, [0] * 9, 3)[0]) == 3)

    print('\n' + ('all checks passed' if ok else 'FAILURES above'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())