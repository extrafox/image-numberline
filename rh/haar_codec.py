"""Lossless integer-Haar codec with the same API as rh.codec.

Stream order: coarse pass (root colour, then one joint 9-value node symbol per parent, levels 1..8
breadth-first), then PASSES bit-plane refinement passes in the same node order, each symbol being one
bit per still-refining value, conditioned on which of those values have negative coarse parts.
Root colours are ordered on the line by OKLab lightness; coarse node symbols by total Y detail, with an
escape (then uniform digits) at the end; refinement symbols in binary order. Unseen refinement symbols
get add-one counts.
"""
import bisect
import math

import numpy as np

from rh import haar as h
from rh import model as m
from rh.coder import Decoder, Encoder


class Model:
    def __init__(self, z):
        n_root = int(np.prod(h.ROOT_RADIX))
        centre = np.array([h.root_coarse(i) for i in range(n_root)]) * (1 << h.SHIFT) + (1 << h.SHIFT) // 2
        lightness = m.oklab_from_linear(h.linear_from_srgb8(h.rgb_from_ycocg(centre)))[:, 0]
        self.root_order = np.argsort(lightness, kind="stable")
        self.root_pos = np.argsort(self.root_order)
        self.root_cum = [0] + np.cumsum(z["root_counts"][self.root_order] + 1).tolist()
        self.keys = z["keys"]
        self.key_coarse = h.keys_to_coarse(self.keys)
        self.pos = {k: i for i, k in enumerate(self.keys.tolist())}
        self.cum = [0] + np.cumsum(np.concatenate([z["counts"], [int(z["escape"])]])).tolist()
        self.escape = len(self.keys)
        cum = lambda t: [[0] + row for row in np.cumsum(t + 1, axis=1).tolist()]  # per-context rows
        self.node_ref = [cum(z[f"node_ref_{j}"]) for j in range(1, h.PASSES + 1)]
        self.root_ref = [cum(z[f"root_ref_{j}"]) for j in range(1, h.PASSES + 1)]
        # Expected values for whatever a prefix leaves undetermined.
        self.root_values = np.array([h.root_coarse(int(i)) for i in self.root_order], np.float64)
        self.root_freqs = np.diff(self.root_cum)
        self.coarse_values = np.vstack([self.key_coarse, [np.zeros(9)]]).astype(np.float64)
        self.coarse_freqs = np.asarray(z["counts"], np.float64)
        self.coarse_mean = self.coarse_freqs @ self.key_coarse / self.coarse_freqs.sum()
        self.coarse_values[-1] = self.coarse_mean  # escape stands for the mean
        self.value_mean = np.asarray(z["value_mean"], np.float64)
        self.coarse_freqs = np.append(self.coarse_freqs, int(z["escape"]))
        self.root_expect = [_expected_bits(z[f"root_ref_{j}"]) for j in range(1, h.PASSES + 1)]
        self.node_expect = [_expected_bits(z[f"node_ref_{j}"]) for j in range(1, h.PASSES + 1)]


def _expected_bits(table):
    """(contexts, symbols) counts -> (contexts, bits) expected value of each refinement bit."""
    nb = int(np.log2(table.shape[1]))
    sym_bits = (np.arange(table.shape[1])[:, None] >> np.arange(nb)) & 1
    p = (table + 1) / (table + 1).sum(axis=1, keepdims=True)
    return p @ sym_bits


def load_model(path):
    return Model(np.load(path))


def _put(enc, cum, i):
    lo, hi, total = cum[i], cum[i + 1], cum[-1]
    enc.encode(lo, hi, total)
    return -math.log2((hi - lo) / total)


def _get(dec, cum):
    i = bisect.bisect_right(cum, dec.target(cum[-1])) - 1
    dec.consume(cum[i], cum[i + 1], cum[-1])
    return i


def encode(mdl, lin):
    root, rq, nodes, nq = h.decompose(lin)
    enc = Encoder()
    level_bits = [_put(enc, mdl.root_cum, int(mdl.root_pos[h.root_index(rq)]))] + [0.0] * h.LEVELS
    for k, q in enumerate(nq, 1):
        for key, digits in zip(h.coarse_keys(q).tolist(), (q + h.NODE_OFFSET).tolist()):
            i = mdl.pos.get(key)
            if i is not None:
                level_bits[k] += _put(enc, mdl.cum, i)
                continue
            level_bits[k] += _put(enc, mdl.cum, mdl.escape)
            for d, r in zip(digits, h.NODE_RADIX.tolist()):
                enc.encode(d, d + 1, r)
                level_bits[k] += math.log2(r)
    for j in range(1, h.PASSES + 1):
        sym, ctx, _ = h.refinement(root[None], rq[None], h.SHIFT, j)
        level_bits[0] += _put(enc, mdl.root_ref[j - 1][int(ctx[0])], int(sym[0]))
        for k, (v, q) in enumerate(zip(nodes, nq), 1):
            sym, ctx, _ = h.refinement(v, q, h.NODE_SHIFT, j)
            for s, c in zip(sym.tolist(), ctx.tolist()):
                level_bits[k] += _put(enc, mdl.node_ref[j - 1][c], s)
    return enc.finish(), level_bits


class _Lockstep:
    """Decode the prefix padded with 0s and with 1s together. Symbols both agree on are fixed by the
    prefix; at the first disagreement the candidates form a range [a, b] and decoding stops."""

    def __init__(self, bits):
        self.lo, self.hi, self.stopped = Decoder(bits, 0), Decoder(bits, 1), False

    def get(self, cum):
        a, b = _get(self.lo, cum), _get(self.hi, cum)
        if a != b:
            self.stopped = True
        return a, b


def _mean(values, freqs, a, b):
    w = np.asarray(freqs[a:b + 1], np.float64)
    return w @ values[a:b + 1] / w.sum()


def decode(mdl, bits, levels=h.LEVELS):
    """Decode any bit list. Whatever the bits leave undetermined takes its expected value under the model
    (property: a prefix shows no invented detail), so complete codes decode exactly and prefixes smoothly."""
    assert levels == h.LEVELS, "the refinement passes follow all levels; only full decodes are supported"
    ls = _Lockstep(bits)
    centre = ((1 << h.SHIFT) - 1) / 2  # mean of the unknown low bits within a coarse bin
    a, b = ls.get(mdl.root_cum)
    root_known = a == b
    rq = mdl.root_values[a] if root_known else _mean(mdl.root_values, mdl.root_freqs, a, b)
    root = rq * (1 << h.SHIFT) + (0 if root_known else centre)
    # Undetermined nodes take the corpus mean of the full values; a partly determined node takes the
    # candidates' mean coarse value plus the bin centre. Only fully known coarse values get refined.
    nodes, nq, known = [], [], []
    for k in range(1, h.LEVELS + 1):
        n_nodes = 4 ** (k - 1)
        v, q, kn = np.tile(mdl.value_mean, (n_nodes, 1)), np.zeros((n_nodes, 9)), np.zeros(n_nodes, bool)
        for n in range(n_nodes):
            if ls.stopped:
                break
            a, b = ls.get(mdl.cum)
            if a != b:
                v[n] = _mean(mdl.coarse_values, mdl.coarse_freqs, a, b) * (1 << h.NODE_SHIFT) + ((1 << h.NODE_SHIFT) - 1) / 2
                continue
            if a < mdl.escape:
                q[n] = mdl.key_coarse[a]
            else:
                digits = [ls.get([*range(r + 1)])[0] for r in h.NODE_RADIX.tolist() if not ls.stopped]
                if ls.stopped:
                    continue
                q[n] = np.array(digits) - h.NODE_OFFSET
            v[n], kn[n] = q[n] * (1 << h.NODE_SHIFT), True
        nodes.append(v), nq.append(q), known.append(kn)
    for j in range(1, h.PASSES + 1):
        if root_known:
            root = root + _bits(ls, mdl, j, True, rq[None], h.SHIFT)[0]
        nodes = [v + _bits(ls, mdl, j, False, q[kn], h.NODE_SHIFT, kn) for v, q, kn in zip(nodes, nq, known)]
    if not ls.stopped:  # everything determined: exact integer inverse
        root, nodes = np.rint(root).astype(np.int64), [np.rint(v).astype(np.int64) for v in nodes]
        return h.linear_from_srgb8(h.rgb_from_ycocg(h.synthesize(root, nodes)))
    rgb = h.rgb_from_ycocg_float(h.synthesize_float(root, nodes))
    return m.srgb_to_linear(np.clip(rgb, 0, 255) / 255.0)


def _bits(ls, mdl, j, is_root, q, shift, rows=None):
    """Pass j contributions (in value units) for every row of q; undetermined rows get expected bits.
    With rows (a mask), q holds only those rows and the result is scattered back into the full array."""
    tables = (mdl.root_ref if is_root else mdl.node_ref)[j - 1]
    expect = (mdl.root_expect if is_root else mdl.node_expect)[j - 1]
    cols = np.nonzero(shift >= j)[0]
    w = 1 << np.arange(len(cols))
    ctx = ((q[:, cols] < 0) * w).sum(axis=1)
    sym_bits = (np.arange(2 ** len(cols))[:, None] >> np.arange(len(cols))) & 1
    out_bits = expect[ctx]
    for n, c in enumerate(ctx.tolist()):
        if ls.stopped:
            break
        a, b = ls.get(tables[c])
        out_bits[n] = sym_bits[a] if a == b else _mean(sym_bits, np.diff(tables[c]), a, b)
    out = np.zeros(q.shape)
    out[:, cols] = out_bits * (1 << (shift[cols] - j))
    if rows is None:
        return out
    full = np.zeros((len(rows), q.shape[1]))
    full[rows] = out
    return full


def reconstruct(lin):
    return h.linear_from_srgb8(h.srgb8_from_linear(lin))
