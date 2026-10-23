"""Integer-Haar codec (same model and stream as rh.haar_codec) with a Hilbert-style ordering of the line.

Two changes to where symbols sit on [0, 1); probabilities, and so bit costs, are unchanged:
  - Snake order within each table: root colours, coarse node symbols and refinement symbols are ordered
    by a reflected mixed-radix (boustrophedon) traversal of their digits, most significant digit first
    (Y before Co/Cg), so neighbouring symbols differ by one step in one value. Escape stays last.
  - Alternating direction along the stream: each table is traversed backwards when the running rank
    of everything coded before it is odd. At every interval boundary the two sides then continue with
    the same symbols, so a boundary only changes the one symbol where the intervals split.
"""
import bisect
import math

import numpy as np

from rh import haar as h
from rh import model as m
from rh.coder import Decoder, Encoder
from rh.haar_codec import _expected_bits, _mean


def snake_rank(digits, radix):
    """Rank of each digit row (most significant first) in the reflected mixed-radix order."""
    g = np.zeros(len(digits), np.int64)
    for i, r in enumerate(np.asarray(radix).tolist()):
        d = digits[:, i]
        g = g * r + np.where(g % 2 == 0, d, r - 1 - d)
    return g


class RootTable:
    """Root colours on the line in snake order, with integer frequencies per position."""

    def __init__(self, order, freqs):
        self.order = order  # position -> root index
        self.pos = np.argsort(order)
        self.cum = [0] + np.cumsum(freqs).tolist()
        self.values = np.array([h.root_coarse(int(i)) for i in order], np.float64)
        self.freqs = np.asarray(freqs, np.float64)

    def put(self, st, rq):
        return st.put(self.cum, int(self.pos[h.root_index(rq)]))

    def get(self, ls):
        """(coarse root, determined?)"""
        a, b = ls.get(self.cum)
        return (self.values[a], True) if a == b else (_mean(self.values, self.freqs, a, b), False)


class CoarseTable:
    """Coarse node symbols in snake order plus a final escape. After an escape the symbol is coded by
    the fallback table if there is one, else as uniform digits."""

    def __init__(self, keys, counts, escape, fallback=None):
        self.keys = np.asarray(keys, np.uint64)
        self.coarse = h.keys_to_coarse(self.keys)
        self.pos = {k: i for i, k in enumerate(self.keys.tolist())}
        self.cum = [0] + np.cumsum(np.concatenate([np.asarray(counts, np.int64), [int(escape)]])).tolist()
        self.escape = len(self.keys)
        self.fallback = fallback
        freqs = np.asarray(counts, np.float64)
        mean = freqs @ self.coarse / freqs.sum() if len(freqs) else np.zeros(9)
        self.values = np.vstack([self.coarse, [mean]]).astype(np.float64)  # escape stands for the mean
        self.freqs = np.append(freqs, int(escape))

    def put(self, st, key, digits):
        i = self.pos.get(key)
        if i is not None:
            return st.put(self.cum, i)
        cost = st.put(self.cum, self.escape)
        if self.fallback is not None:
            return cost + self.fallback.put(st, key, digits)
        return cost + sum(st.put(UNIFORM[r], d) for d, r in zip(digits, h.NODE_RADIX.tolist()))

    def get(self, ls):
        """("known", q) | ("partial", mean q over candidates) | ("stopped", None)."""
        a, b = ls.get(self.cum)
        if a != b:
            return "partial", _mean(self.values, self.freqs, a, b)
        if a < self.escape:
            return "known", self.coarse[a]
        if self.fallback is not None:
            return self.fallback.get(ls)
        digits = []
        for r in h.NODE_RADIX.tolist():
            d, e = ls.get(UNIFORM[r])
            if d != e:
                return "stopped", None
            digits.append(d)
        return "known", np.array(digits) - h.NODE_OFFSET


UNIFORM = {r: list(range(r + 1)) for r in set(h.NODE_RADIX.tolist())}


class Model:
    def __init__(self, z):
        n_root = int(np.prod(h.ROOT_RADIX))
        root_digits = np.array([h.root_coarse(i) for i in range(n_root)]) + h.ROOT_OFFSET
        root_order = np.argsort(snake_rank(root_digits, h.ROOT_RADIX))
        self.root = RootTable(root_order, z["root_counts"][root_order] + 1)
        order = np.argsort(snake_rank(h.keys_to_coarse(z["keys"]) + h.NODE_OFFSET, h.NODE_RADIX))
        self.coarse = CoarseTable(z["keys"][order], z["counts"][order], int(z["escape"]))
        # Refinement: positions in Gray order of the bits (first refining value most significant).
        self.ref_perm, self.ref_inv, self.node_ref, self.root_ref = {}, {}, [], []
        for j in range(1, h.PASSES + 1):
            for name, tables in (("node", self.node_ref), ("root", self.root_ref)):
                t = z[f"{name}_ref_{j}"]
                nb = int(np.log2(t.shape[1]))
                if nb not in self.ref_perm:
                    bits = (np.arange(2 ** nb)[:, None] >> np.arange(nb)) & 1
                    self.ref_perm[nb] = np.argsort(snake_rank(bits, [2] * nb))  # position -> symbol
                    self.ref_inv[nb] = np.argsort(self.ref_perm[nb]).tolist()
                tables.append([[0] + row for row in np.cumsum(t[:, self.ref_perm[nb]] + 1, axis=1).tolist()])
        self.value_mean = np.asarray(z["value_mean"], np.float64)
        self.root_expect = [_expected_bits(z[f"root_ref_{j}"]) for j in range(1, h.PASSES + 1)]
        self.node_expect = [_expected_bits(z[f"node_ref_{j}"]) for j in range(1, h.PASSES + 1)]


def load_model(path):
    return Model(np.load(path))


def _next_parity(parity, cum, pos):
    """Parity of the running rank after coding position pos of a table of len(cum) - 1 symbols."""
    r = len(cum) - 1
    e = r - 1 - pos if parity else pos
    return (parity & (r & 1)) ^ (e & 1)


class _Stream:
    def __init__(self):
        self.enc, self.parity = Encoder(), 0

    def put(self, cum, pos):
        lo, hi, total = cum[pos], cum[pos + 1], cum[-1]
        if self.parity:
            lo, hi = total - hi, total - lo
        self.enc.encode(lo, hi, total)
        self.parity = _next_parity(self.parity, cum, pos)
        return -math.log2((hi - lo) / total)


def _get(dec, cum, parity):
    total = cum[-1]
    t = dec.target(total)
    if parity:
        pos = bisect.bisect_left(cum, total - t) - 1
        dec.consume(total - cum[pos + 1], total - cum[pos], total)
    else:
        pos = bisect.bisect_right(cum, t) - 1
        dec.consume(cum[pos], cum[pos + 1], total)
    return pos


class _Lockstep:
    """As in rh.haar_codec, tracking the running parity while the two decodes agree."""

    def __init__(self, bits):
        self.lo, self.hi, self.stopped, self.parity = Decoder(bits, 0), Decoder(bits, 1), False, 0

    def get(self, cum):
        a, b = _get(self.lo, cum, self.parity), _get(self.hi, cum, self.parity)
        if a != b:
            self.stopped = True
            return min(a, b), max(a, b)
        self.parity = _next_parity(self.parity, cum, a)
        return a, a


def encode(mdl, lin):
    st = _Stream()
    level_bits = encode_pixels(mdl, st, lin, mdl.root, [mdl.coarse] * h.LEVELS)
    return st.enc.finish(), level_bits


def encode_pixels(mdl, st, lin, root_table, level_tables):
    """Code the image's pixels onto st, with the given root table and one coarse table per level."""
    root, rq, nodes, nq = h.decompose(lin)
    level_bits = [root_table.put(st, rq)] + [0.0] * h.LEVELS
    for k, (q, table) in enumerate(zip(nq, level_tables), 1):
        for key, digits in zip(h.coarse_keys(q).tolist(), (q + h.NODE_OFFSET).tolist()):
            level_bits[k] += table.put(st, key, digits)
    for j in range(1, h.PASSES + 1):
        sym, ctx, nb = h.refinement(root[None], rq[None], h.SHIFT, j)
        level_bits[0] += st.put(mdl.root_ref[j - 1][int(ctx[0])], mdl.ref_inv[nb][int(sym[0])])
        for k, (v, q) in enumerate(zip(nodes, nq), 1):
            sym, ctx, nb = h.refinement(v, q, h.NODE_SHIFT, j)
            inv = mdl.ref_inv[nb]
            for s, c in zip(sym.tolist(), ctx.tolist()):
                level_bits[k] += st.put(mdl.node_ref[j - 1][c], inv[s])
    return level_bits


def decode(mdl, bits, levels=h.LEVELS):
    """Decode any bit list; undetermined values take their expectation (see rh.haar_codec.decode)."""
    assert levels == h.LEVELS, "the refinement passes follow all levels; only full decodes are supported"
    return decode_pixels(mdl, _Lockstep(bits), mdl.root, [mdl.coarse] * h.LEVELS)


def decode_pixels(mdl, ls, root_table, level_tables):
    centre = ((1 << h.SHIFT) - 1) / 2
    rq, root_known = root_table.get(ls)
    root = rq * (1 << h.SHIFT) + (0 if root_known else centre)
    nodes, nq, known = [], [], []
    for k, table in enumerate(level_tables, 1):
        n_nodes = 4 ** (k - 1)
        v, q, kn = np.tile(mdl.value_mean, (n_nodes, 1)), np.zeros((n_nodes, 9)), np.zeros(n_nodes, bool)
        for n in range(n_nodes):
            if ls.stopped:
                break
            kind, c = table.get(ls)
            if kind == "partial":
                v[n] = c * (1 << h.NODE_SHIFT) + ((1 << h.NODE_SHIFT) - 1) / 2
            elif kind == "known":
                q[n], v[n], kn[n] = c, c * (1 << h.NODE_SHIFT), True
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
    """Pass j contributions (value units) for rows of q; undetermined rows get expected bits."""
    tables = (mdl.root_ref if is_root else mdl.node_ref)[j - 1]
    expect = (mdl.root_expect if is_root else mdl.node_expect)[j - 1]
    cols = np.nonzero(shift >= j)[0]
    nb = len(cols)
    w = 1 << np.arange(nb)
    ctx = ((q[:, cols] < 0) * w).sum(axis=1)
    pos_bits = ((mdl.ref_perm[nb][:, None] >> np.arange(nb)) & 1).astype(np.float64)  # by position
    out_bits = expect[ctx]
    for n, c in enumerate(ctx.tolist()):
        if ls.stopped:
            break
        a, b = ls.get(tables[c])
        out_bits[n] = pos_bits[a] if a == b else _mean(pos_bits, np.diff(tables[c]), a, b)
    out = np.zeros(q.shape)
    out[:, cols] = out_bits * (1 << (shift[cols] - j))
    if rows is None:
        return out
    full = np.zeros((len(rows), q.shape[1]))
    full[rows] = out
    return full


def reconstruct(lin):
    return h.linear_from_srgb8(h.srgb8_from_linear(lin))
