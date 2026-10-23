"""Encode a 256x256 image as a bitstream (the binary expansion of x in [0, 1)) and decode any bitstream.

Root colours are placed on the line in OKLab-lightness order, so x near 0 is dark and near 1 is light.
Residual symbols are placed in order of their total Y change; the escape symbol sits at the end and is
followed by the 12 residual digits coded uniformly, so every block is representable.
"""
import bisect
import math

import numpy as np

from rh import model as m
from rh.coder import Decoder, Encoder


class Model:
    def __init__(self, root_counts, keys, counts, escape):
        # Root: add-one smoothing over all colours, ordered dark to light.
        lightness = m.oklab_from_linear(m.linear_from_ycbcr(np.array([m.root_recon(i) for i in range(len(root_counts))])))[:, 0]
        self.root_order = np.argsort(lightness, kind="stable")
        self.root_pos = np.argsort(self.root_order)
        self.root_cum = [0] + np.cumsum(np.asarray(root_counts, np.int64)[self.root_order] + 1).tolist()
        # Residuals: keys are already in line order; escape last.
        self.keys = np.asarray(keys, np.uint64)
        self.key_digits = m.keys_to_digits(self.keys)
        self.pos = {int(k): i for i, k in enumerate(self.keys.tolist())}
        self.cum = [0] + np.cumsum(np.concatenate([counts, [escape]]).astype(np.int64)).tolist()
        self.escape = len(self.keys)


def load_model(path):
    z = np.load(path)
    return Model(z["root_counts"], z["keys"], z["counts"], int(z["escape"]))


def _put(enc, cum, i, cost):
    lo, hi, total = cum[i], cum[i + 1], cum[-1]
    enc.encode(lo, hi, total)
    return cost - math.log2((hi - lo) / total)


def _get(dec, cum):
    i = bisect.bisect_right(cum, dec.target(cum[-1])) - 1
    dec.consume(cum[i], cum[i + 1], cum[-1])
    return i


def encode(mdl, lin):
    root, keys, _ = m.symbols(lin)
    enc = Encoder()
    level_bits = [_put(enc, mdl.root_cum, int(mdl.root_pos[root]), 0.0)]
    for level_keys in keys:
        cost = 0.0
        for key in level_keys.tolist():
            i = mdl.pos.get(key)
            if i is not None:
                cost = _put(enc, mdl.cum, i, cost)
                continue
            cost = _put(enc, mdl.cum, mdl.escape, cost)
            for d, r in zip(m.keys_to_digits([key])[0].tolist(), m.RES_DIGITS.tolist()):
                enc.encode(d, d + 1, r)
                cost += math.log2(r)
        level_bits.append(cost)
    return enc.finish(), level_bits


def decode(mdl, bits, levels=m.LEVELS):
    dec = Decoder(bits)
    recon = m.root_recon(int(mdl.root_order[_get(dec, mdl.root_cum)])).reshape(1, 1, 3)
    for k in range(1, levels + 1):
        digits = np.empty((4 ** (k - 1), 12), np.int64)
        for j in range(len(digits)):
            i = _get(dec, mdl.cum)
            if i < mdl.escape:
                digits[j] = mdl.key_digits[i]
                continue
            for c, r in enumerate(m.RES_DIGITS.tolist()):
                d = dec.target(r)
                dec.consume(d, d + 1, r)
                digits[j, c] = d
        recon = m.unblocks(m.residual_recon(digits, recon.reshape(-1, 3)), 2 ** k)
    return m.linear_from_ycbcr(recon)


def reconstruct(lin):
    return m.linear_from_ycbcr(m.symbols(lin)[2][-1])
