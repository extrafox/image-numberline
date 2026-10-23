"""Integer Haar (S-transform) pyramid over YCoCg-R: an exactly reversible map from 8-bit sRGB images.

Each 2x2 node yields one 9-value symbol: horizontal, vertical and diagonal detail for each of Y, Co, Cg.
Values are split into a coarse part q = v >> SHIFT[channel] (coded jointly, like rh.model's residuals)
and SHIFT low bits refined one bit-plane per pass, most significant first. Y keeps more coarse bits than
Co/Cg. All passes together are lossless.
"""
import numpy as np

from rh import model as m

LEVELS = 8
SHIFT = np.array([4, 6, 6])  # per channel Y, Co, Cg
PASSES = int(SHIFT.max())
NODE_SHIFT = np.repeat(SHIFT, 3)  # 9 values per node, ordered (channel, H/V/D)
# Coarse digit radix, tight to the reachable ranges: Y H/V in [-255, 255] and D in [-510, 510]; Co/Cg
# twice that. After the shift: Y 32, 32, 64; Co/Cg 16, 16, 32.
NODE_RADIX = np.array([32, 32, 64, 16, 16, 32, 16, 16, 32])
NODE_OFFSET = NODE_RADIX // 2
NODE_WEIGHTS = np.cumprod(np.concatenate([[1], NODE_RADIX[:-1]])).astype(np.uint64)
ROOT_RADIX = np.array([16, 8, 8])  # Y in [0, 255] >> 4; Co/Cg in [-255, 255] >> 6
ROOT_OFFSET = np.array([0, 4, 4])


def srgb8_from_linear(lin):
    return np.rint(m.linear_to_srgb(np.asarray(lin, np.float64)) * 255).astype(np.int64)


def linear_from_srgb8(rgb):
    return m.srgb_to_linear(np.clip(rgb, 0, 255) / 255.0)


def ycocg_from_rgb(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    return np.stack([t + (cg >> 1), co, cg], axis=-1)


def rgb_from_ycocg(ycc):
    y, co, cg = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    t = y - (cg >> 1)
    g = cg + t
    b = t - (co >> 1)
    return np.stack([b + co, g, b], axis=-1)


def _lift(x, y):
    return (x + y) >> 1, x - y


def _unlift(low, d):
    x = low + ((d + 1) >> 1)
    return x, x - d


def analyze(ycc):
    """(256, 256, 3) ints -> root (3,), [details for levels 1..8, each (4^(k-1), 9)] in raster order."""
    a = ycc
    details = []
    while a.shape[0] > 1:
        l1, h1 = _lift(a[0::2, 0::2], a[0::2, 1::2])
        l2, h2 = _lift(a[1::2, 0::2], a[1::2, 1::2])
        ll, v = _lift(l1, l2)
        h, d = _lift(h1, h2)
        details.append(np.stack([h, v, d], axis=-1).reshape(-1, 9))  # (channel, H/V/D)
        a = ll
    return a[0, 0], details[::-1]


def synthesize(root, details):
    a = np.asarray(root).reshape(1, 1, 3)
    for det in details:
        s = a.shape[0]
        h, v, d = det.reshape(s, s, 3, 3).transpose(3, 0, 1, 2)
        l1, l2 = _unlift(a, v)
        h1, h2 = _unlift(h, d)
        out = np.empty((2 * s, 2 * s, 3), np.int64)
        out[0::2, 0::2], out[0::2, 1::2] = _unlift(l1, h1)
        out[1::2, 0::2], out[1::2, 1::2] = _unlift(l2, h2)
        a = out
    return a


def _unlift_float(low, d):
    x = low + d / 2 + 0.25  # expectation of the integer inverse over the parity of d
    return x, x - d


def synthesize_float(root, details):
    """Real-valued inverse for images with expected (non-integer) values; returns float YCoCg."""
    a = np.asarray(root, np.float64).reshape(1, 1, 3)
    for det in details:
        s = a.shape[0]
        h, v, d = det.reshape(s, s, 3, 3).transpose(3, 0, 1, 2)
        l1, l2 = _unlift_float(a, v)
        h1, h2 = _unlift_float(h, d)
        out = np.empty((2 * s, 2 * s, 3))
        out[0::2, 0::2], out[0::2, 1::2] = _unlift_float(l1, h1)
        out[1::2, 0::2], out[1::2, 1::2] = _unlift_float(l2, h2)
        a = out
    return a


def rgb_from_ycocg_float(ycc):
    y, co, cg = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    t = y - cg / 2 + 0.25
    b = t - co / 2 + 0.25
    return np.stack([b + co, cg + t, b], axis=-1)


def coarse_keys(q):
    """Coarse node digits (n, 9), clipped into range -> uint64 keys."""
    dig = np.clip(q + NODE_OFFSET, 0, NODE_RADIX - 1)
    return (dig.astype(np.uint64) * NODE_WEIGHTS).sum(axis=1, dtype=np.uint64)


def keys_to_coarse(keys):
    keys = np.asarray(keys, np.uint64)
    out = np.empty((len(keys), 9), np.int64)
    for i, r in enumerate(NODE_RADIX.tolist()):
        out[:, i] = (keys % np.uint64(r)).astype(np.int64)
        keys = keys // np.uint64(r)
    return out - NODE_OFFSET


def root_index(q):
    d = np.asarray(q) + ROOT_OFFSET
    return int(d[0] + ROOT_RADIX[0] * (d[1] + ROOT_RADIX[1] * d[2]))


def root_coarse(index):
    d = np.array([index % ROOT_RADIX[0], (index // ROOT_RADIX[0]) % ROOT_RADIX[1], index // (ROOT_RADIX[0] * ROOT_RADIX[1])])
    return d - ROOT_OFFSET


def refinement(values, q, shift, j):
    """Pass j (1-based) symbols and contexts for (n, c) values with coarse parts q and per-column shifts.

    Columns with shift >= j contribute bit (shift - j) of their remainder; the context is which of those
    columns have a negative coarse part (floor remainders of small negatives sit near the top of the bin).
    """
    cols = np.nonzero(shift >= j)[0]
    rem = values[:, cols] - (q[:, cols] << shift[cols])
    bits = (rem >> (shift[cols] - j)) & 1
    w = 1 << np.arange(len(cols))
    return (bits * w).sum(axis=1), ((q[:, cols] < 0) * w).sum(axis=1), len(cols)


def decompose(lin):
    """Everything the coder needs for one image: root/node values and coarse parts."""
    root, details = analyze(ycocg_from_rgb(srgb8_from_linear(lin)))
    rq = root >> SHIFT
    nodes = [d for d in details]
    return root, rq, nodes, [d >> NODE_SHIFT for d in nodes]
