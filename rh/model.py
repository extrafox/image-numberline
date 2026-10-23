"""Symbols, colour spaces and the closed-loop pyramid shared by the model builder and the codec.

An image is a 256x256 linear-light RGB array. Its pyramid has levels of side 1, 2, ..., 256, each the
linear-light 2x2 mean of the level below. The code emits, breadth-first:
  - one root symbol: the 1x1 colour, quantized absolutely in gamma YCbCr (BY bits Y, BC bits Cb/Cr);
  - for levels 1..8, one residual symbol per parent pixel: its four children (TL, TR, BL, BR) minus the
    *reconstructed* parent, quantized with the same steps (closed loop, so the decoder can follow).
"""
import numpy as np

BY, BC = 4, 2
LEVELS = 8
SIDE = 2 ** LEVELS
STEPS = np.array([2 ** BY, 2 ** BC, 2 ** BC], dtype=np.float64)
OFFSET = np.array([0.0, 0.5, 0.5])  # shifts Cb/Cr from [-0.5, 0.5] to [0, 1]
ROOT_RADIX = STEPS.astype(np.int64)  # 16 x 4 x 4 = 256 root colours
RES_RADIX = 2 * STEPS.astype(np.int64) + 1  # residual digit in [-step, step] -> 33 x 9 x 9 per child
RES_DIGITS = np.tile(RES_RADIX, 4)  # 12 digits per residual symbol
RES_WEIGHTS = np.cumprod(np.concatenate([[1], RES_DIGITS[:-1]])).astype(np.uint64)
RES_ALPHABET = int(np.prod(RES_DIGITS.astype(object)))


def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def ycbcr_from_linear(lin):
    g = linear_to_srgb(lin)
    y = g @ np.array([0.299, 0.587, 0.114])
    return np.stack([y, (g[..., 2] - y) / 1.772, (g[..., 0] - y) / 1.402], axis=-1)


def linear_from_ycbcr(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    r, b = y + 1.402 * cr, y + 1.772 * cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return srgb_to_linear(np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0))


def oklab_from_linear(lin):
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.cbrt(np.clip(lin, 0.0, None) @ m1.T) @ m2.T


def clamp_ycbcr(ycc):
    return np.clip(ycc, [0.0, -0.5, -0.5], [1.0, 0.5, 0.5])


def pyramid(lin):
    """Linear-light pyramid as gamma YCbCr, coarsest first: sides 1, 2, ..., 256."""
    levels = [np.asarray(lin, np.float64)]
    while levels[-1].shape[0] > 1:
        a = levels[-1]
        levels.append((a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) / 4)
    return [ycbcr_from_linear(l) for l in levels[::-1]]


def blocks(a):
    """(s, s, 3) -> (s/2 * s/2, 4, 3), children ordered TL, TR, BL, BR, blocks in raster order."""
    return np.stack([a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]], axis=2).reshape(-1, 4, 3)


def unblocks(b, side):
    """Inverse of blocks: (side/2 * side/2, 4, 3) -> (side, side, 3)."""
    h = side // 2
    b = b.reshape(h, h, 4, 3)
    out = np.empty((side, side, 3))
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = b[:, :, 0], b[:, :, 1], b[:, :, 2], b[:, :, 3]
    return out


def root_quantize(ycc):
    q = np.clip(np.floor((ycc + OFFSET) * STEPS), 0, STEPS - 1).astype(np.int64)
    return int(q[0] + ROOT_RADIX[0] * (q[1] + ROOT_RADIX[1] * q[2]))


def root_recon(index):
    q = np.array([index % ROOT_RADIX[0], (index // ROOT_RADIX[0]) % ROOT_RADIX[1], index // (ROOT_RADIX[0] * ROOT_RADIX[1])])
    return (q + 0.5) / STEPS - OFFSET


def residual_digits(children, parent_recon):
    """children (n, 4, 3) targets, parent_recon (n, 3) -> digits (n, 12) in [0, RES_RADIX)."""
    q = np.rint((children - parent_recon[:, None, :]) * STEPS)
    return (np.clip(q, -STEPS, STEPS) + STEPS).astype(np.int64).reshape(-1, 12)


def residual_recon(digits, parent_recon):
    """digits (n, 12), parent_recon (n, 3) -> reconstructed children (n, 4, 3), clamped to valid YCbCr."""
    q = digits.reshape(-1, 4, 3) - STEPS
    return clamp_ycbcr(parent_recon[:, None, :] + q / STEPS)


def digits_to_keys(digits):
    return (digits.astype(np.uint64) * RES_WEIGHTS).sum(axis=1, dtype=np.uint64)


def keys_to_digits(keys):
    keys = np.asarray(keys, np.uint64)
    out = np.empty((len(keys), 12), np.int64)
    for i, r in enumerate(RES_DIGITS):
        out[:, i] = (keys % np.uint64(r)).astype(np.int64)
        keys = keys // np.uint64(r)
    return out


def symbols(lin):
    """Closed-loop symbols for an image: (root index, [keys for levels 1..8]) and the recon pyramid."""
    target = pyramid(lin)
    root = root_quantize(target[0][0, 0])
    recon = [root_recon(root).reshape(1, 1, 3)]
    keys = []
    for k in range(1, LEVELS + 1):
        parent = recon[-1].reshape(-1, 3)
        d = residual_digits(blocks(target[k]), parent)
        keys.append(digits_to_keys(d))
        recon.append(unblocks(residual_recon(d, parent), 2 ** k))
    return root, keys, recon
