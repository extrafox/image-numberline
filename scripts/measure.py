"""Measure how well quantized 2x2x3 block symbols repeat, and what they cost under the corpus model.

Symbols are pooled from every level of each image's pyramid. Parents are linear-light means; symbols are
quantized in gamma-encoded YCbCr with by bits for Y and bc bits for Cb/Cr. Unseen symbols share the
Good-Turing mass n1/N equally. Costs are ideal entropy-coder code lengths (-log2 p).
"""
import sys
from multiprocessing import Pool

import numpy as np

CONFIGS = [(2, 1), (3, 1), (3, 2), (4, 2), (4, 3), (5, 3)]
N_TRAIN, N_HELD, N_NOISE = 6000, 1000, 200
CACHE = sys.argv[1]


def linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def ycbcr(lin):
    g = linear_to_srgb(lin)
    y = g @ np.array([0.299, 0.587, 0.114])
    return np.stack([y, (g[..., 2] - y) / 1.772, (g[..., 0] - y) / 1.402], axis=-1)


def pyramid(lin):
    levels = [lin.astype(np.float64)]
    while levels[-1].shape[0] > 1:
        a = levels[-1]
        levels.append((a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) / 4)
    return levels[::-1]  # side 1, 2, 4, ..., 256


def children(a):
    """(s, s, 3) -> (s/2 * s/2, 4, 3): each 2x2 block, children ordered TL, TR, BL, BR."""
    return np.stack([a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]], axis=2).reshape(-1, 4, 3)


def radices(variant, by, bc):
    if variant == "abs":
        return np.array([2 ** by, 2 ** bc, 2 ** bc])
    return np.array([2 ** (by + 1) + 1, 2 ** (bc + 1) + 1, 2 ** (bc + 1) + 1])


def quantize(variant, by, bc, child, parent):
    steps = np.array([2 ** by, 2 ** bc, 2 ** bc])
    if variant == "abs":
        q = np.floor((child + [0.0, 0.5, 0.5]) * steps)
        return np.clip(q, 0, steps - 1).astype(np.int64)
    q = np.rint((child - parent[:, None, :]) * steps)
    return (np.clip(q, -steps, steps) + steps).astype(np.int64)


def keys(lin, variant, by, bc):
    """Return (keys, level) for every block in the image's pyramid; level k has side 2^k children."""
    ycc = [ycbcr(l) for l in pyramid(lin)]
    r = np.tile(radices(variant, by, bc), 4)
    weights = np.cumprod(np.concatenate([[1], r[:-1]])).astype(np.uint64)
    out, lev = [], []
    for k in range(1, len(ycc)):
        q = quantize(variant, by, bc, children(ycc[k]), ycc[k - 1].reshape(-1, 3)).reshape(-1, 12)
        out.append((q.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64))
        lev.append(np.full(len(q), k, np.uint8))
    return np.concatenate(out), np.concatenate(lev)


def job(args):
    src, i, variant, by, bc = args
    lin = np.load(CACHE, mmap_mode="r")[i] if src == "corpus" else NOISE[i]
    return keys(np.asarray(lin, np.float64), variant, by, bc)


rng = np.random.default_rng(0)
NOISE = ((rng.random((N_NOISE, 256, 256, 3)) + 0.0) )  # uniform in sRGB...
NOISE = np.where(NOISE <= 0.04045, NOISE / 12.92, ((NOISE + 0.055) / 1.055) ** 2.4)  # ...stored linear

if __name__ == "__main__":
    n = np.load(CACHE, mmap_mode="r").shape[0]
    order = rng.permutation(n)
    train, held = order[:N_TRAIN], order[N_TRAIN:N_TRAIN + N_HELD]
    print("variant by bc | distinct  n1/N   seen% | bits/block real  noise  raw | bits/image real  noise")
    with Pool() as pool:
        for variant in ("abs", "res"):
            for by, bc in CONFIGS:
                tr = np.concatenate([k for k, _ in pool.map(job, [("corpus", i, variant, by, bc) for i in train], 16)])
                uniq, counts = np.unique(tr, return_counts=True)
                N, K = counts.sum(), len(uniq)
                p0 = (counts == 1).sum() / N
                M = float(np.prod(radices(variant, by, bc).astype(float)) ** 4)

                def cost(k):
                    idx = np.clip(np.searchsorted(uniq, k), 0, K - 1)
                    seen = uniq[idx] == k
                    bits = np.where(seen, -np.log2((1 - p0) * counts[idx] / N), -np.log2(p0 / (M - K)))
                    return bits, seen

                real = [cost(k) for k, _ in pool.map(job, [("corpus", i, variant, by, bc) for i in held], 16)]
                noise = [cost(k) for k, _ in pool.map(job, [("noise", i, variant, by, bc) for i in range(N_NOISE)], 16)]
                rb = np.concatenate([b for b, _ in real]); nb = np.concatenate([b for b, _ in noise])
                seen = np.concatenate([s for _, s in real]).mean()
                per_img = len(rb) / N_HELD
                print(f"{variant:3s}     {by}  {bc} | {K:9d} {p0:6.3f} {100 * seen:6.1f}% | "
                      f"{rb.mean():8.2f} {nb.mean():8.2f} {np.log2(M):5.1f} | "
                      f"{rb.mean() * per_img:10.0f} {nb.mean() * per_img:8.0f}", flush=True)
