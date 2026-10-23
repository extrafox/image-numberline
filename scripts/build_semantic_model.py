"""Per-branch statistics for the semantic codec: root colours and coarse node symbols of the first
COND_LEVELS levels, counted within each semantic branch at depth K (K bits of the semantic prefix).

Branch tables back off to the global tables: root colours mix branch counts with BETA pseudo-counts
of the global distribution; coarse symbols unseen in the branch escape (Good-Turing n1) to the global
coarse table. K is chosen by ideal held-out code length over the conditioned symbols.

Usage: build_semantic_model.py <tree.npz> <out.npz>
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rh import haar as h  # noqa: E402

COND_LEVELS, BETA, KS = 3, 32, (0, 2, 3, 4, 5, 6)
CACHE = ROOT / "cache/yfcc256.npy"


def symbols(i):
    _, rq, _, nq = h.decompose(np.asarray(np.load(CACHE, mmap_mode="r")[i], np.float64))
    return h.root_index(rq), np.concatenate([h.coarse_keys(q) for q in nq[:COND_LEVELS]])


def branch(paths, k):
    return (paths[:, :k].astype(np.int64) * (1 << np.arange(k - 1, -1, -1))).sum(axis=1) if k else np.zeros(len(paths), np.int64)


def tables(roots, keys, ctx, k):
    out = []
    for c in range(2 ** k):
        sel = ctx == c
        kk, cc = np.unique(np.concatenate(keys[sel]) if sel.any() else np.zeros(0, np.uint64), return_counts=True)
        out.append((np.bincount(roots[sel], minlength=int(np.prod(h.ROOT_RADIX))), kk, cc))
    return out


def ideal_bits(tab, g_root, g_keys, g_counts, g_escape, roots, keys, ctx):
    """Held-out -log2 p of the conditioned symbols (root + COND_LEVELS levels) under branch tables."""
    g_p = (g_root + 1) / (g_root + 1).sum()
    g_total = g_counts.sum() + g_escape
    g_lookup = dict(zip(g_keys.tolist(), g_counts.tolist()))
    total = 0.0
    for r, ks, c in zip(roots, keys, ctx):
        rc, kk, cc = tab[c]
        total -= np.log2((rc[r] + BETA * g_p[r]) / (rc.sum() + BETA))
        local = dict(zip(kk.tolist(), cc.tolist()))
        esc = max(1, int((cc == 1).sum()))
        l_total = cc.sum() + esc
        for key in ks.tolist():
            if key in local:
                total -= np.log2(local[key] / l_total)
                continue
            total -= np.log2(esc / l_total) if len(kk) else 0.0
            gc = g_lookup.get(key)
            total -= np.log2(gc / g_total) if gc else np.log2(g_escape / g_total) - np.log2(float(np.prod(h.NODE_RADIX)))
    return total / len(roots)


if __name__ == "__main__":
    tree = np.load(sys.argv[1])
    g = np.load(ROOT / "models/yfcc_haar.npz")
    n = len(tree["paths"])
    perm = np.random.default_rng(0).permutation(n)
    train, held = perm[:6000], perm[6000:6300]
    with Pool(6) as pool:
        tr = pool.map(symbols, train.tolist(), 32)
        he = pool.map(symbols, held.tolist(), 32)
    tr_roots, tr_keys = np.array([r for r, _ in tr]), np.array([k for _, k in tr], dtype=object)
    he_roots, he_keys = np.array([r for r, _ in he]), np.array([k for _, k in he], dtype=object)
    best = None
    print(" K  branches  held-out bits (root + levels 1-3)")
    for k in KS:
        tab = tables(tr_roots, tr_keys, branch(tree["paths"][train], k), k)
        bits = ideal_bits(tab, g["root_counts"], g["keys"], g["counts"], int(g["escape"]), he_roots, he_keys, branch(tree["paths"][held], k))
        print(f"{k:2d} {2 ** k:9d} {bits:10.1f}  (+{k} semantic-context bits are already in the prefix)")
        if best is None or bits < best[1]:
            best = (k, bits, tab)
    k, _, tab = best
    np.savez(sys.argv[2], k=k, beta=BETA, cond_levels=COND_LEVELS, root_counts=np.stack([t[0] for t in tab]),
             keys=np.concatenate([t[1] for t in tab]), counts=np.concatenate([t[2] for t in tab]),
             offsets=np.cumsum([0] + [len(t[1]) for t in tab]))
    print(f"chose K={k}")
