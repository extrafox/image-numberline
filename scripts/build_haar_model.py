"""Count integer-Haar symbols over the training split: root colours, coarse node symbols pooled across
levels (escape frequency n1, Good-Turing), and per-pass refinement symbols by context. Same split as
scripts/build_model.py.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rh import haar as h  # noqa: E402

N_TRAIN = 6000
CACHE = sys.argv[1]


def tables():
    root = {j: np.zeros((2 ** int((h.SHIFT >= j).sum()),) * 2, np.int64) for j in range(1, h.PASSES + 1)}
    node = {j: np.zeros((2 ** int((h.NODE_SHIFT >= j).sum()),) * 2, np.int64) for j in range(1, h.PASSES + 1)}
    return root, node


def job(indices):
    arr = np.load(CACHE, mmap_mode="r")
    roots, keys, vsum = [], [], np.zeros(9)
    root_ref, node_ref = tables()
    for i in indices:
        root, rq, nodes, nq = h.decompose(np.asarray(arr[i], np.float64))
        roots.append(h.root_index(rq))
        keys.append(h.coarse_keys(np.concatenate(nq)))
        v, q = np.concatenate(nodes), np.concatenate(nq)
        vsum += v.sum(axis=0)
        for j in range(1, h.PASSES + 1):
            s, c, _ = h.refinement(root[None], rq[None], h.SHIFT, j)
            root_ref[j][c[0], s[0]] += 1
            s, c, nb = h.refinement(v, q, h.NODE_SHIFT, j)
            node_ref[j] += np.bincount(c * 2 ** nb + s, minlength=4 ** nb).reshape(2 ** nb, 2 ** nb)
    return roots, np.concatenate(keys), root_ref, node_ref, vsum


if __name__ == "__main__":
    n = np.load(CACHE, mmap_mode="r").shape[0]
    train = np.random.default_rng(0).permutation(n)[:N_TRAIN]
    with Pool() as pool:
        results = pool.map(job, np.array_split(train, 120))
    root_ref, node_ref = tables()
    for _, _, r, nd, _ in results:
        for j in root_ref:
            root_ref[j] += r[j]
            node_ref[j] += nd[j]
    root_counts = np.bincount(sum((r for r, *_ in results), []), minlength=int(np.prod(h.ROOT_RADIX)))
    keys, counts = np.unique(np.concatenate([k for _, k, *_ in results]), return_counts=True)
    ysum = h.keys_to_coarse(keys)[:, :3].sum(axis=1)
    order = np.lexsort((keys, ysum))
    escape = max(1, int((counts == 1).sum()))
    np.savez(sys.argv[2], root_counts=root_counts, keys=keys[order], counts=counts[order], escape=escape,
             value_mean=sum(r[4] for r in results) / counts.sum(),
             **{f"root_ref_{j}": t for j, t in root_ref.items()}, **{f"node_ref_{j}": t for j, t in node_ref.items()})
    print(f"{len(keys)} distinct coarse node symbols, N={counts.sum()}, n1/N={escape / counts.sum():.4f}")
