"""Count closed-loop root and residual symbols over the training split, pooled across pyramid levels.

Writes an .npz with root counts, residual keys in line order (by total Y change), their counts, and the
escape frequency n1 (Good-Turing: unseen mass ~ n1/N). Same split as scripts/measure.py.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rh import model as m  # noqa: E402

N_TRAIN = 6000
CACHE = sys.argv[1]


def job(i):
    root, keys, _ = m.symbols(np.asarray(np.load(CACHE, mmap_mode="r")[i], np.float64))
    return root, np.concatenate(keys)


if __name__ == "__main__":
    n = np.load(CACHE, mmap_mode="r").shape[0]
    train = np.random.default_rng(0).permutation(n)[:N_TRAIN]
    with Pool() as pool:
        results = pool.map(job, train, 16)
    root_counts = np.bincount([r for r, _ in results], minlength=int(np.prod(m.ROOT_RADIX)))
    keys, counts = np.unique(np.concatenate([k for _, k in results]), return_counts=True)
    ysum = m.keys_to_digits(keys)[:, [0, 3, 6, 9]].sum(axis=1)
    order = np.lexsort((keys, ysum))
    escape = max(1, int((counts == 1).sum()))
    np.savez(sys.argv[2], root_counts=root_counts, keys=keys[order], counts=counts[order], escape=escape)
    print(f"{len(keys)} distinct residual symbols, N={counts.sum()}, n1/N={escape / counts.sum():.4f}")
