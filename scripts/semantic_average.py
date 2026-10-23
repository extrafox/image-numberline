"""Average images on the semantic number line (rh.semantic_codec).

For held-out pairs, encodes A and B, averages x exactly and shows, per row:
A | B | dyadic meet (medoid of the smallest semantic branch containing both: the digits xA and xB
share) | semantic mean (medoid of the leaf the mean's semantic bits select) | full decode of the
mean | pixel average. Pairs are "same kind" (each image with its CLIP nearest held-out neighbour)
and random. Reports the CLIP cosine of the semantic midpoint to A and B against a random-photo
baseline, and writes walks from A to B at semantic resolution (medoids of evenly spaced x).

Usage: semantic_average.py [pairs per kind]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from average import lerp_bits, mean_bits, save_grid  # noqa: E402
from rh import semantic_codec as sem  # noqa: E402

CACHE, MODEL = ROOT / "cache/yfcc256.npy", ROOT / "models/yfcc_haar.npz"
WALK_STEPS = 8
_MDL = None


def init():
    global _MDL
    _MDL = sem.load_model(str(MODEL))


def job(pair):
    arr = np.load(CACHE, mmap_mode="r")
    a, b = (np.asarray(arr[i], np.float64) for i in pair)
    ba, bb = sem.encode(_MDL, a)[0], sem.encode(_MDL, b)[0]
    bm = mean_bits(ba, bb)
    node = sem.semantic_node(bm, _MDL.depth)
    shared = next((j for j in range(_MDL.depth) if ba[j] != bb[j]), _MDL.depth)
    meet = sem.semantic_node(ba, shared)
    walk = [sem.medoid(_MDL, sem.semantic_node(lerp_bits(ba, bb, k, WALK_STEPS), _MDL.depth)) for k in range(WALK_STEPS + 1)]
    ra, rb = sem.reconstruct(a), sem.reconstruct(b)
    row = [ra, rb, sem.medoid(_MDL, meet), sem.medoid(_MDL, node), sem.decode(_MDL, bm), (ra + rb) / 2]
    return row, (meet, node), [ra] + walk[1:-1] + [rb]


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tree = np.load(ROOT / "models/semantic_tree.npz")
    emb = np.load(ROOT / "cache/clip_vitb32.npz")["images"].astype(np.float32)
    n = len(emb)
    held = np.random.default_rng(0).permutation(n)[6000:7000]
    rng = np.random.default_rng(3)
    sim = emb[held] @ emb[held].T
    np.fill_diagonal(sim, -2)
    anchors = rng.choice(len(held), k, replace=False)
    pairs = {"same kind": [(held[i], held[sim[i].argmax()]) for i in anchors],
             "random": [tuple(p) for p in rng.choice(held, (k, 2), replace=False)]}
    out = ROOT / "results/semantic"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(min(6, k), init) as pool:
        for name, ps in pairs.items():
            results = pool.map(job, [tuple(map(int, p)) for p in ps])
            cos = lambda nodes: np.mean([(emb[tree["medoid"][nd]] @ emb[a] + emb[tree["medoid"][nd]] @ emb[b]) / 2 for nd, (a, b) in zip(nodes, ps)])
            base = np.mean(emb[rng.choice(n, 2000)] @ emb[np.array(ps).ravel()].T)
            ab = np.mean([emb[a] @ emb[b] for a, b in ps])
            print(f"{name:9s}: CLIP cos(A,B) {ab:.3f} | to A,B: meet {cos([r[1][0] for r in results]):.3f}, "
                  f"semantic mean {cos([r[1][1] for r in results]):.3f}, random photo {base:.3f}")
            tag = name.replace(" ", "_")
            save_grid([r for r, _, _ in results], out / f"average_{tag}.png")
            save_grid([w for _, _, w in results[:4]], out / f"walk_{tag}.png")
    print(f"wrote {out}/average_*.png (A | B | meet | semantic mean | full decode of mean | pixel average) and walk_*.png")
