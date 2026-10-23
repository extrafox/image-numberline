"""Build the semantic prefix: a binary tree over CLIP embeddings whose every split halves probability.

Each node splits its training images at the median of their projection onto the node's top principal
direction, so each semantic bit carries exactly one bit and siblings are conceptually alike. Children
are oriented for the line: at the root the darker side comes first (x near 0 is dark); below it, the
child closer to the region immediately to its left on the line comes first, so neighbouring branches
are similar (the Hilbert idea at the semantic level).

Usage: semantic_tree.py <embeddings.npz> <out.npz> [depth]
Writes directions/thresholds/flip per node (heap order: node n has children 2n+1, 2n+2), each
cache image's path bits, per-node prototype images (64x64 linear-light means of member images) and
per-node medoids (the member closest to the node's mean embedding; index and a 128x128 sRGB copy).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rh import haar as h  # noqa: E402
from rh import model as m  # noqa: E402

DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 10
PROTO_SIDE, MEDOID_SIDE = 64, 128


def image_key(lin):
    """Identify an image by its 8-bit sRGB pixels (what the lossless codec actually codes)."""
    return hashlib.sha1(h.srgb8_from_linear(lin).astype(np.uint8).tobytes()).hexdigest()


def principal(e):
    c = e - e.mean(axis=0)
    return np.linalg.svd(c, full_matrices=False)[2][0]


def build(emb, lum, members):
    """Heap-ordered split parameters for a complete tree of DEPTH levels."""
    n_nodes = 2 ** DEPTH - 1
    dirs, thr, flip = np.zeros((n_nodes, emb.shape[1]), np.float32), np.zeros(n_nodes, np.float32), np.zeros(n_nodes, bool)
    node_members = {0: members}
    centroid = {0: emb[members].mean(axis=0)}
    left_neighbour = {0: None}  # centroid of the region just before this node on the line
    for n in range(n_nodes):
        idx = node_members[n]
        d = principal(emb[idx])
        p = emb[idx] @ d
        t = np.median(p)
        lo, hi = idx[p <= t], idx[p > t]
        if n == 0:
            first_is_hi = lum[hi].mean() < lum[lo].mean()  # darker side first
        elif left_neighbour[n] is not None:
            first_is_hi = np.linalg.norm(emb[hi].mean(0) - left_neighbour[n]) < np.linalg.norm(emb[lo].mean(0) - left_neighbour[n])
        else:
            first_is_hi = False
        dirs[n], thr[n], flip[n] = d, t, first_is_hi
        first, second = (hi, lo) if first_is_hi else (lo, hi)
        a, b = 2 * n + 1, 2 * n + 2
        node_members[a], node_members[b] = first, second
        centroid[a], centroid[b] = emb[first].mean(0), emb[second].mean(0)
        left_neighbour[a], left_neighbour[b] = left_neighbour[n], centroid[a]
    return dirs, thr, flip


def path(e, dirs, thr, flip):
    """Semantic bits for one embedding: bit 0 = first child on the line."""
    n, bits = 0, []
    for _ in range(DEPTH):
        side_hi = e @ dirs[n] > thr[n]
        bit = int(side_hi != flip[n])
        bits.append(bit)
        n = 2 * n + 1 + bit
    return bits


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    emb = z["images"].astype(np.float32)
    arr = np.load(ROOT / "cache/yfcc256.npy", mmap_mode="r")
    n = len(arr)
    assert len(emb) == n, "embeddings must be in cache order"
    held = np.random.default_rng(0).permutation(n)[6000:7000]  # held-out split, as in evaluate.py
    members = np.setdiff1d(np.arange(n), held)
    small = np.empty((n, PROTO_SIDE, PROTO_SIDE, 3), np.float32)  # box-downsampled linear images
    keys = []
    f = 256 // PROTO_SIDE
    for i in range(n):
        lin = np.asarray(arr[i], np.float64)
        small[i] = lin.reshape(PROTO_SIDE, f, PROTO_SIDE, f, 3).mean(axis=(1, 3))
        keys.append(image_key(lin))
    lum = small.mean(axis=(1, 2)) @ np.array([0.2126, 0.7152, 0.0722])
    dirs, thr, flip = build(emb, lum, members)
    paths = np.array([path(e, dirs, thr, flip) for e in emb], np.uint8)
    # Prototypes: mean member image of every node (members = training images whose path passes through it).
    protos = np.zeros((2 ** (DEPTH + 1) - 1, PROTO_SIDE, PROTO_SIDE, 3), np.float32)
    counts = np.zeros(len(protos))
    for i in members:
        node = 0
        protos[node] += small[i]
        counts[node] += 1
        for b in paths[i]:
            node = 2 * node + 1 + int(b)
            protos[node] += small[i]
            counts[node] += 1
    protos /= np.maximum(counts, 1)[:, None, None, None]
    # Medoids: the training member nearest each node's mean embedding.
    node_members = [[] for _ in range(len(protos))]
    for i in members:
        node = 0
        node_members[node].append(i)
        for b in paths[i]:
            node = 2 * node + 1 + int(b)
            node_members[node].append(i)
    medoid = np.array([mm[int(np.argmax(emb[mm] @ emb[mm].mean(axis=0)))] for mm in node_members])
    f = 256 // MEDOID_SIDE
    medoid_imgs = np.array([(m.linear_to_srgb(np.asarray(arr[i], np.float64).reshape(MEDOID_SIDE, f, MEDOID_SIDE, f, 3).mean(axis=(1, 3))) * 255 + 0.5).astype(np.uint8) for i in medoid])
    np.savez(sys.argv[2], dirs=dirs, thr=thr, flip=flip, paths=paths, keys=np.array(keys), protos=protos.astype(np.float16),
             medoid=medoid, medoid_imgs=medoid_imgs, counts=counts, depth=DEPTH, texts=z["texts"], prompts=z["prompts"])
    leaf = counts[2 ** DEPTH - 1:]
    print(f"depth {DEPTH}: {len(members)} tree images, leaves {leaf.min():.0f}..{leaf.max():.0f} members")
