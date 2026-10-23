"""Go/no-go check for the semantic prefix: do similar images share longer prefixes?

On held-out images, compares the semantic tree's path bits with the pixel codec's bits (rh.snake_codec):
  - Spearman correlation between shared-prefix length and pair similarity, measured two ways: CLIP
    cosine (what the tree is built from) and, independently, OKLab ΔE between 32x32 thumbnails;
  - mean shared prefix with each image's nearest neighbour versus a random partner.
Also writes results/semantic/branches.png (depth-3 branches: prototype, then members) and prints where
text prompts land on the line.

Usage: semantic_eval.py <tree.npz> [images | prompts]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rh import model as m  # noqa: E402

CACHE = ROOT / "cache/yfcc256.npy"


def shared_prefix(a, b):
    n = min(len(a), len(b))
    d = np.nonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0]
    return int(d[0]) if len(d) else n


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(v)).astype(float) for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def snake_bits(i):
    from rh import snake_codec as sc
    mdl = sc.load_model(str(ROOT / "models/yfcc_haar.npz"))
    return sc.encode(mdl, np.asarray(np.load(CACHE, mmap_mode="r")[i], np.float64))[0][:256]


def thumb(i, side=32):
    a = np.asarray(np.load(CACHE, mmap_mode="r")[i], np.float64)
    f = 256 // side
    return a.reshape(side, f, side, f, 3).mean(axis=(1, 3))


def x_of(bits):
    return sum(b / 2 ** (k + 1) for k, b in enumerate(bits))


def prompt_report(t, emb, paths):
    """Where text prompts land on the line, and how many bits they share with their closest images."""
    # Where do text prompts land? CLIP's text and image embeddings occupy different regions (the
    # modality gap), so route each prompt through its images: the mean embedding of its 50 closest.
    for prompt, e in zip(t["prompts"], t["texts"].astype(np.float32)):
        e = emb[np.argsort(-(emb @ e))[:50]].mean(axis=0)
        node, bits = 0, []
        for _ in range(int(t["depth"])):
            bit = int((e @ t["dirs"][node] > t["thr"][node]) != t["flip"][node])
            bits.append(bit)
            node = 2 * node + 1 + bit
        close = np.argsort(-(emb @ (t["texts"][list(t["prompts"]).index(prompt)].astype(np.float32))))[:200]
        agree = np.mean([shared_prefix(paths[i], bits) for i in close])
        print(f"  {prompt:28s} x ∈ [{x_of(bits):.4f}, {x_of(bits) + 2 ** -len(bits):.4f})  "
              f"prefix shared with its 200 closest images: {agree:4.1f} bits")


if __name__ == "__main__":
    t = np.load(sys.argv[1])
    emb = np.load(ROOT / "cache/clip_vitb32.npz")["images"].astype(np.float32)
    if len(sys.argv) > 2 and sys.argv[2] == "prompts":
        prompt_report(t, emb, t["paths"])
        sys.exit()
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n = len(emb)
    held = np.random.default_rng(0).permutation(n)[6000:6000 + k]
    paths = t["paths"]
    with Pool(6) as pool:
        codes = pool.map(snake_bits, held.tolist())
        thumbs = np.array(pool.map(thumb, held.tolist()))
    lab = m.oklab_from_linear(thumbs)
    iu = np.triu_indices(k, 1)
    cos = (emb[held] @ emb[held].T)[iu]
    de = np.array([np.linalg.norm(lab[i] - lab[j], axis=-1).mean() for i, j in zip(*iu)])
    sem = np.array([shared_prefix(paths[held[i]], paths[held[j]]) for i, j in zip(*iu)])
    pix = np.array([shared_prefix(codes[i], codes[j]) for i, j in zip(*iu)])
    print(f"{k} held-out images, {len(cos)} pairs")
    print("                         semantic prefix   pixel code (snake)")
    print(f"mean shared prefix bits  {sem.mean():15.2f}   {pix.mean():18.2f}")
    print(f"spearman vs CLIP cosine  {spearman(sem, cos):15.3f}   {spearman(pix, cos):18.3f}")
    print(f"spearman vs -ΔE(thumbs)  {spearman(sem, -de):15.3f}   {spearman(pix, -de):18.3f}")
    sim = emb[held] @ emb[held].T
    np.fill_diagonal(sim, -2)
    nn = sim.argmax(axis=1)
    rnd = np.random.default_rng(1).permutation(k)
    for name, bits in (("semantic", [paths[i] for i in held]), ("pixel", codes)):
        a = np.mean([shared_prefix(bits[i], bits[nn[i]]) for i in range(k)])
        b = np.mean([shared_prefix(bits[i], bits[rnd[i]]) for i in range(k) if rnd[i] != i])
        print(f"{name:8s}: shared prefix with CLIP nearest neighbour {a:5.2f} bits vs random partner {b:5.2f}")

    # Branch mosaics at depth 3: prototype, then up to 9 training members, one row per branch.
    depth, side, gap = 3, 128, 4
    held_set = set(np.random.default_rng(0).permutation(n)[6000:7000].tolist())
    rows = []
    for b in range(2 ** depth):
        prefix = [(b >> (depth - 1 - j)) & 1 for j in range(depth)]
        node = 0
        for bit in prefix:
            node = 2 * node + 1 + bit
        members = [i for i in range(n) if i not in held_set and list(paths[i][:depth]) == prefix][:9]
        proto = np.asarray(t["protos"][node], np.float64)
        row = [np.kron(proto, np.ones((side // proto.shape[0], side // proto.shape[0], 1)))] + [thumb(i, side) for i in members]
        rows.append(row)
    cols = max(len(r) for r in rows)
    grid = np.ones((len(rows) * (side + gap) - gap, cols * (side + gap) - gap, 3))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            grid[r * (side + gap):r * (side + gap) + side, c * (side + gap):c * (side + gap) + side] = m.linear_to_srgb(img)
    out = ROOT / "results/semantic"
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(grid, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out / "branches.png")
    print(f"wrote {out / 'branches.png'} (rows: depth-{depth} branches in line order; first column = prototype)")
    prompt_report(t, emb, paths)
