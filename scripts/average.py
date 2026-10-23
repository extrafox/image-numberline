"""Average images on the number line: encode two held-out images, average their real values exactly,
decode the result. Writes results/average/average_<codec>.png, one row per run:
A | B | decode((xA + xB) / 2) | pixel average, and walk_<codec>.png: decodes of x evenly spaced from
xA to xB for the first few runs.

Usage: average.py [runs] [seed] [codec module, default rh.haar_codec].
"""
import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rh import model as m  # noqa: E402

CACHE, MODEL = ROOT / "cache/yfcc256.npy", ROOT / "models/yfcc_haar.npz"
WALK_STEPS, WALK_RUNS = 8, 4
codec = None


def to_int(bits):
    return int("".join(map(str, bits)) or "0", 2)


def mean_bits(a, b):
    """Exact (x_a + x_b) / 2 for x = 0.b1b2... as a bit list (one bit longer than the longer input)."""
    n = max(len(a), len(b))
    s = (to_int(a) << (n - len(a))) + (to_int(b) << (n - len(b)))  # both over 2^n; the sum over 2^(n+1) is the mean
    return [int(c) for c in format(s, f"0{n + 1}b")]


def x_float(bits):
    return to_int(bits[:53]) / 2.0 ** min(53, len(bits))


def lerp_bits(a, b, k, steps):
    """Exact xA + (k / steps) (xB - xA), as bits (a few bits longer than the longer input)."""
    n = max(len(a), len(b)) + steps.bit_length()
    xa, xb = to_int(a) << (n - len(a)), to_int(b) << (n - len(b))
    x = (xa * (steps - k) + xb * k) // steps
    return [int(c) for c in format(x, f"0{n}b")]


def oklab_de(p, q):
    return np.linalg.norm(m.oklab_from_linear(p) - m.oklab_from_linear(q), axis=-1).mean()


def init(name):
    global codec
    codec = importlib.import_module(name)


def job(args):
    run, pair = args
    mdl = codec.load_model(str(MODEL))
    arr = np.load(CACHE, mmap_mode="r")
    a, b = (np.asarray(arr[i], np.float64) for i in pair)
    ba, bb = codec.encode(mdl, a)[0], codec.encode(mdl, b)[0]
    bm = mean_bits(ba, bb)
    mid = codec.decode(mdl, bm)
    ra, rb = codec.reconstruct(a), codec.reconstruct(b)
    pix = (ra + rb) / 2
    walk = [codec.decode(mdl, lerp_bits(ba, bb, k, WALK_STEPS)) for k in range(WALK_STEPS + 1)] if run < WALK_RUNS else []
    stats = (x_float(ba), x_float(bb), x_float(bm), oklab_de(ra, rb), oklab_de(mid, ra), oklab_de(mid, rb), oklab_de(mid, pix))
    return [ra, rb, mid, pix], stats, walk


def save_grid(rows, path, side=256, gap=4):
    grid = np.ones((len(rows) * (side + gap) - gap, len(rows[0]) * (side + gap) - gap, 3))
    for r, imgs in enumerate(rows):
        for c, img in enumerate(imgs):
            grid[r * (side + gap):r * (side + gap) + side, c * (side + gap):c * (side + gap) + side] = m.linear_to_srgb(img)
    Image.fromarray((np.clip(grid, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    name = sys.argv[3] if len(sys.argv) > 3 else "rh.haar_codec"
    n = np.load(CACHE, mmap_mode="r").shape[0]
    held = np.random.default_rng(0).permutation(n)[6000:7000]  # held-out split, as in evaluate.py
    pairs = np.random.default_rng(seed).choice(held, (runs, 2), replace=False).tolist()
    with Pool(min(runs, 6), init, (name,)) as pool:
        results = pool.map(job, list(enumerate(pairs)))
    print(" run     x_A      x_B   x_mean | ΔE(A,B)  ΔE(mid,A)  ΔE(mid,B)  ΔE(mid,pixel avg)")
    for r, (_, (xa, xb, xm, dab, dma, dmb, dmp), _) in enumerate(results):
        print(f"{r:4d} {xa:8.5f} {xb:8.5f} {xm:8.5f} | {dab:7.3f} {dma:9.3f} {dmb:10.3f} {dmp:12.3f}")
    s = np.array([r[1][3:] for r in results])
    print(f"mean                       | {s[:, 0].mean():7.3f} {s[:, 1].mean():9.3f} {s[:, 2].mean():10.3f} {s[:, 3].mean():12.3f}")
    out = ROOT / "results/average"
    out.mkdir(parents=True, exist_ok=True)
    tag = name.split(".")[-1]
    save_grid([r[0] for r in results], out / f"average_{tag}.png")
    save_grid([r[2] for r in results if r[2]], out / f"walk_{tag}.png")
    print(f"wrote {out}/average_{tag}.png (A | B | decode of mean x | pixel average) and walk_{tag}.png (xA -> xB)")
