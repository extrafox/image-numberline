"""Evaluate the codec as a map from images to real numbers x in [0, 1] (the code's binary expansion).

Runs on held-out images (permutation indices 6000..6999, same rng as the model builder) plus uniform
sRGB noise. Tests: 1 round trip, 2 fair-coin output, 3 cost, 4 ordering (x vs OKLab L), 5 bit
importance (flip bit k), 6 truncation (PSNR of n-bit prefix), 7 seams (prefix+0111.. vs prefix+1000..),
8 random numbers. Prints a summary per test; writes <out>/eval.json and PNGs to <out>/png/.
ΔE is the per-pixel Euclidean OKLab distance, averaged over the image. PSNR is in linear RGB (peak 1).
"""
import argparse
import json
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib  # noqa: E402

from rh import model  # noqa: E402

codec = None

N_TRAIN, N_HELD = 6000, 1000
_MDL = None


def _init(path, codec_name="rh.codec"):
    global _MDL, codec
    codec = importlib.import_module(codec_name)
    _MDL = codec.load_model(path)


def _decode(bits):
    return np.asarray(codec.decode(_MDL, list(bits), levels=model.LEVELS), np.float64)


def _encode(lin):
    bits, level_bits = codec.encode(_MDL, lin)
    return list(bits), [float(b) for b in level_bits]


def _recon(lin):
    return np.asarray(codec.reconstruct(lin), np.float64)


def log_positions(n):
    """1, 2, 4, ... < n."""
    return [1 << j for j in range(int(n).bit_length()) if (1 << j) < n]


def delta_e(a, b):
    return float(np.linalg.norm(model.oklab_from_linear(a) - model.oklab_from_linear(b), axis=-1).mean())


def psnr(a, b):
    mse = float(((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


def as_real(bits):
    b = np.zeros(53)
    b[:min(53, len(bits))] = bits[:53]
    return float((b * 0.5 ** np.arange(1, 54)).sum())


def rank(a):
    """Ranks 0..n-1 with ties averaged."""
    r = np.empty(len(a))
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, r) / cnt)[inv]


def spearman(a, b):
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    return float(np.corrcoef(ra, rb)[0, 1])


def to_png(lin, path):
    Image.fromarray((model.linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8)).save(path)


def grid(imgs, cols, pad=2):
    s = imgs[0].shape[0]
    rows = -(-len(imgs) // cols)
    out = np.ones((rows * (s + pad) - pad, cols * (s + pad) - pad, 3))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        out[r * (s + pad):r * (s + pad) + s, c * (s + pad):c * (s + pad) + s] = im
    return out


def clean(v):
    """JSON-safe copy: numpy -> python, non-finite floats -> None."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v.item() if isinstance(v, np.generic) else v


def table(header, rows):
    print("  " + header)
    for r in rows:
        print("  " + r)


class Ctx:
    def __init__(self, args, pool):
        self.args, self.pool = args, pool
        self.map = pool.map if pool else lambda f, xs: list(map(f, xs))
        n = np.load(args.cache, mmap_mode="r").shape[0]
        order = np.random.default_rng(0).permutation(n)
        self.held_idx = order[N_TRAIN:N_TRAIN + N_HELD][:args.images]
        data = np.load(args.cache, mmap_mode="r")
        self.held = [np.asarray(data[i], np.float64) for i in self.held_idx]
        rng = np.random.default_rng(args.seed)
        self.noise = [model.srgb_to_linear(rng.random((model.SIDE, model.SIDE, 3))) for _ in range(args.noise)]
        self.rng = rng
        self._enc, self._nenc, self._full = None, None, None

    def enc(self):
        if self._enc is None:
            self._enc = self.map(_encode, self.held)
        return self._enc

    def nenc(self):
        if self._nenc is None:
            self._nenc = self.map(_encode, self.noise)
        return self._nenc

    def full(self):
        if self._full is None:
            self._full = self.map(_decode, [b for b, _ in self.enc()])
        return self._full


def t_roundtrip(c):
    recons = c.map(_recon, c.held)
    diffs = [float(np.abs(d - r).max()) if d.shape == r.shape else float("inf") for d, r in zip(c.full(), recons)]
    ok = sum(d == 0 for d in diffs)
    print(f"  exact {ok}/{len(diffs)}   max|decode-reconstruct| {max(diffs):.3g}")
    to_png(grid([c.held[0], recons[0], c.full()[0]], 3), c.png / "roundtrip.png")
    return {"max_abs_diff": diffs, "exact": ok, "n": len(diffs)}


def t_fair(c):
    b = np.concatenate([np.asarray(bits, np.uint8) for bits, _ in c.enc()])
    d = b - b.mean()
    ac = float((d[:-1] * d[1:]).mean() / d.var())
    packed = np.packbits(b).tobytes()
    z = len(zlib.compress(packed, 9)) / len(packed)
    print(f"  bits {len(b)}   ones {b.mean():.4f}   lag1 autocorr {ac:+.4f}   zlib9/raw {z:.4f}")
    return {"n_bits": int(len(b)), "frac_ones": float(b.mean()), "lag1_autocorr": ac, "zlib_ratio": z}


def t_cost(c):
    px = model.SIDE ** 2
    out = {}
    for name, enc in (("real", c.enc()), ("noise", c.nenc())):
        if not enc:
            continue
        n = np.array([len(b) for b, _ in enc])
        lv = np.array([l for _, l in enc])
        out[name] = {"bits": n.tolist(), "level_bits": lv.tolist(), "median_bits": float(np.median(n)),
                     "mean_bits": float(n.mean()), "bpp": float(n.mean() / px),
                     "mean_level_bits": lv.mean(0).tolist()}
    names = list(out)
    table("            " + "".join(f"{n:>12s}" for n in names),
          [f"{'bits/img':12s}" + "".join(f"{out[n]['mean_bits']:12.0f}" for n in names),
           f"{'bits/px':12s}" + "".join(f"{out[n]['bpp']:12.3f}" for n in names)]
          + [f"{'L' + str(k) + (' root' if k == 0 else ''):12s}" + "".join(f"{out[n]['mean_level_bits'][k]:12.1f}" for n in names)
             for k in range(model.LEVELS + 1)])
    return out


def t_order(c):
    x = [as_real(b) for b, _ in c.enc()]
    L = [float(model.oklab_from_linear(im)[..., 0].mean()) for im in c.held]
    rho = spearman(x, L)
    print(f"  spearman(x, mean OKLab L) = {rho:+.4f}  over {len(x)} images")
    return {"spearman": rho, "x": x, "L": L}


def t_flip(c):
    m = min(c.args.flip_images, len(c.held))
    jobs, keys = [], []
    for i in range(m):
        bits = c.enc()[i][0]
        for k in log_positions(len(bits)) + [len(bits)]:
            f = list(bits)
            f[k - 1] ^= 1
            jobs.append(f)
            keys.append((i, k if k & (k - 1) == 0 else 0))  # 0 = last bit
    outs = c.map(_decode, jobs)
    per = {}
    for (i, k), o in zip(keys, outs):
        per.setdefault(k, []).append(delta_e(o, c.full()[i]))
    ks = sorted(per, key=lambda k: k or np.inf)
    table(f"{'bit k':>9s} {'n':>4s} {'median dE':>10s}",
          [f"{k or 'last':>9} {len(per[k]):4d} {np.median(per[k]):10.4f}" for k in ks])
    return {"k": ks, "median_dE": [float(np.median(per[k])) for k in ks], "dE": {str(k): per[k] for k in ks}}


def t_trunc(c):
    m = min(c.args.trunc_images, len(c.held))
    jobs, keys = [], []
    for i in range(m):
        bits = c.enc()[i][0]
        for n in log_positions(len(bits)):
            jobs.append(bits[:n])
            keys.append((i, n))
    outs = c.map(_decode, jobs)
    per = {}
    for (i, n), o in zip(keys, outs):
        per.setdefault(n, []).append(psnr(o, c.full()[i]))
    ns = sorted(per)
    table(f"{'n bits':>9s} {'imgs':>4s} {'median PSNR':>12s}",
          [f"{n:9d} {len(per[n]):4d} {np.median(per[n]):12.2f}" for n in ns])
    # strip of image 0 at several prefix lengths, then the full decode
    nb = len(c.enc()[0][0])
    strip = [n for n in log_positions(nb) if n in (8, 64, 512, 4096, 16384, 65536, 131072)]
    strip_imgs = [o for (i, n), o in zip(keys, outs) if i == 0 and n in strip] + [c.full()[0]]
    to_png(grid(strip_imgs, len(strip_imgs)), c.png / "truncation.png")
    return {"n": ns, "median_psnr": [float(np.median(per[n])) for n in ns], "psnr": per, "strip_n": strip + [nb]}


def t_seam(c):
    m = min(c.args.seam_images, len(c.held))
    jobs, keys = [], []
    for i in range(m):
        bits = c.enc()[i][0]
        for L in [0] + log_positions(len(bits)):
            p, t = list(bits[:L]), c.args.seam_tail or len(bits) + 64
            jobs += [p + [0] + [1] * t, p + [1] + [0] * t]
            keys.append(L)
    outs = c.map(_decode, jobs)
    per = {}
    for j, L in enumerate(keys):
        per.setdefault(L, []).append(delta_e(outs[2 * j], outs[2 * j + 1]))
    Ls = sorted(per)
    table(f"{'prefix L':>9s} {'n':>4s} {'median dE':>10s} {'max dE':>8s}",
          [f"{L:9d} {len(per[L]):4d} {np.median(per[L]):10.4f} {max(per[L]):8.4f}" for L in Ls])
    a, b = outs[0], outs[1]  # L=0 pair: x just below vs just above 1/2
    to_png(grid([a, b], 2), c.png / "seam_half.png")
    return {"L": Ls, "median_dE": [float(np.median(per[L])) for L in Ls], "dE": {str(L): per[L] for L in Ls}}


def t_random(c):
    n = int(np.median([len(b) for b, _ in c.enc()]))
    streams = [c.rng.integers(0, 2, n).tolist() for _ in range(c.args.random)]
    outs, errs = [], []
    for s in streams:  # serial so one failure doesn't hide the others
        try:
            outs.append(_decode(s) if c.pool is None else c.pool.apply(_decode, (s,)))
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
    ok = [o for o in outs if o.shape == (model.SIDE, model.SIDE, 3) and np.isfinite(o).all()]
    print(f"  {len(ok)}/{len(streams)} decoded to valid {model.SIDE}x{model.SIDE} images (length {n} bits)"
          + (f"; errors: {errs[:3]}" if errs else ""))
    if ok:
        to_png(grid(ok, 4), c.png / "random.png")
    return {"n_bits": n, "ok": len(ok), "n": len(streams), "errors": errs}


TESTS = {1: ("round trip", t_roundtrip), 2: ("fair coin", t_fair), 3: ("cost", t_cost), 4: ("ordering", t_order),
         5: ("bit importance", t_flip), 6: ("truncation", t_trunc), 7: ("seams", t_seam), 8: ("random numbers", t_random)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "cache/yfcc256.npy"))
    ap.add_argument("--codec", default="rh.codec", help="codec module (rh.codec or rh.haar_codec)")
    ap.add_argument("--model", default=str(ROOT / "models/yfcc_y4c2.npz"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--images", type=int, default=50, help="held-out images (tests 1-4; pool for 5-8)")
    ap.add_argument("--noise", type=int, default=10, help="noise images (test 3)")
    ap.add_argument("--flip-images", type=int, default=20, help="images for test 5")
    ap.add_argument("--trunc-images", type=int, default=20, help="images for test 6")
    ap.add_argument("--seam-images", type=int, default=20, help="images for test 7")
    ap.add_argument("--seam-tail", type=int, default=0, help="test 7 tail length (0 = past the code's end)")
    ap.add_argument("--random", type=int, default=16, help="random bitstreams for test 8")
    ap.add_argument("--tests", default="1,2,3,4,5,6,7,8", help="comma-separated test numbers")
    ap.add_argument("--workers", type=int, default=1, help="processes for encode/decode (1 = in-process)")
    ap.add_argument("--seed", type=int, default=1, help="seed for noise images and random bitstreams")
    args = ap.parse_args()
    tests = [int(t) for t in args.tests.split(",")]

    out = Path(args.out)
    (out / "png").mkdir(parents=True, exist_ok=True)
    pool = Pool(args.workers, _init, (args.model, args.codec)) if args.workers > 1 else None
    if pool is None:
        _init(args.model, args.codec)
    c = Ctx(args, pool)
    c.png = out / "png"
    results = {"args": vars(args), "held_idx": c.held_idx.tolist()}
    t_all = time.time()
    try:
        for t in tests:
            name, fn = TESTS[t]
            print(f"[{t}] {name}", flush=True)
            t0 = time.time()
            results[f"{t}_{name.replace(' ', '_')}"] = fn(c)
            print(f"  ({time.time() - t0:.1f}s)", flush=True)
    finally:
        if pool:
            pool.close()
    results["seconds"] = time.time() - t_all
    (out / "eval.json").write_text(json.dumps(clean(results), allow_nan=False))
    print(f"wrote {out / 'eval.json'} and {out / 'png'}/  ({results['seconds']:.0f}s total)")


if __name__ == "__main__":
    main()
