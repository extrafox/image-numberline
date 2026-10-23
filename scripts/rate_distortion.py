"""Quality vs bits for each codec against the same reference: the held-out image itself.

For each image, decode n-bit prefixes of its code (n log-spaced, plus the full code) and report median
PSNR (linear RGB, peak 1) and mean OKLab ΔE versus the original. Usage: rate_distortion.py [images].
"""
import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rh import model as m  # noqa: E402

CODECS = {"residual": ("rh.codec", "models/yfcc_y4c2.npz"), "haar": ("rh.haar_codec", "models/yfcc_haar.npz")}
LENGTHS = [2 ** k for k in range(8, 21)]


def job(args):
    name, i = args
    mod, path = CODECS[name]
    codec = importlib.import_module(mod)
    mdl = codec.load_model(str(ROOT / path))
    lin = np.asarray(np.load(ROOT / "cache/yfcc256.npy", mmap_mode="r")[i], np.float64)
    bits, _ = codec.encode(mdl, lin)
    rows = []
    for n in [n for n in LENGTHS if n < len(bits)] + [len(bits)]:
        out = codec.decode(mdl, bits[:n])
        mse = ((np.clip(out, 0, 1) - lin) ** 2).mean()
        de = np.linalg.norm(m.oklab_from_linear(out) - m.oklab_from_linear(lin), axis=-1).mean()
        rows.append((n, 10 * np.log10(1 / max(mse, 1e-12)), de, n == len(bits)))
    return name, rows


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n = np.load(ROOT / "cache/yfcc256.npy", mmap_mode="r").shape[0]
    held = np.random.default_rng(0).permutation(n)[6000:6000 + k]
    with Pool(6) as pool:
        results = pool.map(job, [(c, int(i)) for c in CODECS for i in held])
    for name in CODECS:
        rows = [r for c, rs in results if c == name for r in rs]
        print(f"\n{name}:      bits   images  PSNR dB   ΔE")
        for n in LENGTHS:
            sel = [r for r in rows if r[0] == n and not r[3]]
            if sel:
                print(f"  {n:>14d} {len(sel):6d} {np.median([r[1] for r in sel]):8.2f} {np.median([r[2] for r in sel]):7.4f}")
        full = [r for r in rows if r[3]]
        print(f"  full {np.median([r[0] for r in full]):>9.0f} {len(full):6d} {np.median([r[1] for r in full]):8.2f} {np.median([r[2] for r in full]):7.4f}")
