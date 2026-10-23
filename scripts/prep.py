"""Centre-crop each image to a square and box-downsample in linear light to 256x256.

Writes a float16 linear-light cache: out.npy with shape (n, 256, 256, 3), plus out.txt listing sources.
Images whose short side is under 256 are skipped (no upsampling).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

SIDE = 256


def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def load(path):
    try:
        im = Image.open(path)
        im.draft("RGB", (SIDE, SIDE))
        im = im.convert("RGB")
    except Exception:
        return None
    w, h = im.size
    s = min(w, h)
    if s < SIDE:
        return None
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    lin = srgb_to_linear(np.asarray(im, dtype=np.float32) / 255.0)
    chans = [Image.fromarray(lin[..., c], mode="F").resize((SIDE, SIDE), Image.BOX) for c in range(3)]
    return np.stack([np.asarray(c) for c in chans], axis=-1).astype(np.float16)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    paths = sorted(src.rglob("*.jpg"))
    with Pool() as pool:
        results = pool.map(load, paths, chunksize=32)
    kept = [(p, r) for p, r in zip(paths, results) if r is not None]
    np.save(out.with_suffix(".npy"), np.stack([r for _, r in kept]))
    out.with_suffix(".txt").write_text("\n".join(str(p) for p, _ in kept) + "\n")
    print(f"kept {len(kept)} of {len(paths)}")
