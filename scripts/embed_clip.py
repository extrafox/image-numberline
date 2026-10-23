"""Embed corpus images (and a few text prompts) with OpenCLIP, in the order of the local cache.

Runs on the machine holding the images (we use the dev box). Input: a list of image paths, one per
line, in cache order (cache/yfcc256.txt). Each image is centre-cropped to a square, matching
scripts/prep.py, then CLIP-preprocessed. Output: <out>.npz with unit-norm float16 `images` (n, d),
`texts` (m, d) and the `prompts` themselves.

Usage: embed_clip.py <paths.txt> <out.npz> [threads]
"""
import sys

import numpy as np
import open_clip
import torch
from PIL import Image

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
PROMPTS = ["a photo of a beach", "a portrait of a person", "a photo of food", "a city street at night",
           "a cat", "a dog", "a mountain landscape", "a crowd at a concert", "a car", "a flower",
           "a group photo of friends", "a screenshot", "a black and white photo", "a sunset", "a building",
           "the inside of a house"]
BATCH = 64


def square(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


if __name__ == "__main__":
    paths = [p for p in open(sys.argv[1]).read().split("\n") if p]
    torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            batch = torch.stack([preprocess(square(p)) for p in paths[i:i + BATCH]])
            f = model.encode_image(batch)
            feats.append((f / f.norm(dim=-1, keepdim=True)).numpy().astype(np.float16))
            print(f"{min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
        t = model.encode_text(open_clip.get_tokenizer(MODEL)(PROMPTS))
        t = (t / t.norm(dim=-1, keepdim=True)).numpy().astype(np.float16)
    np.savez(sys.argv[2], images=np.concatenate(feats), texts=t, prompts=np.array(PROMPTS))
    print("done", np.concatenate(feats).shape)
