"""Semantic codec: x = [semantic prefix][pixel code].

The first DEPTH bits are the image's path in the semantic tree (scripts/semantic_tree.py): each splits
its branch at the median of a principal direction of CLIP embeddings, so each is an exact halving of
probability and the line is ordered by kind. The pixel code is rh.snake_codec's, unconditioned:
conditioning the root colour and first coarse levels on the branch saved under 1 bit per image
(scripts/build_semantic_model.py), less than its cost. A prefix that ends inside the semantic bits
decodes to the branch medoid, the corpus photo nearest the branch's mean embedding.

Same API as rh.codec. load_model takes the pixel model path and expects semantic_tree.npz beside it.
Images not in the embedded corpus (e.g. noise) have no embedding here, so the encoder gives them the
all-zeros path; any path is decodable.
"""
import hashlib
from pathlib import Path

import numpy as np

from rh import haar as h
from rh import model as m
from rh import snake_codec as sc

BIT = [0, 1, 2]  # cumulative frequencies of a fair bit


def image_key(lin):
    return hashlib.sha1(h.srgb8_from_linear(lin).astype(np.uint8).tobytes()).hexdigest()


class Model:
    def __init__(self, pixel_path):
        self.px = sc.Model(np.load(pixel_path))
        tree = np.load(Path(pixel_path).parent / "semantic_tree.npz")
        self.depth = int(tree["depth"])
        self.paths = {key: p.tolist() for key, p in zip(tree["keys"].tolist(), tree["paths"])}
        self.medoids = tree["medoid_imgs"]


def load_model(path):
    return Model(path)


def encode(mdl, lin):
    path = mdl.paths.get(image_key(lin), [0] * mdl.depth)
    st = sc._Stream()
    for b in path:
        st.put(BIT, b)
    level_bits = sc.encode_pixels(mdl.px, st, lin, mdl.px.root, [mdl.px.coarse] * h.LEVELS)
    level_bits[0] += mdl.depth
    return st.enc.finish(), level_bits


def decode(mdl, bits, levels=h.LEVELS):
    assert levels == h.LEVELS, "only full decodes are supported"
    ls = sc._Lockstep(bits)
    node, path = 0, []
    for _ in range(mdl.depth):
        a, b = ls.get(BIT)
        if a != b:  # prefix ends inside the semantic bits: show the branch medoid
            return medoid(mdl, node)
        path.append(a)
        node = 2 * node + 1 + a
    return sc.decode_pixels(mdl.px, ls, mdl.px.root, [mdl.px.coarse] * h.LEVELS)


def medoid(mdl, node):
    """The branch medoid at full size, linear light."""
    img = mdl.medoids[node].astype(np.float64) / 255.0
    f = 2 ** h.LEVELS // img.shape[0]
    return m.srgb_to_linear(np.kron(img, np.ones((f, f, 1))))


def semantic_node(bits, depth):
    """Deepest semantic-tree node fixed by a bit list's first depth bits."""
    node = 0
    for b in bits[:depth]:
        node = 2 * node + 1 + int(b)
    return node


def reconstruct(lin):
    return sc.reconstruct(lin)
