import numpy as np


def decode_rle(rle, size):
    h, w = size
    mask = np.zeros(h*w, dtype=np.float32)
    if len(rle) > 0:
        rle = rle.reshape(-1, 2)
        runs, lengths = rle[:, 0], rle[:, 1]
        for r, n in zip(runs, lengths):
            mask[r:r+n] = 1.0
    return mask.reshape(h, w)

def encode_rle(mask):
    mask = mask.reshape(-1)
    values, runs, lengths = np.unique(mask, return_index=True, return_counts=True)
    positive_cluster = values == 1
    return np.stack((runs[positive_cluster], lengths[positive_cluster])).astype(np.uint31).reshape(-1)


def _prefix_dictionary_keys(d, prefix=''):
    return d.__class__([(prefix + k, v) for k, v in d.items()])
