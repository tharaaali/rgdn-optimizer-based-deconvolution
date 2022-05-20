"""Open image dataset"""
from sweet.util.io import to_opt
from website.demo.web_utils import load_img

import numpy as np

from pathlib import Path
import os.path as osp


def load_imgs(count, fs, seed):
    assert len(fs) >= count
    imgs = []

    for f, _ in zip(fs, range(count)):
        r = load_img(osp.abspath(f))
        r = np.tile(np.array(r)[..., np.newaxis], (1,1,3))
        imgs.append(r)

    imgs = np.array([to_opt(x) for x in imgs])
    return imgs

def load_oid_imgs_small(count=10, root='.../../training_datasets/OpenImageDatasetV4/train/0/0', seed=0):
    fs = sorted(list(Path(root).glob('*')))
    return load_imgs(count, fs, seed)

def load_oid_imgs_big(count=10, root='../../training_datasets/OpenImageDatasetV4/train/0/', seed=0):
    fs = sorted(list(Path(root).glob('*/*')))
    return load_imgs(count, fs, seed)

# def load_oid_imgs_huge(count=10, root='../../training_datasets/OpenImageDatasetV4/train/', seed=0):
#     fs = sorted(list(Path(root).glob('*/*/*')))
#     return load_imgs(count, fs, seed)

def load_oid_imgs_auto(count, nn_role, train_root, seed=0):
    if nn_role == 'train':
        root = Path(train_root) / 'a'
    elif nn_role == 'val':
        # using another train dir as val, it has ~100K images, expected to be enough.
        # 0/0 is ~3 times bigger then the a/0 or a/1
        root = Path (train_root) / 'b'
    else:
        raise RuntimeError(f"Unexpected nn_role {nn_role}")

    if count <= 6000:
        # approx 6400 imgs in '0' subdir. No need to download more data in such cases
        return load_oid_imgs_small(count, root / '0', seed)
    else:
        return load_oid_imgs_big(count, root, seed)

# def load_imgs(*args, **kwargs):
#     return load_oid_imgs_auto(*args, **kwargs)
