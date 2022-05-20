"""Material design icons dataset"""
from sweet.util.data import defaults
from sweet.draw.helper import pad
from sweet.util.io import to_opt

import numpy as np
import imageio

import glob
from pathlib import Path
import random

def load_mdi_imgs(count=10, root='../../training_datasets/material-design-icons/train/', seed=0):
    fs = sorted(list(Path(root).glob('*.png')))

    rand = random.Random(seed)
    rand.shuffle(fs)

    icons = []
    for f, _ in zip(fs, range(count)):
        r = imageio.imread(f)
        p = 255 - np.tile(pad(r[:, :, 3], (512, 512))[..., np.newaxis], (1,1,3))
        icons.append(p)

    icons = np.array([to_opt(x) for x in icons])
    return icons

def load_imgs(*args, **kwargs):
    return load_mdi_imgs(*args, **kwargs)