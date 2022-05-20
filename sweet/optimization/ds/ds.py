"""Dataset (images + PSFs) factory"""

from sweet.draw import draw_char, draw_char_multiscale
from sweet.transform.psf import PSF
from sweet.sweet import Sweet
from sweet.util.config import calc_recommended_opt_size
from ._oid import load_oid_imgs_auto
from ._mdi import load_mdi_imgs
from tqdm import tqdm as tqdm
from sweet.util.data import defaults


import numpy as np
import tensorflow as tf

import random
from pathlib import Path

class SimpleDS():
    def __init__(self, **args):
        self.kers = []

        sweet = Sweet()
        # sweet.set_eye_prescription(A=3*np.pi/7, C=-1.5, S=-7.25)
        sweet.set_eye_participant('WADB', 'L')
        sweet.set_experiment_params(pupil_diam=3, view_dist=100)
        self.kers.append(sweet._psf()[..., np.newaxis])

        for S in [-7, -6]:
            sweet.set_eye_prescription(A=3*np.pi/7, C=-1.5, S=S)
            self.kers.append(sweet._psf()[..., np.newaxis])


        self.data = [draw_char('E', size=args['N'], char_ratio=0.4).astype(np.float32)[..., np.newaxis]] * len(self.kers)

        self.data = np.array(self.data)
        self.kers = np.array(self.kers)

    def __getitem__(self, i):
        return self.data[i], self.kers[i]

    def __len__(self):
        return len(self.data)

    def get_some_val_data(self, count=None):
        return self.data, self.kers


def load_OID_MDI(
    groups_size={'OID_train': 1000, 'OID_val': 200, 'MDI_train': 1000, 'MDI_val': 200},
    train_seed=0,
    debug_mode=False
):
    """load std imgs groups

    Parameters
    ----------
    groups_size: dict
        how many images to use in a dataset
    train_seed: int
        seed for random in train ds part

    Returns
    ----------
    np.array
        train imgs
    np.array
        val imgs
    """
    if debug_mode:
        groups_size = {k:v//100 for k,v in groups_size.items()}


    rand = random.Random(train_seed)
    td_path = Path(__file__).resolve().parents[3] / 'data' / 'training_datasets'


    train_imgs = []
    train_imgs += list(load_oid_imgs_auto(count=groups_size['OID_train'], nn_role='train', train_root=str(td_path / 'OpenImageDatasetV4/train/'), seed=rand.randint(0, 2**32)))
    train_imgs += list(load_mdi_imgs(count=groups_size['MDI_train'], root=str(td_path / 'material-design-icons/train/'), seed=rand.randint(0, 2**32)),)
    rand.shuffle(train_imgs)
    train_imgs = np.array(train_imgs)[..., np.newaxis]

    rand = random.Random(1)  # the validation is nor randomized
    val_imgs = []
    # using only a small fraction of OID. some (other) train subdir is used for the validation
    val_imgs += list(load_oid_imgs_auto(count=groups_size['OID_val'], nn_role='val', train_root=str(td_path / 'OpenImageDatasetV4/train/'), seed=rand.randint(0, 2**32)))
    val_imgs += list(load_mdi_imgs(count=groups_size['MDI_val'], root=str(td_path / 'material-design-icons/val/'), seed=rand.randint(0, 2**32)),)

    rand.shuffle(val_imgs)
    val_imgs = np.array(val_imgs)[..., np.newaxis]

    return train_imgs, val_imgs


def generate_random_kernels_v1(count, seed, debug_mode=False, viz=False):
    """generate kernels (PSFs) for random eyes

    Parameters
    ----------
    count: int
        how many kers to generate
    seed: int
        seed for random
    debug_mode: bool
        print the debug info about generated eyes, optimal eye size parameters

    Returns
    ----------
    np.array
        kernels array with shape: (count,  N, N, 1)
    """

    rand = random.Random(seed)

    params = {
        # 'S': {'l': -7, 'r': -4},
        'S': {'l': -11, 'r': -3},

        'C': {'l': -3, 'r': 0},
        'A': {'l': 0, 'r': 180},
    }

    kers = []

    range_count = range(count)
    if viz:
        range_count = tqdm(range_count)

    for i in range_count:
        sweet = Sweet()
        eye_params = {
            k: v['l'] + rand.random() * (v['r'] - v['l'])
            for k, v in params.items()
        }
        sweet.set_eye_prescription(**eye_params)

        D0 = 3.0
        dist=100.0
        # size = 10.0

        sph_eq = eye_params['S'] + (eye_params['C'] / 2)
        rec_canv_corrected = defaults['k_ratio'] * calc_recommended_opt_size(spherical_equivalent=sph_eq, D0=D0*1.7, view_dist=dist)
        size = rec_canv_corrected * 2**(2*rand.random() - 1)

        sweet.set_experiment_params(pupil_diam=D0, view_dist=dist, canvas_size_h=size)
        kers.append(sweet._psf()[..., np.newaxis])

        if debug_mode:
            sph_eq = eye_params['S'] + (eye_params['C'] / 2)
            rec_canv = defaults['k_ratio'] * calc_recommended_opt_size(spherical_equivalent=sph_eq, D0=D0, view_dist=dist)

            print(f"Generated ker with params {eye_params}")
            print(f"   SphEq = {sph_eq:.2f}")
            print(f"   CanvSize = {size/rec_canv_corrected:.2f} * best")
            print(f"   (CanvSize is {size}. Formula gives {rec_canv:.2f}. Optimal is ~{rec_canv_corrected:.2f} given D0 is usually 1.5-2 times smaller)")
            print()

    return np.array(kers)


class TFDS():
    def __init__(self, seed, viz=False, debug_mode=False):
        self.train_imgs, self.val_imgs = load_OID_MDI(debug_mode=debug_mode, train_seed=seed)

        # kers_train_count = 997 # too slow
        kers_train_count = 97
        if debug_mode:
            kers_train_count //= 30
        self.train_kers = generate_random_kernels_v1(kers_train_count, seed=seed, viz=viz, debug_mode=debug_mode)
        # train kers depend on initialization

        # kers_val_count = 397 # too slow
        kers_val_count = 37
        if debug_mode:
            kers_val_count //= 30
        self.val_kers = generate_random_kernels_v1(kers_val_count, seed=2, viz=viz, debug_mode=debug_mode)
        # seeds are different to make different kers in a same function

    def get_datasets(self, batch_size):
        def _to_shuffled(x):
            return

        def _prepare(imgs, kers):
            # generate same images simultaneously
            imgs_ds = tf.data.Dataset.from_tensor_slices(imgs).shuffle(len(imgs), seed=1, reshuffle_each_iteration=True).repeat()
            imgs_ds_copy = tf.data.Dataset.from_tensor_slices(imgs).shuffle(len(imgs), seed=1, reshuffle_each_iteration=True).repeat()
            kers_ds = tf.data.Dataset.from_tensor_slices(kers).shuffle(len(kers), seed=1, reshuffle_each_iteration=True).repeat()

            zipped = tf.data.Dataset.zip((
                tf.data.Dataset.zip((imgs_ds, kers_ds)), # x
                imgs_ds_copy, # y
            ))

            return zipped.batch(batch_size), len(imgs) // batch_size

        train, train_steps = _prepare(self.train_imgs, self.train_kers)
        val, val_steps = _prepare(self.val_imgs, self.val_kers)
        return train, train_steps, val, val_steps


    def get_some_val_data(self, count=None):
        if count is None:
            count = min(
                len(self.val_imgs),
                len(self.val_kers),
                20,
            )

        assert (count <= len(self.val_imgs)) and (count <= len(self.val_kers))
        return self.val_imgs[:count], self.val_kers[:count]



NAME2CLASS = {
    'SimpleDS': SimpleDS,
    'TFDS': TFDS,
}

class DSFactory():
    def make(self, config):
        config = config.copy()
        obj_type = config.pop('type')

        if obj_type in NAME2CLASS:
            return NAME2CLASS[obj_type](**config)
        elif obj_type == 'CachedTFDS':
            # optimization hack
            import pickle
            with open('/dev/shm/TFDS_seed1.pickle', 'rb') as f:
                ds = pickle.load(f)
            return ds
        else:
            raise NotImplementedError(f"Unexpected type {obj_type}")
