import math
import numpy as np
from tqdm import tqdm as tqdm
import json
import os
import os.path as osp
import setGPU

import tensorflow as tf


def gauss_conv(kernlen, sigma):
    assert kernlen % 2 == 1
    half = (kernlen - 1) // 2
    conv = np.zeros([kernlen, kernlen], dtype="float32")

    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            conv[i + half, j + half] = np.exp((-(i ** 2) - j ** 2) / (sigma ** 2))

    conv /= np.sum(conv)
    return conv


def gauss_convs_weighted_sum(kernlen, weights, sigmas):
    conv = np.zeros([kernlen, kernlen])
    for w, s in zip(weights, sigmas):
        conv += w * gauss_conv(kernlen, sigma=s)
    conv /= np.sum(conv)
    return conv


def gauss_multikernel(kernlen, viewdist_in_inch, dpi, weights, sigmas):
    assert kernlen % 2 == 1

    va = 1
    sampPerDeg = dpi * viewdist_in_inch * math.tan(va * math.pi / 180)

    sigmas = [np.array(s) * sampPerDeg for s in sigmas]

    multikernel = np.zeros([kernlen, kernlen, 3, 3])
    for i in range(3):
        multikernel[..., i, i] = gauss_convs_weighted_sum(
            kernlen, weights[i], sigmas[i]
        )
    return multikernel


def gauss_sCIElab_multikernel(kernlen, viewdist_in_inch, dpi):
    # weights = [
    #     [1.00327, 0.114416, -0.117686],
    #     [0.616725, 0.383275],
    #     [0.567885, 0.432115],
    # ]
    # sigmas = [
    #     [0.0283, 0.133, 4.336],
    #     [0.0392, 0.494],
    #     [0.0536, 0.386],
    # ]
    weights = [
        [1.00327, 0.11442, -0.11769],
        [0.61673, 0.38328],
        [0.56789, 0.43212],
    ]
    sigmas = [
        [0.05, 0.225, 7.0],
        [0.0686, 0.8260],
        [0.0920, 0.6451],
    ]
    return gauss_multikernel(kernlen, viewdist_in_inch, dpi, weights, sigmas)


def gauss_sCIElab_simplified_multikernel(kernlen, viewdist_in_inch, dpi):
    weights = [[0.5, 0.5]] * 3
    sigmas = [[0.0283, 0.133]] * 3
    return gauss_multikernel(kernlen, viewdist_in_inch, dpi, weights, sigmas)


def cube_root_clipped(vals, min_val=0.001):
    vals_pos = tf.maximum(vals, min_val)
    return tf.pow(vals_pos, 1 / 3.0)


def lab_func(img):
    # img = img * 255

    LAB_X = 0.95047
    LAB_Y = 1.00
    LAB_Z = 1.08883

    fX = cube_root_clipped(img[..., 0] / LAB_X)
    fY = cube_root_clipped(img[..., 1] / LAB_Y)
    fZ = cube_root_clipped(img[..., 2] / LAB_Z)

    L = 116.0 * fY - 16.0
    a = 500.0 * (fX - fY)
    b = 200.0 * (fY - fZ)

    return tf.stack([L, a, b], axis=-1)


XYZ_TO_OPP_MAT = (
    np.array(
        [
            [278.7336, 721.8031, -106.5520],
            [-448.7736, 289.8056, 77.1569],
            [85.9513, -589.9859, 501.1089],
        ]
    )
    / 1000.0
).astype("float32")
OPP_TO_XYZ_MAT = np.linalg.inv(XYZ_TO_OPP_MAT)


def mul_matrixes(imgs, mats, is_vect=False):
    # imgs = ind*x*y*c, mats = x*y*cf*ct
    if is_vect:
        mats = tf.expand_dims(mats, axis=0)
        return mats * imgs
    else:
        imgs = tf.expand_dims(imgs, axis=-2)
        mats = tf.expand_dims(mats, axis=0)
        return tf.reduce_sum(mats * imgs, axis=-1)


def mse(a, b):
    return tf.reduce_mean(tf.square(a - b))


def L2(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b)))


def sqrt_root_clipped(vals, min_val=0.000001):
    vals_pos = tf.maximum(vals, min_val)
    return tf.pow(vals_pos, 1.0 / 2.0)


def mean_L2(a, b):
    return tf.reduce_mean(sqrt_root_clipped(tf.reduce_sum(tf.square(a - b), axis=3)))


def det3(m):
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) +
        m[0, 1] * (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) +
        m[0, 2] * (m[2, 1] * m[1, 0] - m[2, 0] * m[1, 1])
    )


def xyz_to_o1o2o3(imgs):
    mats = XYZ_TO_OPP_MAT[np.newaxis, np.newaxis, ...]
    return mul_matrixes(imgs, mats)


def o1o2o3_to_xyz(imgs):
    mats = OPP_TO_XYZ_MAT[np.newaxis, np.newaxis, ...]
    return mul_matrixes(imgs, mats)


def fixed_conv_padded(imgs, pad_size, kernel):
    padded = tf.pad(
        imgs, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT"
    )
    padded = tf.cast(padded, tf.float32)
    return tf.nn.conv2d(padded, kernel, (1, 1, 1, 1), padding="VALID")


def SCIELAB_representation(img_XYZ, pad_size, kernel):
    img_o1o2o3 = xyz_to_o1o2o3(img_XYZ)
    img_o1o2o3_blur = fixed_conv_padded(img_o1o2o3, pad_size, kernel)
    img_LAB = lab_func(o1o2o3_to_xyz(img_o1o2o3_blur))
    return img_LAB


def _ensure_4_axis(img, strict=True): 
    shape = tf.shape(img)

    if len(shape) == 4: 
        return img
    elif not strict and len(shape) in [2, 3]: 
        if len(shape) == 2:
            return img[np.newaxis, :, :, np.newaxis]           
        else: 
            return img[np.newaxis, :, :]           
    else: 
        raise RuntimeError(f'Unexpected shape {shape}')


class ScielabDist():
    def __init__(self, kernlen, viewdist_in_inch, dpi, strict=True, pad_size=1, norm_coef=1/10000): 
        self.kernlen = kernlen
        self.strict = strict
        self.pad_size = pad_size
        self.norm_coef = norm_coef
        self.kernel = gauss_sCIElab_multikernel(kernlen, viewdist_in_inch, dpi)

    def __call__(self, img1_XYZ, img2_XYZ, mean_res=True): 
        img1_XYZ = _ensure_4_axis(img1_XYZ, self.strict)
        img2_XYZ = _ensure_4_axis(img2_XYZ, self.strict)

        img1_lab = SCIELAB_representation(img1_XYZ, self.pad_size, self.kernel) 
        img2_lab = SCIELAB_representation(img2_XYZ, self.pad_size, self.kernel) 
        
        if mean_res:
            return mse(img1_lab, img2_lab) * self.norm_coef
        else: 
            return tf.square(img1_lab - img2_lab) * self.norm_coef
