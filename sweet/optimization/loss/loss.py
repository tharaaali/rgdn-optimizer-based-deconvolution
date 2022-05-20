"""loss factory"""
import tensorflow as tf
from sweet.optimization.tf import ssim_cs_compl, l2, l4, stress, l2_normed, ssim_compl, ms_ssim_compl, cor


def selected_loss(a, b, check_types=True):
    return l2_normed(a, b, check_types) + 20*stress(a, b, check_types) + 5 - 5*cor(a, b, check_types)


def selected_loss_params(a, b, check_types=True, wl=1.0, ws=20.0, wc=5.0):
    return wl*l2_normed(a, b, check_types) + ws*stress(a, b, check_types) + wc - wc*cor(a, b, check_types)


def l2_3ssim_cs(a,b, check_types=True):
    # mse_3ssim_cs
    return l2(a, b, check_types) + 3*ssim_cs_compl(a, b, check_types)


def mse_3ssim_cs(a,b, check_types=True):
    # mse_3ssim_cs
    return l2(a, b, check_types) + 3*ssim_cs_compl(a, b, check_types)


def l4_3ssim_cs(a,b, check_types=True):
    return l4(a, b, check_types) + 3*ssim_cs_compl(a, b, check_types)


import numpy as np
from sweet.util.fft._fft_tf import fft_conv, shift
def _build(k):
    ar = (np.arange(512, dtype=np.float32) - 511/2) / k
    ar = np.exp(-ar**2)
    ar = ar[:, np.newaxis] * ar[np.newaxis, :]
    ar = shift(ar)
    ar = ar / np.sum(ar)
    ar = ar[np.newaxis, ...]
    return ar

ar128 = _build(128)
ar64 = _build(64)
ar32 = _build(32)

from sweet.optimization.tf import prepare_type

def ignore_bg_preprocessed_l2_3ssimcs(a,b, check_types=True):
    a = prepare_type(a, check_types=check_types)
    b = prepare_type(b, check_types=check_types)
    avg_a = a[..., 0] - fft_conv(a[..., 0], ar128)
    avg_b = b[..., 0] - fft_conv(b[..., 0], ar128)
    return l2_3ssim_cs(avg_a[..., np.newaxis], avg_b[..., np.newaxis], check_types)

def ignore_bg_preprocessed_l4_2ssimcs(a, b, check_types=True):
    a = prepare_type(a, check_types=check_types)
    b = prepare_type(b, check_types=check_types)
    avg_a = a[..., 0] - fft_conv(a[..., 0], ar64)
    avg_b = b[..., 0] - fft_conv(b[..., 0], ar64)
    return l4(avg_a[..., np.newaxis], avg_b[..., np.newaxis], check_types) + 2*ssim_cs_compl(a, b, check_types)


def ms_ssim_compl_gamma(a, b, check_types=True):
    return ms_ssim_compl(a**(1/2.2), b**(1/2.2), check_types=check_types)

LOSS_FUNC_BY_NAME = {
    'ssim_cs_compl': ssim_cs_compl,
    'ssim_compl': ssim_compl,
    'l2': l2,
    'l2_normed': l2_normed,
    'stress': stress,
    'l2_3ssim_cs': l2_3ssim_cs,
    'l4_3ssim_cs': l4_3ssim_cs,
    'mse_3ssim_cs': mse_3ssim_cs,
    'ib_l2s3': ignore_bg_preprocessed_l2_3ssimcs,
    'ib_l4s3': ignore_bg_preprocessed_l4_2ssimcs,
    'selected_loss': selected_loss,
    'selected_loss_params': selected_loss_params,
    'ms_ssim_compl': ms_ssim_compl,
    'ms_ssim_compl_gamma': ms_ssim_compl_gamma,
}


class FuncLoss(tf.keras.losses.Loss):
    def __init__(self, **config):
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name="CustomFuncLoss")
        config = config.copy()
        self.__name__ = f'Loss FL'
        loss_name = config.pop('name')
        self.gamma = config.pop('gamma', False)
        self.ignore_bg = config.pop('ignore_bg', False)
        self.check_types = config.pop('check_types', True)



        assert len(config) == 0, f"Error! Unused params: {config}"


        self.loss = LOSS_FUNC_BY_NAME[loss_name]
        self.config = config

        # print(self.config)

    def _prepare_inp(self, x):
        if self.ignore_bg:
            x = (x[..., 0] - fft_conv(x[..., 0], ar128))[..., np.newaxis]

        if self.gamma:
            x = x ** (1/2.2)

        return x

    def call(self, a, b):
        a = self._prepare_inp(a)
        b = self._prepare_inp(b)

        return self.loss(a, b, check_types=self.check_types, **self.config)



class LossFactory():
    def make(self, config, check_types=True):
        obj_type = config.pop('type')
        if obj_type in ['selected_loss_params', 'FL']:
            return FuncLoss(check_types=check_types, **config)

        elif obj_type in LOSS_FUNC_BY_NAME:
            return lambda a, b: LOSS_FUNC_BY_NAME[obj_type](a, b, check_types=check_types)

        else:
            raise NotImplementedError(f"Unexpected type {obj_type}")