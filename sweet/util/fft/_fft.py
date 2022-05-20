import numpy as np


def fft(inp, shape=None):
    if shape is None:
        shape = inp.shape
    spectrum = np.fft.fft2(inp, s=shape, norm='ortho')
    return spectrum


def ifft(inp, shape=None):
    if shape is None:
        shape = inp.shape

    res = np.fft.ifft2(inp, s=shape, norm='ortho')
    return res


def shift(inp, inverse=False):
    if inverse:
        return np.fft.ifftshift(inp)
    else:
        return np.fft.fftshift(inp)


def ishift(inp):
    return shift(inp, inverse=True)


def fft_conv(inp1, inp2, scale_output=True):
    if inp1.shape != inp2.shape:
        kernel = shift(inp2)
        sz = (inp1.shape[0] - kernel.shape[0], inp1.shape[1] - kernel.shape[1])  # total amount of padding
        kernel = np.pad(
            kernel,
            (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)),
            'constant'
        )
        inp2 = ishift(kernel)

    f1 = fft(inp1)
    f2 = fft(inp2, shape=inp1.shape)
    convolved = ifft(f1 * f2, shape=inp1.shape)
    res = np.abs(convolved)

    if scale_output:
        return res * np.sum(inp1) / np.sum(res)
    else:
        return res


def scale(f):
    return np.log(1 + np.abs(f))
