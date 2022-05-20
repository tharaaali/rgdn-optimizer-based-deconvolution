import tensorflow as tf


def fft(inp, shape=None):
    if shape is None:
        shape = inp.shape
    spectrum = tf.signal.fft2d(inp)
    return spectrum


def ifft(inp, shape=None):
    if shape is None:
        shape = inp.shape

    res = tf.signal.ifft2d(inp)
    return res


def shift(inp, axes=None, inverse=False):
    """Shift image(s) to center
    Args:
        inp(np.ndarray or tf): input images
        axes(tuple): axes to shift across
            beware, if inp is array of images, default shift change the images order
        inverse(bool)
    """
    if inverse:
        return tf.signal.ifftshift(inp, axes=axes)
    else:
        return tf.signal.fftshift(inp, axes=axes)


def ishift(inp, axes=None):
    return shift(inp, axes=axes, inverse=True)


def fft_conv(inp1, inp2, scale_output=True):
    """
    inputs are shaped: [N, N] of [B, N, N]
    """

    inp1_c = tf.cast(inp1, tf.complex64)
    inp2_c = tf.cast(inp2, tf.complex64)

    f1 = fft(inp1_c)
    f2 = fft(inp2_c, shape=inp1.shape)
    convolved = ifft(f1*f2, shape=inp1.shape)
    res = tf.abs(convolved)

    if scale_output:
        assert len(inp1.shape) in [2, 3], f'Unexpected shape {inp1_c.shape}'
        return res * tf.reduce_sum(inp1, axis=[-1, -2], keepdims=True) / tf.reduce_sum(res, axis=[-1, -2], keepdims=True)

    else:
        return res


def scale(f):
    return tf.math.log(1 + tf.abs(f))
