import tensorflow as tf
import tensorflow.math as tf_math
from sweet.util.fft._fft_tf import fft, ifft

# Implementation of Inverse Blur Filter proposed by (3.29)
def inverse_blur_filtering(image, psf, K=0.01, real_output=True):
    assert len(image.shape) == 4
    assert image.shape[-1] == 1
    assert len(psf.shape) == 4

    # shape[0] may be None
    assert image.shape[1:] == psf.shape[1:], \
        f"Expected equal shapes, got: image={image.shape}, psf={psf.shape}"

    image = tf.cast(image[..., 0], tf.complex64)
    psf = tf.cast(psf[..., 0], tf.complex64)

    # shape = image.shape
    otf = fft(psf)
    mtf = tf.abs(otf)

    # normalize MTF & OTF
    mtf_max = tf.reduce_max(mtf, axis=(1,2), keepdims=True)  # change to mtf[0,0] once tested properly
    mtf = tf.cast(mtf / mtf_max, tf.complex64)
    otf = otf / tf.cast(mtf_max, tf.complex64)

    # we manually set values for ration in IBF as {num} / {denum}
    # to be able to handle zero division cases
    num = fft(image) * tf.cast(tf.square(mtf), tf.complex64)
    denum = otf * (tf.square(mtf) + tf.cast(K, tf.complex64))

    ratio = tf.divide(num, denum)
    ratio = tf.where(denum == 0., tf.constant(0., dtype=tf.complex64), ratio)

    # the result is already shifted to center
    result = ifft(ratio)
    if real_output:
        return tf_math.real(result)[..., tf.newaxis]
    else:  # complex output if requested
        return result
