from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn

from tensorflow.python.ops.image_ops_impl import _fspecial_gauss

from .tf import prepare_type
import numpy as np

    # c1 = (k1 * max_val)**2
    # c2 = (k2 * max_val)**2

    # # SSIM luminance measure is
    # # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    # mean0 = reducer(x)
    # mean1 = reducer(y)
    # num0 = mean0 * mean1 * 2.0
    # den0 = math_ops.square(mean0) + math_ops.square(mean1)
    # luminance = (num0 + c1) / (den0 + c1)

    # # SSIM contrast-structure measure is
    # #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    # #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    # #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    # num1 = reducer(x * y) * 2.0
    # den1 = reducer(math_ops.square(x) + math_ops.square(y))
    # c2 *= compensation
    # cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    # # SSIM score is the product of the luminance and contrast-structure measures.
    # return luminance, cs
        


def ssim_manual(x, y, max_val=1.0, compensation=1.0, k=0.03, filter_size=11, filter_sigma=1.5, mean_res=True):
    # image1 = prepare_type(image1, check_types=check_types)
    # image2 = prepare_type(image2, check_types=check_types)

    
    
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    
    def reducer(x):
        shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
        y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="VALID")
        return array_ops.reshape(
            y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0)
        )


    r"""Helper function for computing SSIM.
    SSIM estimates covariances with weighted sums.  The default parameters
    use a biased estimate of the covariance:
    Suppose `reducer` is a weighted sum, then the mean estimators are
    \mu_x = \sum_i w_i x_i,
    \mu_y = \sum_i w_i y_i,
    where w_i's are the weighted-sum weights, and covariance estimator is
    cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    with assumption \sum_i w_i = 1. This covariance estimator is biased, since
    E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
    For SSIM measure with unbiased covariance estimators, pass as `compensation`
    argument (1 - \sum_i w_i ^ 2).
    Arguments:
    x: First set of images.
    y: Second set of images.
    reducer: Function that computes 'local' averages from the set of images. For
        non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
        for convolutional version, this is usually tf.nn.avg_pool2d or
        tf.nn.conv2d with weighted-sum kernel.
    max_val: The dynamic range (i.e., the difference between the maximum
        possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
    Returns:
    A pair containing the luminance measure, and the contrast-structure measure.
    """

    # r - regularizer
    # m0 = E(x), m1 = E(y)
    # s0 = (E((x-m0)**2))**0.5
    # num0 = E(x) * E(y)
    # den0 = E(x)*E(x) + E(y)*E(y)
    # num1 = E(xy) 
    # den1 = E(x*x + y*y)

    # num1 - num0 = E(xy) - E(x)E(y) = E((x-Ex)(y-Ey))
    # den1 - den0 = [E(xx) - E(x)E(x)] + [same for y] = E((x-Ex)(x-Ex)) + [same for y] = (without reg) s0**2 + s1**2


    r = (k * max_val)**2

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = math_ops.square(mean0) + math_ops.square(mean1)
    luminance = (num0 + r/9.) / (den0 + r/9.)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = reducer(x * y) * 2.0
    den1 = reducer(math_ops.square(x) + math_ops.square(y))
    # cs = (num1 - num0 + r) / (den1 - den0 + r)


    # print(x.shape, y.shape)
    assert filter_size % 2 == 1
    p = filter_size // 2
    # s0 = math_ops.sqrt(reducer(math_ops.square(x[:, p:-p, p:-p, :] - mean0)))
    # s1 = math_ops.sqrt(reducer(math_ops.square(y[:, p:-p, p:-p, :] - mean1)))

    s0 = math_ops.sqrt(nn.relu(reducer(x*x) - mean0*mean0))
    s1 = math_ops.sqrt(nn.relu(reducer(y*y) - mean1*mean1))


    # contrast
    # (2 * sigma(x) * sigma(y)) / (cov_{xx} + cov_{yy} + c2)
    # structure
    # (2 * cov_{xy} + c2) / (2 * sigma(x) * sigma(y))
    contrast = (2*s0*s1 + r) / (den1 - den0 + r)
    structure = (num1 - num0 + r) / (2*s0*s1 + r)

    # SSIM score is the product of the luminance and contrast-structure measures.
    ssim_parts = luminance, contrast, structure

    if mean_res: 
        return [math_ops.reduce_mean(p) for p in ssim_parts]
    else: 
        return ssim_parts


def ssim_compl_luminance(x,y,**args): 
    return 1 - ssim_manual(x[np.newaxis, ..., np.newaxis], y[np.newaxis, ..., np.newaxis], **args)[0]

def ssim_compl_contrast(x,y,**args): 
    return 1 - ssim_manual(x[np.newaxis, ..., np.newaxis], y[np.newaxis, ..., np.newaxis], **args)[1]

def ssim_compl_structure(x,y,**args): 
    return 1 - ssim_manual(x[np.newaxis, ..., np.newaxis], y[np.newaxis, ..., np.newaxis], **args)[2]
    


def _ssim_per_channel(
    img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
):
    """Computes SSIM index between img1 and img2 per color channel.
    This function matches the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.
    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.
    Args:
      img1: First image batch.
      img2: Second image batch.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Default value 11 (size of gaussian filter).
      filter_sigma: Default value 1.5 (width of gaussian filter).
      k1: Default value 0.01
      k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
    Returns:
      A pair of tensors containing and channel-wise SSIM and contrast-structure
      values. The shape is [..., channels].
    """
    filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)

    shape1, shape2 = array_ops.shape_n([img1, img2])
    checks = [
        control_flow_ops.Assert(
            math_ops.reduce_all(math_ops.greater_equal(shape1[-3:-1], filter_size)),
            [shape1, filter_size],
            summarize=8,
        ),
        control_flow_ops.Assert(
            math_ops.reduce_all(math_ops.greater_equal(shape2[-3:-1], filter_size)),
            [shape2, filter_size],
            summarize=8,
        ),
    ]

    # Enforce the check to run before computation.
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)

    # TODO(sjhwang): Try to cache kernels and compensation factor.
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    compensation = 1.0

    # TODO(sjhwang): Try FFT.
    # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
    #   1-by-n and n-by-1 Gaussian filters instead of an n-by-n filter.
    def reducer(x):
        shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
        y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="VALID")
        return array_ops.reshape(
            y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0)
        )

    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1, k2)

#     viz(luminance[0, ..., 0], cs[0, ..., 0], titles=['luminance', 'cs'])
    # Average over the second and the third from the last: height, width.
    
#     print(luminance.shape, cs.shape)
    
    
    axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
    ssim_val = math_ops.reduce_mean(luminance * cs, axes)
    cs = math_ops.reduce_mean(cs, axes)
    return ssim_val, cs