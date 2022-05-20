import tensorflow as tf
import numpy as np
from tqdm import tqdm as tqdm
from sweet.util.fft._fft_tf import fft_conv
from sweet.util.viz.viz import viz
from tensorflow.python.ops.image_ops_impl import _ssim_per_channel

import warnings


# def l2_loss(images, res_images):
#     return tf.reduce_sum(tf.square(target_image - res))


# def ssim_loss(images, target_images=image_h, verbose=False):
#     return tf.image.ssim(
#         res[np.newaxis, ..., np.newaxis],
#         tf.constant(target_image[np.newaxis, ..., np.newaxis]),
#         max_val=1
#     )), tf.float32)

#     if verbose:
#         print(f'ssim_loss={ssim_loss}, reg_loss={reg_loss}')

#     # ssim: bigger-better
#     return 10 - ssim_loss*10

def convert_to_4d(image):
    if len(image.shape) == 2:
        image = image[None, ..., None]

    elif len(image.shape) == 3:
        warnings.warn('Adding dimension is ambigous')
        image = image[None, ...]

    return image


def prepare_type(image, check_types=True):
    if check_types:
        assert type(image) is not np.ndarray
    else:
        # convert to tensor? https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/framework/ops.py#L1508
        if type(image) is np.ndarray:
            image = tf.constant(image.astype(np.float32))

    if len(image.shape) != 4:
        warnings.warn(f"Shape {image.shape} is deprecated. Use 4d (batch x H x W x F)", DeprecationWarning)
        image = convert_to_4d(image)

    return image



def l2(image1, image2, check_types=True):
    image1 = prepare_type(image1, check_types=check_types)
    image2 = prepare_type(image2, check_types=check_types)

    return tf.reduce_mean(tf.square(image1-image2), axis=(1,2,3)) # no sqrt, mse actually, not l2

def l4(image1, image2, check_types=True):
    image1 = prepare_type(image1, check_types=check_types)
    image2 = prepare_type(image2, check_types=check_types)

    return  tf.sqrt(tf.sqrt(tf.reduce_mean(tf.square(tf.square(image1-image2)), axis=(1,2,3))))


def l2_normed(ref_image, exp_image, check_types=True):
    ref = prepare_type(ref_image, check_types=check_types)
    exp = prepare_type(exp_image, check_types=check_types)
    dd = tf.reduce_sum(tf.square(ref - exp), axis=(1,2,3))
    rr = tf.reduce_sum(tf.square(ref), axis=(1,2,3))
    return tf.sqrt(dd/rr)


def stress(a, b, check_types=True, r=1e-4):
    a = prepare_type(a, check_types=check_types)
    b = prepare_type(b, check_types=check_types)

    aa = tf.reduce_sum(tf.square(a), axis=(1,2,3))
    bb = tf.reduce_sum(tf.square(b), axis=(1,2,3))
    ab = tf.reduce_sum(a*b, axis=(1,2,3))
    return tf.sqrt(1 - ab*ab/(aa*bb + r))


def cor(a,b, check_types=True, r=1e-4):
    a = prepare_type(a, check_types=check_types)
    b = prepare_type(b, check_types=check_types)

    a = a - tf.reduce_mean(a, axis=(1,2,3))
    b = b - tf.reduce_mean(b, axis=(1,2,3))
    return tf.reduce_sum(a*b, axis=(1,2,3)) / (tf.sqrt(tf.reduce_sum(a*a, axis=(1,2,3)) * tf.reduce_sum(b*b, axis=(1,2,3)) + r))


def ssim_compl(image1, image2, check_types=True, **args):
    image1 = prepare_type(image1, check_types=check_types)
    image2 = prepare_type(image2, check_types=check_types)

    return 1 - tf.image.ssim(
        image1,
        image2,
        max_val=1,
        **args,
    )


def ms_ssim_compl(image1, image2, check_types=True, **args):
    image1 = prepare_type(image1, check_types=check_types)
    image2 = prepare_type(image2, check_types=check_types)

    return 1 - tf.image.ssim_multiscale(
        image1,
        image2,
        max_val=1,
        **args,
    )


def ssim_cs_compl(image1, image2, check_types=True, add_axis=False, **args):
    image1 = prepare_type(image1, check_types=check_types)
    image2 = prepare_type(image2, check_types=check_types)

    if add_axis:
        warnings.warn(f"Params add_axis={add_axis} is deprecated.", DeprecationWarning)
    #     image1 = image1[np.newaxis, ..., np.newaxis]
    #     image2 = image2[np.newaxis, ..., np.newaxis]

    _, cs = _ssim_per_channel(
        image1,
        image2,
        **args,
    )
    return 1 - cs




def optimize(
    start_img,
    loss_func,
    optimizer = 'Adam',
    clip = 'False',
    epochs = 100,
):
    if optimizer == 'SGD':
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1., decay_steps=50, decay_rate=.1)
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1., decay_steps=50, decay_rate=.1)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    var = tf.Variable(start_img.astype(np.float32))
    for i in tqdm(range(epochs)):
        with tf.GradientTape() as tape:
            loss = loss_func(var)
            gr = tape.gradient(loss, var)
        opt.apply_gradients([[gr, var]])

        if clip:
            var.assign(
                tf.clip_by_value(var, 0, 1)
            )

    return var.numpy()


def optimize_image(
    start_img = np.random.random([1024, 1024]),
    goal_imgs = None,
    loss_func = l2,  # actually distance function, but it's usually called loss
    process_func = lambda x: x,
    **args
):
    goal = tf.constant(goal_imgs.astype(np.float32))
    var_loss_func = lambda x: loss_func(process_func(x), goal)

    return optimize(
        start_img,
        var_loss_func,
        **args
    )


class Trainer():
    def __init__(self, variables, loss_func, optimizer, callbacks=[], clippers=[]):
        self.vars = [tf.Variable(x) for x in variables]
        self.opt = optimizer
        self.loss_func = loss_func
        self.callbacks = callbacks
        self.clippers = clippers


    def train(self, epochs):
        for i in tqdm(range(epochs)):
            with tf.GradientTape() as tape:
                loss = self.loss_func(self.vars)
                grs = tape.gradient(loss, self.vars)

            self.opt.apply_gradients([[gr, var] for (gr, var) in zip(grs, self.vars)])

            for callback in self.callbacks:
                callback(i, self.vars)

            for clip in self.clippers:
                for var in self.vars:
                    var.assign(clip(var))


        return [var.numpy() for var in self.vars]


class MetricL2SSIM():
    def __init__(self, coef=1.):
        self.coef = coef

    def __call__(self, inp1, inp2):
        return l2(inp1, inp2) + self.coef*ssim_compl(inp1, inp2)


class ImageTransformLoss():
    def __init__(self, kers, goals, metric_func):
        self.kers = tf.constant(kers)
        self.goals = tf.constant(goals)
        self.metric_func = metric_func

    def __call__(self, imgs):
        assert len(imgs) == 1
        images = tf.tile(imgs[0][..., 0], [len(self.kers), 1, 1])
        conved = fft_conv(images, self.kers[..., 0])[..., np.newaxis]
        loss = self.metric_func(conved, self.goals)
        return loss


class VizualizeSometimes():
    def __init__(self, step=1):
        self.step=step

    def __call__(self, i, variables):
        if i % self.step == 0:
            viz(variables[0][0])


class ClipVar():
    def __init__(self, l=0., r=1.):
        self.l = l
        self.r = r

    def __call__(self, var):
        return tf.clip_by_value(var, self.l, self.r)
