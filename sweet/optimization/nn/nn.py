"""Neural network factory"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Conv2DTranspose, Lambda, Concatenate, MaxPooling2D, UpSampling2D, Flatten, Dense
from sweet.optimization.ibf_tf import inverse_blur_filtering
from sweet.util.fft._fft_tf import fft_conv, fft
from tensorflow.keras import regularizers


##############################################################################################
# Util

def clip01(x):
    return tf.clip_by_value(x, 0, 1)


def single_input_ibf(args):
    return inverse_blur_filtering(*args)


def single_input_fft_conv(args, scale_output=True):
    arg1, arg2 = args
    assert arg1.shape[-1] == 1
    assert arg2.shape[-1] == 1

    arg1 = arg1[..., 0]
    arg2 = arg2[..., 0]

    res = fft_conv(arg1, arg2, scale_output=scale_output)
    # have reduced interface
    # todo: convert all the interfaces to 4d (or all to 3d)

    return res[..., tf.newaxis]


def single_input_fft_conv_no_scale_output(args):
    return single_input_fft_conv(args, scale_output=False)


class IBF_Trainable_K(tf.keras.layers.Layer):
    def __init__(self):
        super(IBF_Trainable_K, self).__init__()
        self.K = self.add_weight(
            shape=(1, 512, 512), initializer=tf.keras.initializers.Constant(0.01), trainable=True
        )

    def call(self, inputs):
        assert len(inputs) == 2
        return inverse_blur_filtering(inputs[0], inputs[1], K=tf.nn.relu(self.K) + 0.00001)


def init_inps(N=512):
    inp_img = Input((N, N, 1))
    inp_ker = Input((N, N, 1))
    return inp_img, inp_ker


def preprocess_inputs(inps, norm_kers=True, concatenate_res=True):
    inp_img, inp_ker = inps
    kers = inp_ker
    if norm_kers:
        kers = Lambda(norm_sum)(kers)

    pre = Lambda(single_input_ibf)([inp_img, kers])
    cur = pre

    if concatenate_res is True:
        pre_clip = Lambda(clip01)(pre)
        ibf_modelled = Lambda(single_input_fft_conv_no_scale_output)([pre_clip, kers])
        cur = Concatenate()([inp_img, pre, ibf_modelled, inp_ker])

    elif concatenate_res == 'many_inps':
        ar = []
        for l,r in [(0.2, 0.8), (0,1), (-0.5, 1.5), (-1, 2)]:
            clipped = tf.clip_by_value(pre, l, r)
            ibf_modelled = single_input_fft_conv_no_scale_output([clipped, kers])
            ar += [clipped, ibf_modelled]
        cur = Concatenate()([inp_img] + ar)

    elif concatenate_res is False:
        pass

    else:
        raise RuntimeError(f"concatenate_res = {concatenate_res}")


    return cur, kers


def final_ibf(cur, kers):
    cur = Lambda(clip01)(cur)  # just in case of some error
    cur = Lambda(single_input_fft_conv_no_scale_output)([cur, kers])
    return cur

##############################################################################################
# NNs

from sweet.optimization.nn.nn import *
def build_conv_k_nn(N=512, pre_args={}):
    inp_img, inp_ker = init_inps(N)
    kers = Lambda(norm_sum)(inp_ker)
    reg_params = {
        'kernel_regularizer': regularizers.l1_l2(l1=1e-5, l2=1e-4)
    }

    img_fft = fft(tf.cast(inp_img[..., 0], tf.complex64))
    psf_fft = fft(tf.cast(inp_ker[..., 0], tf.complex64))

    cur = tf.keras.layers.concatenate([
        tf.math.abs(img_fft)[..., tf.newaxis],
        # tf.math.imag(img_fft)[..., tf.newaxis],
        tf.math.abs(psf_fft)[..., tf.newaxis],
        # tf.math.imag(psf_fft)[..., tf.newaxis],
    ])

    # small initializers for some meaningful values..
    cur = Conv2D(20, (3,3), padding='same', kernel_initializer=tf.keras.initializers.Constant(0.0001), activation='relu', **reg_params)(cur)
    K = Conv2D(1, (3,3), padding='same', activation='relu', bias_initializer=tf.keras.initializers.Constant(0.01), kernel_initializer=tf.keras.initializers.Constant(0.0001), **reg_params)(cur)

    cur = Lambda(single_input_ibf)([inp_img, inp_ker, K[..., 0] + 0.0001])

    cur = final_ibf(cur, kers)

    model = Model([inp_img, inp_ker], cur)
    return model


def global_clip_simple(N=512, w=4, d=8, final_w=128, pre_args={}):
    assert 2**depth == N

    inps = init_inps(N)
    cur, kers = preprocess_inputs(inps, **pre_args)
    reg_params = {
        'kernel_regularizer': regularizers.l1_l2(l1=1e-5, l2=1e-4)
    }

    cur = Conv2D(10, (3,3), strides=(2,2), padding='same', activation='relu', **reg_params)(cur)
    cur = Conv2D(20, (3,3), strides=(2,2), padding='same', activation='relu', **reg_params)(cur)
    cur = Conv2D(40, (3,3), strides=(2,2), padding='same', activation='relu', **reg_params)(cur)

    cur = GlobalAveragePooling2D()(cur)
    cur = Dense(128, **reg_params)(cur)
    m = (Dense(1, activation='sigmoid', **reg_params)(cur))[:, tf.newaxis, tf.newaxis, :]
    d = (Dense(1, activation='sigmoid', **reg_params)(cur))[:, tf.newaxis, tf.newaxis, :]

    cur = Lambda(trainable_encoded_clip)([pre, m, d])
    cur = final_ibf(cur, kers)

    model = Model(inps, cur)
    return model



def build_trainable_K_nn(N, concatenate_res=True):
    inp_img = Input((N, N, 1))
    inp_ker = Input((N, N, 1))

    pre = IBF_Trainable_K()([inp_img, inp_ker])
    clipped = Lambda(clip01)(pre)
    res = Lambda(single_input_fft_conv)([clipped, inp_ker])
    model = Model([inp_img, inp_ker], res)
    return model


def trainable_global_clip(N, concatenate_res=True, w=4, d=8, final_w=128, mix_stage2=False):
    """Not-trainable model of precompensation + clip by conv-predicted GLOBAL thresholds"""
    inp_img = Input((N, N, 1))
    inp_ker = Input((N, N, 1))

    kers = Lambda(norm_sum)(inp_ker)
    pre = Lambda(single_input_ibf)([inp_img, kers])
    cur = pre
    if concatenate_res:
        pre_clip = Lambda(clip01)(pre)
        ibf_modelled = Lambda(single_input_fft_conv_no_scale_output)([pre_clip, kers])
        cur = Concatenate()([inp_img, pre, ibf_modelled, inp_ker])

    for i in range(d):
        cur = Conv2D(w*(2**i), 1, strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)
        cur = Conv2D(w*(2**i), 3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)

    cur = Flatten()(cur)
    cur = Dense(final_w, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)
    m = (Dense(1, activation='sigmoid')(cur))[:, tf.newaxis, tf.newaxis, :]
    d = (Dense(1, activation='sigmoid')(cur))[:, tf.newaxis, tf.newaxis, :]

    if mix_stage2:
        cur = Lambda(trainable_encoded_clip_to_zero_one)([pre, m, d])
    else:
        cur = Lambda(trainable_encoded_clip)([pre, m, d])

    cur = Lambda(clip01)(cur)  # just in case of some error
    cur = Lambda(single_input_fft_conv_no_scale_output)([cur, kers])

    model = Model([inp_img, inp_ker], cur)
    return model


def trainable_global_clip_avg(N, concatenate_res=True, w=4, d=8, final_w=128):
    """Not-trainable model of precompensation + clip by conv-predicted GLOBAL thresholds"""
    inp_img = Input((N, N, 1))
    inp_ker = Input((N, N, 1))

    kers = Lambda(norm_sum)(inp_ker)
    pre = Lambda(single_input_ibf)([inp_img, kers])
    cur = pre
    if concatenate_res:
        pre_clip = Lambda(clip01)(pre)
        ibf_modelled = Lambda(single_input_fft_conv_no_scale_output)([pre_clip, kers])
        cur = Concatenate()([inp_img, pre, ibf_modelled, inp_ker])

    for i in range(d):
        cur = Conv2D(w*(2**i), 1, strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)
        cur = Conv2D(w*(2**i), 3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)

    cur = Flatten()(cur)
    cur = Dense(final_w, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)
    m = (Dense(1, activation='sigmoid')(cur))[:, tf.newaxis, tf.newaxis, :]
    d = (Dense(1, activation='sigmoid')(cur))[:, tf.newaxis, tf.newaxis, :]

    cur = Lambda(trainable_encoded_clip)([pre, m, d])
    cur = Lambda(clip01)(cur)  # just in case of some error
    cur = Lambda(single_input_fft_conv_no_scale_output)([cur, kers])

    model = Model([inp_img, inp_ker], cur)
    return model


def build_easy_nn(**args):
    inp_img = Input((args['N'], args['N'], 1))
    inp_ker = Input((args['N'], args['N'], 1))
    # inp_psf =

    cur = inp_img
    # cur = Conv2D(10, (3,3), activation='relu', padding='same')(cur)
    # cur = Conv2D(10, (3,3), activation='relu', padding='same')(cur)
    cur = Conv2D(1, (3,3), activation='relu', padding='same')(cur)

    model = Model([inp_img, inp_ker], cur)
    return model


def fixed_precompensation(**args):
    """Not-trainable model of precompensation + blur"""
    inp_img = Input((args['N'], args['N'], 1))
    inp_ker = Input((args['N'], args['N'], 1))

    pre = Lambda(single_input_ibf)([inp_img, inp_ker])
    clipped = Lambda(clip01)(pre)
    res = Lambda(single_input_fft_conv)([clipped, inp_ker])
    model = Model([inp_img, inp_ker], res)
    return model


def few_convs_updated_ibf(**args):
    """Not-trainable model of precompensation + blur"""
    inp_img = Input((args['N'], args['N'], 1))
    inp_ker = Input((args['N'], args['N'], 1))

    pre = Lambda(single_input_ibf)([inp_img, inp_ker])
    cur = pre
    cur = Conv2D(32, 3, padding='same', activation='relu')(cur)
    cur = Conv2D(1, 3, padding='same', activation='relu')(cur)

    clipped = Lambda(clip01)(cur)
    res = Lambda(single_input_fft_conv)([clipped, inp_ker])
    model = Model([inp_img, inp_ker], res)
    return model


def few_convs_updated_ibf_sigmoid_clip(**args):
    """Not-trainable model of precompensation + blur"""
    inp_img = Input((args['N'], args['N'], 1))
    inp_ker = Input((args['N'], args['N'], 1))

    pre = Lambda(single_input_ibf)([inp_img, inp_ker])
    cur = pre
    cur = Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=9, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(1, 3, padding='same', activation='sigmoid')(cur)

    clipped = cur
    # clipped = Lambda(clip01)(cur)
    res = Lambda(single_input_fft_conv)([clipped, inp_ker])
    model = Model([inp_img, inp_ker], res)
    return model


def few_convs_updated_ibf_sigmoid_clip_v2(N, downscale=False):
    """Not-trainable model of precompensation + blur"""
    inp_img = Input((N, N, 1))
    inp_ker = Input((N, N, 1))

    pre = Lambda(single_input_ibf)([inp_img, inp_ker])
    cur = pre

    cur = Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=9, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=27, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=27, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=9, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(32, 3, dilation_rate=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),)(cur)
    cur = Conv2D(1, 3, padding='same', activation='sigmoid')(cur)

    clipped = cur
    # clipped = Lambda(clip01)(cur)
    res = Lambda(single_input_fft_conv)([clipped, inp_ker])
    model = Model([inp_img, inp_ker], res)
    return model


def trainable_encoded_clip(args, r=1e-5):
    vals, mid_enc, diff_enc = args
    mid_enc = mid_enc * 5 - 2 # -2 - 3 (0.5 -> 0.5)
    diffs = diff_enc * 2.5 + r# 0 - 2.5 (0.5 -> 1.25)

    mins = mid_enc - diffs # -4.5 - 3
    # maxs = mid_enc + diffs # -2 - 5.5
    return tf.clip_by_value(vals-mins, 0, 2*diffs) / (2*diffs)


def trainable_encoded_clip_to_zero_one(args, r=1e-5, coef=0.5):
    #
    vals, mid_enc, diff_enc = args
    mid_enc = mid_enc * 5 - 2 # -2 - 3 (0.5 -> 0.5)
    diffs = diff_enc * 2.5 + r# 0 - 2.5 (0.5 -> 1.25)

    mins = mid_enc - diffs # -4.5 - 3
    mins = coef*mins # + (1-coef)*0
    lens = coef*2*diffs + (1-coef)*1
    # maxs = mid_enc + diffs # -2 - 5.5
    return tf.clip_by_value(vals-mins, 0, lens) / lens


def norm_sum(kers):
    assert len(kers.shape) == 4
    return kers / tf.reduce_sum(kers, axis=(1,2,3), keepdims=True)


def add_unet(start_layer, inp_filters, rec_steps=0, reg=regularizers.l1_l2(l1=1e-8, l2=1e-6)):
    cur = start_layer

    if rec_steps != 0:
        cur = Conv2D(inp_filters, 3, padding='same', activation='relu', kernel_regularizer=reg)(cur)
        cur = Conv2D(inp_filters, 3, padding='same', activation='relu', kernel_regularizer=reg)(cur)

        rec_path = cur
        rec_path = MaxPooling2D((2,2))(rec_path)
        rec_path = add_unet(rec_path, inp_filters*2, rec_steps-1, reg=reg)
        rec_path = Conv2DTranspose(inp_filters, 1, strides=(2, 2), activation='relu', kernel_regularizer=reg)(rec_path)

        cur = Concatenate()([cur, rec_path])
        cur = Conv2D(inp_filters, 3, padding='same', activation='relu', kernel_regularizer=reg)(cur)
        cur = Conv2D(inp_filters, 3, padding='same', activation='relu', kernel_regularizer=reg)(cur)

    else:
        cur = Conv2D(inp_filters, 3, padding='same', activation='relu', kernel_regularizer=reg)(cur)

    return cur


def ibf_trainable_clip(N, norm_kers=False, concatenate_res=False, downscale=False):
    """Not-trainable model of precompensation + clip by conv-predicted thresholds"""
    inp_img = Input((N, N, 1))
    inp_ker = Input((N, N, 1))

    kers = inp_ker
    if norm_kers:
        kers = Lambda(norm_sum)(kers)

    pre = Lambda(single_input_ibf)([inp_img, kers])
    cur = pre

    if concatenate_res:
        ibf_modelled = Lambda(single_input_fft_conv)([cur, kers]) # not clipped!!!
        cur = Concatenate()([inp_img, pre, ibf_modelled])

    if downscale:
        cur = MaxPooling2D(downscale)(cur)


    cur = Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6),)(cur)
    cur = Conv2D(32, 3, dilation_rate=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6),)(cur)
    cur = Conv2D(32, 3, dilation_rate=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6),)(cur)
    cur = Conv2D(32, 3, dilation_rate=9, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6),)(cur)

    mid_enc = Conv2D(1, 3, padding='same', activation='sigmoid')(cur)
    diffs_enc = Conv2D(1, 3, padding='same', activation='sigmoid')(cur)

    if downscale:
        cur = UpSampling2D(downscale)(cur)

    res = Lambda(trainable_encoded_clip)([pre, mid_enc, diffs_enc])

    clipped = res
    # clipped = Lambda(clip01)(cur)
    if norm_kers:
        res = Lambda(single_input_fft_conv_no_scale_output)([clipped, kers])
    else:
        res = Lambda(single_input_fft_conv)([clipped, kers])
    model = Model([inp_img, inp_ker], res)
    return model



def unet_output_nn_v2(N, concatenate_res=True, unet_w=16, unet_d=5, output='clip'):
    """Not-trainable model of precompensation + clip by conv-predicted thresholds1

    v2:
    * using u-net, conved input
    * concatenate with conved
    * norm_kers == true
    """

    inp_img = Input((N, N, 1))
    inp_ker = Input((N, N, 1))

    kers = Lambda(norm_sum)(inp_ker)
    pre = Lambda(single_input_ibf)([inp_img, kers])
    cur = pre

    if concatenate_res:
        pre_clip = Lambda(clip01)(pre)
        ibf_modelled = Lambda(single_input_fft_conv_no_scale_output)([pre_clip, kers])
        cur = Concatenate()([inp_img, pre, ibf_modelled, inp_ker])


    cur = add_unet(cur, unet_w, rec_steps=unet_d)

    if output == 'clip':

        mid_enc = Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)
        diffs_enc = Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)
        cur = Lambda(trainable_encoded_clip)([pre, mid_enc, diffs_enc])

    elif output == 'out':
        # cur = Concatenate()([cur, inp_img])
        # cur = Conv2D(10, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur) #
        cur = Conv2D(1, 1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)

    else:
        raise NotImplementedError()

    cur = Lambda(clip01)(cur)  # just in case of some error
    cur = Lambda(single_input_fft_conv_no_scale_output)([cur, kers])

    model = Model([inp_img, inp_ker], cur)
    return model


def unet_output_nn_v3(N, concatenate_res=True, unet_w=16, unet_d=5, output='should be specified', pre_args={}):
    """Not-trainable model of precompensation + clip by conv-predicted thresholds1

    v3:
    * simpler common, preprocessing
    """

    inps = init_inps(N)
    cur, kers = preprocess_inputs(inps, **pre_args)

    cur = add_unet(cur, unet_w, rec_steps=unet_d)

    if output == 'clip':
        mid_enc = Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)
        diffs_enc = Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)
        cur = Lambda(trainable_encoded_clip)([pre, mid_enc, diffs_enc])

    elif output == 'out':
        # cur = Concatenate()([cur, inp_img])
        # cur = Conv2D(10, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur) #
        cur = Conv2D(1, 1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-6))(cur)

    else:
        raise NotImplementedError()

    cur = final_ibf(cur, kers)
    model = Model(inps, cur)
    return model



def ibf_trainable_clip_v2(*args, **kwargs):
    return unet_output_nn_v2(*args, **kwargs)


def ibf_unet_out_v2(*args, **kwargs):
    return unet_output_nn_v2(*args, output='out', **kwargs)




NAME2FUNC = {
    'EasyNN': build_easy_nn,
    'FixedPrecompensation': fixed_precompensation,
    'FewConvsUpdatedIBF': few_convs_updated_ibf,
    'FewConvsUpdatedIBFSigmoidClip': few_convs_updated_ibf_sigmoid_clip,
    'FewConvsUpdatedIBFSigmoidClipV2': few_convs_updated_ibf_sigmoid_clip_v2,
    'IBFTrainableClip': ibf_trainable_clip,
    'IBFTrainableClipV2': ibf_trainable_clip_v2,
    'IBFUnetOutV2': ibf_unet_out_v2,
    'UnetV3': unet_output_nn_v3,
    'IBFTrainableK': build_trainable_K_nn,
    'IBFConvK': build_conv_k_nn,
    'GlobalClip': trainable_global_clip,
    'GlobalClipSimple': global_clip_simple,
}


class NNFactory():
    def make(self, config):
        config = config.copy()
        obj_type = config.pop('type')
        if obj_type in NAME2FUNC:
            model = NAME2FUNC[obj_type](**config)
        else:
            raise NotImplementedError(f"Unexpected type {obj_type}")

        model.summary()
        return model
