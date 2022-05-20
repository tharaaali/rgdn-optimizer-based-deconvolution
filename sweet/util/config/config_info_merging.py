from .generate import make_precompensation_config, make_wavefront_config
from ._defaults import load_defaults, defaults
from .util import calc_recommended_opt_size, check_precompensation_config
from sweet.util.data._data import load_zernike
from sweet.transform.psf import PSF_params

import numpy as np
import logging
logger = logging.getLogger(__name__)

def _get_d(tracker_info, eye, D0, multiplicator=1.0, round_num=2):
    assert multiplicator==1.0, "multiplicator functionality moved to w_multiplicator"

    if tracker_info is not None:
        field = {
            'L': 'left_D',
            'R': 'right_D',
        }[eye]
        D = tracker_info[field] * multiplicator
    else:
        D = D0 * multiplicator

    min_val = 1  # magick default constant
    if min_val <= D and D <= D0:
        return round(D, round_num)
    else:
        logger.warning(f'Incorrect D = {D}, should be {min_val}-{D0}, will be clipped')
        return np.clip(D, min_val, D0)


def build_updated_config(extra_params):
    # reopen every run for online reaction
    params = load_defaults()

    # update via extra parameters
    if extra_params is not None:
        if defaults['Mode'].lower() == 'debug':
            print('Updating config with extra_params:', extra_params)

        extra_params = extra_params.copy()
        params["PSF_params"]["regularization_const_K"] = 10 ** extra_params.pop('logK')
        params["Demo"]["ManualEyeParams"]["S"] = extra_params.pop('S')
        params["Demo"]["UseManualEyeParams"] = extra_params.pop('use_client_config')
        params["Demo"]['Eye'] = extra_params.pop('eye')

        params["Demo"]["Participant"] = extra_params.pop('user_id')
        params["view_dist"] = extra_params.pop('distance') * 100
        params["EyeTracker"]["PupilMultiplicator"] = extra_params.pop('pupil_multiplicator')

        params["CalibMode"] = extra_params.pop('enable_calib_mode', False)

        assert len(extra_params) == 0, f"Error, some extra params {extra_params} are ignored!"

    # set wv_params
    if params['Demo']['UseManualEyeParams']:
        # manual params for camera or debug
        eye_params = params['Demo']['ManualEyeParams']
        params['WavefrontParams'] = {
            'wavefront_coeffs_source': 'prescription',
            'D0': eye_params['D0'],
            'wavefront_args': {
                k: eye_params[k] for k in ['A', 'C', 'S']
            },
            'spherical_equivalent': eye_params['S'] + (eye_params['C'] / 2),
        }

    else:
        eye_params = load_zernike(params['Demo']["Participant"], params['Demo']["Eye"])
        params['WavefrontParams'] = {
            'wavefront_coeffs_source': 'aberrometer',
            'D0': eye_params['D0'],
            'wavefront_args': {
                f'Z{i}': eye_params[f'Z{i}'] for i in range(28)
            },  # zernike coeffs
            'spherical_equivalent': eye_params['ZSph'] + (eye_params['ZCyl'] / 2),
        }

    # calculate opt_size
    if params['auto_opt_size']:
        params['opt_size_h'] = calc_recommended_opt_size(
            spherical_equivalent = params['WavefrontParams']['spherical_equivalent'],
            D0 = params['WavefrontParams']['D0'],
            view_dist = params['view_dist'],
        )
        params['opt_size_h'] *= params['auto_opt_size_multiplicator']
        if defaults['Mode'].lower() == 'debug':
            print(f"calculated opt_size={params['opt_size_h']}")

    return params


def join_to_precompensation_config(params, tracker_info):
    WV_params = params['WavefrontParams']
    D0 = WV_params['D0']
    # D = _get_d(tracker_info, params["Demo"]["Eye"], D0, params["EyeTracker"]["PupilMultiplicator"])
    D = _get_d(tracker_info, params["Demo"]["Eye"], D0) # multiplicator moved to w_multiplicator
    logger.debug(f"D={D}")


    wavefront_config = make_wavefront_config(WV_params['wavefront_coeffs_source'], WV_params['wavefront_args'])

    # wavefront, wavefront_config = build_wavefront(
    #     D0,
    #     WV_params['wavefront_coeffs_source'],
    #     frozenset(WV_params['wavefront_args'].items()),
    # )

    config = make_precompensation_config(
        wavefront_config = wavefront_config,
        wavefront_coeffs_source = params['WavefrontParams']['wavefront_coeffs_source'],

        D = D,
        D0 = D0,

        k_ratio = params["k_ratio"],
        opt_size_h = params["opt_size_h"],
        view_dist = params["view_dist"],
        N = params["N"],
        regularization_const_K = params["PSF_params"]["regularization_const_K"],
        apply_hist_clipping = True,
        hist_clips = params["PSF_params"]["hist_clips"],
    )

    check_precompensation_config(config)
    _alpha, betta = PSF_params(config["opt_size_h"], config["canvas_opt_ratio_k"], config["view_dist"])
    return config, betta
