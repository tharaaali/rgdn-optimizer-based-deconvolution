import numpy as np


def make_prescription_config(A=0.0, C=0.0, S=0.0, Vd=12):
    assert -180 <= A <= 180, f"Expected A in [-180; 180] degrees, got {A}"
    assert -3 <= C <= 0, f"Expected C in [-3; 0] D, got {C}"
    assert -20 <= S <= 20, f"Expected S in [-20; 20] D, got {S}"
    assert 0 <= Vd <= 20, f"Expected Vd in [0; 20] mm, got {Vd}"

    return {'A': A, 'C': C, 'S': S, 'Vd': Vd}


def make_aberrometer_config(coeffs):
    assert isinstance(coeffs, (list, dict, np.ndarray)), \
        f"Expected `list`, `dict` or `np.ndarray`, got {type(coeffs)}"

    return {'zernike_coeffs': coeffs}


def make_default_config():
    wavefront_config = make_prescription_config()
    return make_precompensation_config(wavefront_config)


def make_wavefront_config(wavefront_coeffs_source, wavefront_args):
    if wavefront_coeffs_source == 'prescription':
        wavefront_config = make_prescription_config(**wavefront_args)
    else:
        # 'aberrometer'
        wavefront_config = make_aberrometer_config(wavefront_args)
    return wavefront_config


def make_precompensation_config(
    wavefront_config: dict,
    wavefront_coeffs_source: str = "prescription",  # `prescription` or `aberrometer`
    D: float = 4.0,  # pupil size
    D0: float = 6.0,  # pupil size from aberrometer
    k_ratio: float = 2.0,  # (canvas size / optotype size) ratio
    opt_size_h: float = 1.0,
    view_dist: float = 75,
    N: int = 1024,
    regularization_const_K: float = 0.1,  # empirical const K
    apply_hist_clipping: bool = True,  # return clipped image or not
    hist_clips: float or tuple or list = 1 / 1000,
    display_device_range: tuple = (0, 1)
):
    config = dict()
    config["D"] = D
    config["D0"] = D0
    config["opt_size_h"] = opt_size_h
    config["canvas_opt_ratio_k"] = k_ratio
    config["view_dist"] = view_dist
    config["N"] = N
    config["wavefront_config"] = wavefront_config
    config["wavefront_coeffs_source"] = wavefront_coeffs_source
    config["regularization_const_K"] = regularization_const_K
    config["apply_hist_clipping"] = apply_hist_clipping
    config["hist_clips"] = hist_clips
    config["display_device_range"] = display_device_range

    return config