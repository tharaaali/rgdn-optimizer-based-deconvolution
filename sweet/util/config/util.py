import numpy as np


def check_wavefront_config(data, mode):
    assert mode in ('prescription', 'aberrometer'), f"Unknown mode: {mode}"
    assert isinstance(data, dict), f"Expected wavefront data to be of `dict` type, got {type(data)}"

    if mode == 'aberrometer':
        assert "zernike_coeffs" in data, f"Missing `zernike_coeffs` param in wavefront data"
    else:
        for coeff in ("A", "C", "S", "Vd"):
            assert coeff in data, f"Missing `{coeff}` param in wavefront data"


def check_precompensation_config(config):
    for key in config.keys():
        assert key in (
            "D", "D0", "opt_size_h", "canvas_opt_ratio_k", "view_dist", "N",  # psf params
            "wavefront_config", "wavefront_coeffs_source",  # wavefront params
            "regularization_const_K",  # precompensation params
            "apply_hist_clipping", "hist_clips", "display_device_range",  # clipping params
        )

    assert config["wavefront_coeffs_source"] in ('prescription', 'aberrometer'), \
        f"Unknown wavefront coeffs source: {config['wavefront_coeffs_source']}"
    assert config["D"] >= 1.0, f"Expected D >= 1.0 cm, got {config['D']}"
    assert config["D0"] >= config["D"], f"Expected D0 >= D, got D0={config['D0']}, D={config['D']}"
    assert config["canvas_opt_ratio_k"] >= 1, \
        f"Expected canvas/opt ratio >= 1, got {config['canvas_opt_ratio_k']}"
    assert config["opt_size_h"] > 0, f"Expected opt_size_h > 0, got {config['opt_size_h']}"
    assert config["view_dist"] > 0, f"Expected view_dist > 0, got {config['view_dist']}"
    assert config["N"] > 0, f"Expected N > 0, got {config['N']}"
    assert len(config["display_device_range"]) == 2
    if type(config["hist_clips"]) == list or type(config["hist_clips"]) == tuple:
        assert 0 < len(config["hist_clips"]) <= 2, \
            f"Expected hist_clips to be either a list/tuple like (low, high) " \
            f"or single float param, got {config['hist_clips']}"


def calc_recommended_opt_size(spherical_equivalent, D0, view_dist, VD=12):
    def _vd_compensation(S):
        return S / (1 - VD*S/1000)

    def _dist_compensation(S):
        return S + (100 / view_dist)

    S = _vd_compensation(spherical_equivalent)
    S = _dist_compensation(S)
    alpha_rad = abs(D0 * S / 1000)
    opt_size = np.tan(alpha_rad / 2) * view_dist * 2

    return opt_size