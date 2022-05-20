from .prescription import PrescriptionWavefront
from .aberrometer import AberrationWavefront
from functools import lru_cache

@lru_cache(None)
def _build_wavefront_caching(D0, wavefront_coeffs_source, wavefront_args_frozenset):
    """wavefront builder helper to enable caching"""
    wavefront_args = dict(wavefront_args_frozenset)
    if wavefront_coeffs_source == 'prescription':
        return PrescriptionWavefront(D0=D0, **wavefront_args)
    elif wavefront_coeffs_source == 'aberrometer':
        return AberrationWavefront(D0=D0, coeffs=wavefront_args)

def build_wavefront_by_config(config):
    D0 = config['D0']
    wavefront_coeffs_source = config['wavefront_coeffs_source']
    wavefront_config = config["wavefront_config"]

    if wavefront_coeffs_source == 'prescription':
        wv_args = wavefront_config
    else:
        wv_args = wavefront_config['zernike_coeffs']

    return _build_wavefront_caching(D0, wavefront_coeffs_source, frozenset(wv_args.items()))