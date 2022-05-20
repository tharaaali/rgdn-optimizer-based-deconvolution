import yaml
import time
from pathlib import Path
from copy import deepcopy

DATA_PATH = (Path(__file__).parents[2] / 'config.yaml').resolve()
CACHED_DEFAULTS = {
    'time': None,
    'vals': None,
}

def _load_defaults(subname=None, timeout=2):
    """Load defaults.
        Useful, if you want online updates on defaults changed

    :param string subname: subname to load only specific subconfig
    :param float timeout: cache timeout in seconds
    """
    if CACHED_DEFAULTS['time'] is None or time.time() - CACHED_DEFAULTS['time'] >= timeout:
        with open(DATA_PATH) as f:
            defaults = yaml.safe_load(f)
        CACHED_DEFAULTS['time'] = time.time()
        CACHED_DEFAULTS['vals'] = defaults
    else:
        defaults = deepcopy(CACHED_DEFAULTS['vals'])

    if subname is not None:
        defaults = defaults[subname]

    return defaults


defaults = _load_defaults()
def load_defaults(subname=None):
    return _load_defaults(subname)