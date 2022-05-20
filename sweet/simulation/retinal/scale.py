import numpy as np
from sweet.transform.hist.histogram import scale


# Helper func to obtain scaling coeffs used in (3.34)
# for scaled retinal image simulation
def scale_intensity_coeffs(image, d=(0, 1)):
    assert len(d) == 2 and d[0] <= d[1], f"Incorrect d: {d}"

    dmin, dmax = d
    cmin, cmax = np.min(image), np.max(image)

    alpha = (dmax - dmin) / (cmax - cmin)
    betta = dmin - cmin * (dmax - dmin) / (cmax - cmin)

    return alpha, betta
