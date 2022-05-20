import numpy as np
import warnings

# Helper func to obtain clipping bounds of histogram
# Starting from the peak and going to left (right) for lower (higher) bound
# when the observed frequency is lower than set threshold (% of peak)
# the bound is considered to be reached
def histogram_clipping_bounds(freq, bin_edges, clips=0.01):
    if type(clips) != list:
        clips = [clips] if type(clips) != tuple else list(clips)

    assert 0 < len(clips) < 3, f"Incorrect clips param passed: {clips}"

    clow, chigh = clips[0], clips[-1]

    peak, index = np.max(freq), np.argmax(freq)
    l, h = index, index

    while l > 0 and freq[l] >= peak * clow:
        l -= 1

    while (h + 1) < len(freq) and freq[h] >= peak * chigh:
        h += 1

    # numpy's bins from histogram func represent intervals
    # in a following way: [bin[0], bin[1]), [bin[1], bin[2]), ..., [bin[n-2], bin[n-1]]
    # thus we take bin edge at lower index as lower bound and
    # the following bin after upper index as higher bound
    # so that we have a range = [cl; ch)
    cl = bin_edges[l]  # lower bound
    ch = bin_edges[h + 1]  # upper bound

    return cl, ch


# Histogram clipping method described in 3.2.5
def hist_clipping(image, clips=0.01, image_range=None, d=(0, 1)):
    # warnings.warn("This method is deprecated, condseder using simple/hist methods, please")

    assert len(d) == 2 and d[0] <= d[1], f"Incorrect d: {d}"
    assert np.min(image) >= 0 and np.max(image) <= 1, \
        f"Expected image with range [0; 1], got: [{np.min(image)}; {np.max(image)}]"

    dmin, dmax = d
    image = image.copy()

    freq, edges = np.histogram(image, bins=256, range=image_range)
    cl, ch = histogram_clipping_bounds(freq, edges, clips=clips)

    # range = [cl; ch)
    # thus we use `<` instead of `<=` as it was proposed in baseline paper
    middle_mask = (image >= cl) & (image < ch)
    low_mask = image < cl
    high_mask = image >= ch

    image[middle_mask] = dmin + (image[middle_mask] - cl) * (dmax - dmin) / (ch - cl)
    image[low_mask] = dmin
    image[high_mask] = dmax

    # returning clipping data may be helpful
    return image, (cl, ch)


def simple_clipping(img, l, r):
    return ((img - l) / (r - l)).clip(0, 1)


def hist_clipping_new(img, l, r):
    """Clipping by histogram.

    Parameters
    ----------
    img : np.array
        image (linear, no gamma-correction)

    l : float
        clip share of left histogram values

    r: float
        clip share of right histogram values


    Returns:
    --------
    img : np.array
    """
    assert 0 <= l < r <= 1, f"Incorrect l={l}, r={r}"

    flatten = img.flatten()
    l = np.percentile(flatten, l*100)
    r = np.percentile(flatten, r*100)

    return simple_clipping(img, l, r)


def scale(image, d=(0, 1)):
    assert len(d) == 2 and d[0] <= d[1], f"Incorrect d: {d}"

    dmin, dmax = d
    cmin, cmax = np.min(image), np.max(image)
    return dmin + ((dmax - dmin) * (image - cmin)) / (cmax - cmin)


def clipping(image, clipping_config):
    method = clipping_config['method']
    l, r = clipping_config['args']

    if method == 'simple':
        return simple_clipping(image, l, r)

    elif method == 'hist':
        return hist_clipping_new(image, l, r)

    elif method == 'hist_clipping_deprecated':
        image = scale(image, (0,1))
        return hist_clipping(image, (l, 1-r))[0]

    else:
        raise NotImplementedError(f'Unknown clipping method {method}, use one of "simple", "hist", "hist_clipping_deprecated"')