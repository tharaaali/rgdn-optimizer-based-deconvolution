import matplotlib.pyplot as plt
import numpy as np
import math
import warnings

from sweet.util.io.convert import _autocontrast as autocontrast


def prepare_img(img, vmin=0, vmax=1, apply_gamma=True, apply_ac=False):
    if apply_ac:
        img = autocontrast(img)

    img = np.clip(img, vmin, vmax)

    if apply_gamma:
        # images in the project are *linear*
        # the standard interfaces (imshow, imsave) expect *gamma-correction*
        img = img ** (1/2.2)
    return img


def std_imshow(axs, img, cmap='gray', vmin=0, vmax=1, apply_gamma=True, apply_ac=False, **args):
    """Show function. Runs plt imshow with project-specific parameters:
    """
    warnings.warn('Function is deprecated')
    img = prepare_img(img, vmin, vmax, apply_gamma, apply_ac)
    return axs.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, **args)


def viz_dict(imgs_dict, **args):
    names = list(imgs_dict.keys())
    images = [imgs_dict[n] for n in names]

    return viz(*images, titles=names, **args)


def viz_ar(images, titles=None, **args):
    return viz(*images, titles=titles, **args)


def viz(*images, titles=None, figsize=(8,6), cmap='gray', cols='auto', show_range=True, apply_ac=False, prepared_img_mode=None, return_image=False):
    # deprecation warnings
    if prepared_img_mode is None:
        if images[0].dtype == np.uint8:
            prepared_img_mode = True
        else:
            warnings.warn('float images viz is deprecated, use explicit format change (from_psf/from_opt)')

        if apply_ac:
            warnings.warn('apply_ac is deprecated, use explicit format change (from_psf/from_opt)')

    plt.figure(figsize=figsize, constrained_layout=True)

    if cols == 'auto':
        cols = math.floor(len(images)**0.5)
    rows = math.ceil(len(images) / cols)

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=figsize, squeeze=False)

    for y in range(rows):
        for x in range(cols):
            i = y*cols + x

            if i < len(images):
                img = images[i]
                if prepared_img_mode:
                    assert not apply_ac
                    axs[y,x].imshow(img)

                else:
                    img = np.array(img).astype(np.float32)
                    std_imshow(axs[y,x], img, apply_ac=apply_ac)

                if titles or show_range:
                    title_parts = []
                    if titles:
                        title_parts.append(titles[i])
                    if show_range:
                        title_parts.append(f'({img.min():.2f}-{img.max():.2f})')

                    axs[y,x].set_title(' '.join(title_parts))

            else:
                fig.delaxes(axs[y,x])


    # for i, img in enumerate(images):
    #     y = i // cols
    #     x = i % cols

    if return_image:
        # thx https://stackoverflow.com/a/57988387
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image_from_plot
    else:
        plt.show()


# todo: move to from_opt/from_psf functions
def _imshow_check_correctness(show=True):
    """function for manual check of gamma-correction and vmin,vmax"""
    n = 3
    image1 = np.zeros([4*n+1, 4*n+1])
    for count in range(3):
        l = 2*n - n//2
        r = l + count + 1
        image1[l:r, 2+4*count] += 0.5 / (r-l)

    if show:
        std_imshow(plt, image1)
        plt.title('Pixel groups have the same brightness')
        plt.show()

    image2 = np.zeros([4*n+1, 4*n+1]) + 0.2
    image2[n:-n, n:-n] += 0.2

    if show:
        std_imshow(plt, image2)
        plt.title('Image have no pure black or white pixels')
        plt.show()

    N = 100
    image3 = np.zeros([N, N])
    image3[1::2, 0::2] = image3[0::2, 1::2] = 1  # chess board
    image3[:, N//2:] = 0.5

    if show:
        std_imshow(plt, image3)
        plt.title('Image halves should have approximately the same brightness')
        plt.show()

    return image1, image2, image3