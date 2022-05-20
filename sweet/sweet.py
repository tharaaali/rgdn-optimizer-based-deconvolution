"""Class with precompensation methods"""

from sweet.util.config import make_precompensation_config, check_precompensation_config, \
    make_aberrometer_config, make_prescription_config
from sweet.util.data import load_zernike, load_nn, defaults, load_new_nn
from sweet.util.io import to_opt, from_opt, from_psf
from sweet.util.wave import PrescriptionWavefront, AberrationWavefront
from sweet.transform.filter import inverse_blur_filtering
from sweet.transform.hist import clipping
from sweet.transform.psf import PSF, PSF_params
from sweet.simulation import retinal

from skimage.transform import resize
from skimage.util import img_as_ubyte

import numpy as np


class _Core:
    @staticmethod
    def _check(ar):
        assert ar.dtype in [np.float32, np.float64], "The inputs should be in the opt format"
        assert ar.shape == (defaults["N"], defaults["N"]), "The inputs should be in the opt format"

    @staticmethod
    def calc_blur(image, psf):
        _Core._check(image)
        _Core._check(psf)
        return retinal.retinal(image, psf)

    @staticmethod
    def calc_precompensated(image, psf, stage, nn_name=None, **kwargs):
        _Core._check(image)
        _Core._check(psf)

        if stage == 1:
            ibf = inverse_blur_filtering(image, psf, defaults["PSF_params"]["regularization_const_K"])
            clipping_config = {
                "method": "hist_clipping_deprecated",
                "args": [0.01, 0.99],
            }
            precompensated = clipping(ibf, clipping_config)

        elif stage == 2:
            ibf = inverse_blur_filtering(image, psf, defaults["PSF_params"]["regularization_const_K"])
            clipping_config = {
                "method": "simple",
                "args": [0, 1],
            }
            precompensated = clipping(ibf, clipping_config)

        elif stage == 3:
            NN = load_nn()
            precompensated = NN([image[None, ..., None], psf[None, ..., None]])[0, ..., 0]

        elif stage == 4:
            min_clip_nn, max_clip_nn  = kwargs.get('nn_clips')
            clipping_config = {
                "method": "simple",
                "args": [min_clip_nn, max_clip_nn],
            }
            ibf = inverse_blur_filtering(image, psf,
                                         defaults["PSF_params"]["regularization_const_K"])
            precompensated = clipping(ibf, clipping_config)

        elif stage == "test":
            NN = load_new_nn(nn_name)
            precompensated = NN([image[None, ..., None], psf[None, ..., None]])[0, ..., 0]

        else:
            raise RuntimeError(f"Incorrect stage value {stage}")

        return precompensated

    @staticmethod
    def get_clip_nn(image, psf, stage=4):

        _Core._check(image)
        _Core._check(psf)
        NN = load_nn(stage=stage)

        clip_nn= NN([image[None, ..., None], psf[None, ..., None]])#[0, ..., 0]
        min_clip_nn, max_clip_nn = np.array(clip_nn[0]).reshape(1, -1)[0][0], \
                                    np.array(clip_nn[1]).reshape(1, -1)[0][0]
        return min_clip_nn, max_clip_nn



class ImTransform:
    def __init__(self, im_shape: np.array, std_size: int=512):
        """Convert image to square

        Parameters
        ----------
        im_shape: np.array 1d
            image shape
        std_size: int
            std image size (new width and height image)
        """
        self.max_size = np.max(im_shape[:2])
        self.OH, self.OW, self.OC = im_shape
        self.ay, self.ax = 0, 0
        self.std_size = std_size

    def _im_to_square(self, im: np.array):
        """Convert image to square

        Parameters
        ----------
        im: np.array
            Image with shape NxMx3 in usual np.uint8 format

        Returns
        ----------
        np.array
            Square image  with shape NxNx3 (N is max size from N and M) in usual np.uint8 format
        """

        if self.OH == self.OW:
            return im

        square_im = np.zeros((self.max_size, self.max_size, self.OC), np.uint8)

        ax, ay = (self.max_size - self.OW) // 2, (self.max_size - self.OH) // 2

        square_im[ay:self.OH + ay, ax:ax + self.OW] = im

        if self.OH > self.OW:
            square_im[:, :ax, :] = im[:, :1, :]
            square_im[:, ax + self.OW:, :] = im[:, -1:, :]
        elif self.OW > self.OH:
            square_im[:ay, :, :] = im[:1, :, :]
            square_im[ay + self.OH:, :, :] = im[-1:, :, :]

        self.ax, self.ay = ax, ay
        return square_im

    def _im_resize(self, im: np.array, resize_to: int=512):
        """Resize square image

        Parameters
        ----------
        im: np.array
            Image with shape NxNx3 (image must be square) in usual np.uint8 format

        resize_to: int
            size of new image

        Returns
        ----------
        np.array
            Square image with shape MxMx3 (M is size_to) in usual np.float32 format
        """

        H, W = im.shape[:2]
        assert W == H, 'width and height of image must be equals,' \
                       ' please call img_to_square at first'

        resize_im = resize(im, (resize_to, resize_to))

        return resize_im

    def _im_to_uint8(self, im: np.array):
        """Convert image to uint8

        Parameters
        ----------
        im: np.array
            Image with shape NxNx3 in usual np.float32 format

        Returns
        ----------
        np.array
             Image with shape NxNx3 in usual np.uint8 format
        """
        return img_as_ubyte(im)

    def im_to_std(self, im: np.array):
        """Convert image to square and resize

        Parameters
        ----------
        im: np.array
            Image with shape NxMx3 in usual np.uint8 format

        Returns
        ----------
        np.array
             Image with shape NxNx3 (N is max size from N and M) in usual np.uint8 format
        """
        im = self._im_to_square(im)
        im = self._im_resize(im, resize_to=self.std_size)
        im = self._im_to_uint8(im)
        return im

    def im_to_original(self, im: np.array):
        """Resize image and select rectangle from square

        Parameters
        ----------
        im: np.array
            Image with shape NxNx3 in usual np.uint8 format

        Returns
        ----------
        np.array
             Image with shape NxMx3 in usual np.uint8 format
        """
        im = self._im_resize(im, resize_to=self.max_size)
        im = self._im_to_uint8(im)
        im = im[self.ay: self.OH + self.ay, self.ax: self.OW + self.ax]
        return im


class Sweet():
    def __init__(self):
        """Initialize sweet processing"""
        self.wavefront = None
        self.wavefront_config = None
        self.wavefront_coeffs_source = None
        self.D0 = None

        self.exp_config = None

    def set_eye_prescription(self, S: float, A: float, C: float, pupil_diam_0: float = 10.,
                             VD: float = 12.):
        """Set eye configuration with prescription data

        Parameters
        ----------
        S: float
            Spherical refraction error (in diopters)
        A: float
            Axis of the cylindrical refraction error (in degrees)
        C: float
            Cylindrical refraction error (in diopters)
        pupil_diam_0: float = 10.
            Upper bound for pupil size (in mm). Is used for a small optimization and aberrometer-like interface
        VD: float = 12.
            Vertex distance (in mm).
        """

        self.wavefront_config = make_prescription_config(S=S, A=A, C=C)
        self.wavefront = PrescriptionWavefront(D0=pupil_diam_0, **self.wavefront_config)
        self.wavefront_coeffs_source = 'prescription'
        self.D0 = pupil_diam_0

    def set_eye_aberrometer(self, zernike_coeffs: dict, pupil_diam_0: float):
        """Set eye configuration with aberrometer data

        Parameters
        ----------
        zernike_coeffs: dict
            Dictionary with zernike coefficients in format 'Z{i} -> value'.
        pupil_diam_0: float
            Pupil diam for the calculated zernike coefficients
        """

        assert len(zernike_coeffs) == 28 and all(
            f'Z{i}' in zernike_coeffs for i in range(28)), 'Unexpected zernike coeffs.'

        self.wavefront_config = make_aberrometer_config(zernike_coeffs)
        self.wavefront = AberrationWavefront(D0=pupil_diam_0,
                                             coeffs=self.wavefront_config['zernike_coeffs'])
        self.wavefront_coeffs_source = 'aberrometer'
        self.D0 = pupil_diam_0

    def set_eye_participant(self, user_id: str, eye: str):
        """Set eye params to participant's values from the project aberrometer data

        Parameters
        ----------
        user_id: str
            Anonymous user id (see data/aberrommeter), e.g. "WADB"
        eye: str
            Eye ("L" or "R")
        """

        assert eye in ["L", "R"]
        eye_params = load_zernike(user_id, eye)
        self.set_eye_aberrometer(
            zernike_coeffs={f'Z{i}': eye_params[f'Z{i}'] for i in range(28)},
            pupil_diam_0=eye_params['D0'],
        )

    def set_experiment_params(self, pupil_diam: float, view_dist: float = 400.,
                              canvas_size_h: float = 50.):
        """Set experiment config params

        Parameters
        ----------
        pupil_diam: float
            Pupil diameter in the experiment (in mm)
        view_dist: float = 400.
            Viewing distance, how far is the participant from the monitor (in cm)
        canvas_size_h: float = 50.
            Canvas size, how big is image (with padding) on monitor (in cm)
        """

        if self.wavefront is None:
            raise RuntimeError(
                "Set eye configuration first, please. It can be done via: set_eye_prescription, set_eye_aberrometer, set_eye_participant methods")

        self.config = make_precompensation_config(
            D=pupil_diam,
            view_dist=view_dist,
            opt_size_h=canvas_size_h / defaults["k_ratio"],

            wavefront_config=self.wavefront_config,
            wavefront_coeffs_source=self.wavefront_coeffs_source,
            D0=self.D0,

            N=defaults["N"],
            k_ratio=defaults["k_ratio"],
            regularization_const_K=defaults["PSF_params"]["regularization_const_K"],
            apply_hist_clipping=True,
            hist_clips=defaults["PSF_params"]["hist_clips"],
        )
        check_precompensation_config(self.config)

    def _check_image(self, image):
        expected_shape = (defaults["N"], defaults["N"], 3)
        assert image.shape == expected_shape, f"Image shape is unexpected, convert to {expected_shape}, please"
        assert image.dtype == np.uint8, f"Image dtype is expected to be np.uint8, convert to np.uint8, please"

    def _psf(self):
        assert self.config is not None, "Set experiments params, please, with set_experiment_params method"

        _, betta = PSF_params(
            self.config["opt_size_h"],
            self.config["canvas_opt_ratio_k"],
            self.config["view_dist"]
        )
        psf = PSF(
            self.config["D"],
            self.config["D0"],
            betta,
            self.config["view_dist"],
            self.wavefront,
            self.config["N"],
            norm=False,
            calc_scale=defaults["PSF_params"]["calc_scale"]
        )
        return psf


    def get_nn_clip_rgb(self, image, psf, clipping):
        """
        Get clipping range for RGB image

        Parameters
        ----------
        image: np.array
            Image with shape NxNx3 (N=512) in usual np.uint8 format
        psf: np.array
            -
        clipping: str
            Type of clipping (Median, Minmax, Green)
        Returns
        ----------
        float, float
            Min clip number, max clip number
        """

        if clipping == 'Green':
            channel = np.dstack([image[..., 1]] * 3)
            im_transform = ImTransform(channel.shape, self.config["N"])
            channel = im_transform.im_to_std(channel)
            channel = to_opt(channel)
            _Core._check(channel)
            _Core._check(psf)
            min_clip_nn, max_clip_nn = _Core.get_clip_nn(channel, psf)
        else:
            maxs_clip_nn = []
            mins_clip_nn = []
            for channel in [np.dstack([image[..., i]] * 3) for i in range(3)]:
                im_transform = ImTransform(channel.shape, self.config["N"])
                channel = im_transform.im_to_std(channel)
                channel = to_opt(channel)
                _Core._check(channel)
                _Core._check(psf)
                min_clip_nn, max_clip_nn = _Core.get_clip_nn(channel, psf)
                maxs_clip_nn.append(max_clip_nn)
                mins_clip_nn.append(min_clip_nn)
            if clipping == 'Median':
                min_clip_nn, max_clip_nn =  np.median(mins_clip_nn + [0]), np.median(maxs_clip_nn + [1]),
            elif clipping == 'Minmax':
                min_clip_nn = min(mins_clip_nn)
                max_clip_nn = max(maxs_clip_nn)
        print(f" min_clip= {min_clip_nn}, max_clip={max_clip_nn}")
        return min_clip_nn, max_clip_nn

    def calc_precompensated(self, image: np.array, stage: int = 3,
                            nn_name: bool = None, rgb: bool = None):
        """Calculate the Huang precompensation for the image

        Parameters
        ----------
        image: np.array
            Image with shape NxNx3 (N=512) in usual np.uint8 format
        stage: int
            Use algorithm for the stage
        nn_name: bool
            -
        rgb: bool
            Set true for working with RGB image

        Returns
        ----------
        np.array
            Precompensated image
        """

        psf = self._psf()
        if rgb:
            RGB_IMAGE = None
            min_clip_nn, max_clip_nn = None, None
            #Median
            if stage == 4.1:
                min_clip_nn, max_clip_nn = self.get_nn_clip_rgb(image, psf, 'Median')
            #Minmax
            elif stage == 4.2:
                min_clip_nn, max_clip_nn = self.get_nn_clip_rgb(image, psf, 'Minmax')
            #Green
            elif stage == 4.3:
                min_clip_nn, max_clip_nn = self.get_nn_clip_rgb(image, psf, 'Green')
            stage = int(stage)
            for channel in [np.dstack([image[..., i]] * 3) for i in range(3)]:
                im_transform = ImTransform(channel.shape, self.config["N"])
                channel = im_transform.im_to_std(channel)
                channel = to_opt(channel)
                precompensated = _Core.calc_precompensated(
                    channel,
                    psf,
                    stage,
                    nn_clips=(min_clip_nn, max_clip_nn) \
                        if all([min_clip_nn, max_clip_nn]) else None
                    )

                precompensated_image = from_opt(precompensated, float64_ok=True)
                precompensated_image = im_transform.im_to_original(precompensated_image)
                precompensated_image = precompensated_image[..., 0]
                if RGB_IMAGE is None:
                    RGB_IMAGE = precompensated_image
                else:
                    RGB_IMAGE = np.dstack([RGB_IMAGE, precompensated_image])

            return RGB_IMAGE

        im_transform = ImTransform(image.shape, self.config["N"])
        image = im_transform.im_to_std(image)
        image = to_opt(image)
        precompensated = _Core.calc_precompensated(image, psf, stage, nn_name=nn_name)

        precompensated_image = from_opt(precompensated, float64_ok=True)
        precompensated_image = im_transform.im_to_original(precompensated_image)

        return precompensated_image

    def calc_modelled_blur(self, image: np.array, rgb: bool = False):
        """Calculate retinal image blur, modelled for the previously set eye

        Parameters
        ----------
        image: np.array
            Image with shape NxNx3 (N=512) in usual np.uint8 format
        rgb: bool
            Set true for working with RGB image

        Returns
        ----------
        np.array
            Precompensated image
        """

        psf = self._psf()
        if rgb:
            RGB_IMAGE = None
            for channel in [np.dstack([image[..., i]] * 3) for i in range(3)]:
                im_transform = ImTransform(channel.shape, self.config["N"])
                channel = im_transform.im_to_std(channel)
                channel = to_opt(channel)

                retinal_image = _Core.calc_blur(channel, psf)

                retinal_image = from_opt(retinal_image, float64_ok=True)
                retinal_image = im_transform.im_to_original(retinal_image)
                retinal_image = retinal_image[..., 0]
                if RGB_IMAGE is None:
                    RGB_IMAGE = retinal_image
                else:
                    RGB_IMAGE = np.dstack([RGB_IMAGE, retinal_image])

            return RGB_IMAGE

        im_transform = ImTransform(image.shape, self.config["N"])
        image = im_transform.im_to_std(image)
        image = to_opt(image)

        retinal_image = _Core.calc_blur(image, psf)

        retinal_image = from_opt(retinal_image, float64_ok=True)
        retinal_image = im_transform.im_to_original(retinal_image)
        return retinal_image
