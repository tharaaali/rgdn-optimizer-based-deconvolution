import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

import sweet.util.fft as fft
from sweet.transform.psf import PSF, PSF_params
from sweet.simulation.retinal import retinal
from sweet.util.wave import PrescriptionWavefront, AberrationWavefront
from sweet.util.io import from_psf, to_psf, from_opt, to_opt

from .util import make_box_layout, make_vertical_layout


"""
Currently input image is used as it is.
Only PSF is flipped left to right when retinal image is computed.

Fix later when issue with to/from_opt/psf converters is resolved.
"""

class PSF_widget(widgets.VBox):

    def __init__(self, input_image, pupil_d, k_ratio, mode='prescription', zernike_coeffs=None):
        super().__init__()
        output = widgets.Output()

        assert pupil_d > 2, f"Incorrect value of pupil diam from aberrometer (D0) = {pupil_d}"
        assert len(input_image.shape) == 2 and input_image.shape[0] == input_image.shape[1], \
            f"Incorrect shape of input image: expected (NxN), got {input_image.shape}"
        assert mode in ('prescription', 'aberrometer'), f"Unknown mode: {mode}"

        if mode == 'aberrometer':
            # zernike_coeffs != None
            # len(zernike_coeffs) > 0
            assert zernike_coeffs, "Provide zernike_coeffs for `aberrometer` mode"

        self.mode = mode

        if mode == 'prescription':
            self.wavefront = PrescriptionWavefront(D0=pupil_d, A=0.0, C=0.0, S=0.0)
        else:
            self.wavefront = AberrationWavefront(D0=pupil_d, coeffs=zernike_coeffs)

        self.src = input_image
        self.N = input_image.shape[0]
        self.D = 3.0
        self.D0 = pupil_d
        self.opt_size = 1.0
        self.k_ratio = k_ratio
        self.view_dist = 50

        alpha, betta = PSF_params(self.opt_size, self.k_ratio, self.view_dist)
        self.alpha = alpha
        self.betta = betta

        with output:
            self.fig, self.axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4))

        self.axs[0].set_title('Input image')
        self.axs[1].set_title('PSF')
        self.axs[2].set_title('Output image')
        self.fig.canvas.toolbar_position = 'bottom'

        self.psf = PSF(
            D=self.D,
            D0=self.D0,
            betta=self.betta,
            d=self.view_dist,
            W=self.wavefront,
            N=self.N,
        )
        self.out = retinal(self.src, np.fliplr(self.psf))

        self.input_plot = self.axs[0].imshow(
            from_opt(self.src, float64_ok=True), extent=self._plot_bounds()
        )
        self.psf_plot = self.axs[1].imshow(
            from_psf(fft.scale(fft.shift(self.psf)), float64_ok=True), extent=self._plot_bounds()
        )
        self.out_plot = self.axs[2].imshow(
            from_opt(self.out, float64_ok=True), extent=self._plot_bounds()
        )

        # define basic widget controls
        # D
        pupil_d = widgets.FloatSlider(
            value=self.D,
            min=2.0,
            max=pupil_d,
            step=0.1,
            description="Pupil's D:",
            style={'description_width': 'initial'},
            readout_format='.1f',
            continuous_update=False
        )
        # h
        opt_size = widgets.FloatSlider(
            value=self.opt_size,
            min=1.0,
            max=50.0,
            step=0.1,
            description="Optotype size h:",
            style={'description_width': 'initial'},
            readout_format='.1f',
            continuous_update=False
        )
        # d
        view_dist = widgets.IntSlider(
            value=self.view_dist,
            min=20,
            max=1000,
            description="Viewing dist d:",
            style={'description_width': 'initial'},
            continuous_update=False
        )
        self.angle_label = widgets.Label(self._angle_label_text())
        widget_controls = [
            widgets.HBox([self.angle_label]),
            widgets.HBox([pupil_d, widgets.Label(value='mm')]),
            widgets.HBox([opt_size, widgets.Label(value='cm')]),
            widgets.HBox([view_dist, widgets.Label(value='cm')]),
        ]

        # observe stuff
        pupil_d.observe(self.update_pupil, 'value')
        opt_size.observe(self.update_h, 'value')
        view_dist.observe(self.update_d, 'value')

        if mode == 'prescription':
            # define controls for A, C, S
            A_coeff = widgets.IntSlider(value=0.0, min=-180, max=180, description="A:")
            C_coeff = widgets.FloatSlider(
                value=0.0,
                min=-3.0,
                max=0.0,
                step=0.25,
                description="C:",
                readout_format='.2f',
            )
            S_coeff = widgets.FloatSlider(
                value=0.0,
                min=-20.0,
                max=20.0,
                step=0.25,
                description="S:",
                readout_format='.2f',
            )
            Vd_coeff = widgets.IntSlider(value=12, min=0, max=20, description="Vd:")
            A_coeff.continuous_update = False
            C_coeff.continuous_update = False
            S_coeff.continuous_update = False
            Vd_coeff.continuous_update = False
            prescription_widget_controls = [
                widgets.HBox([A_coeff, widgets.Label(value='°')]),
                widgets.HBox([C_coeff, widgets.Label(value='D')]),
                widgets.HBox([S_coeff, widgets.Label(value='D')]),
                widgets.HBox([Vd_coeff, widgets.Label(value='mm')]),
            ]

            # observe additional controls stuff
            A_coeff.observe(self.update_A, 'value')
            C_coeff.observe(self.update_C, 'value')
            S_coeff.observe(self.update_S, 'value')
            Vd_coeff.observe(self.update_Vd, 'value')

            controls = widgets.HBox([
                widgets.VBox(widget_controls, layout=make_vertical_layout()),
                widgets.VBox(prescription_widget_controls, layout=make_vertical_layout())
            ])
        else:
            controls = widgets.VBox(widget_controls, layout=make_vertical_layout())

        # setup controls layout
        controls.layout.border = '1px solid black'
        out_box = widgets.Box([output])
        out_box.layout = make_box_layout()

        # add to children
        self.children = [out_box, controls]

    def _plot_bounds(self):
        return [-self.betta / 2, self.betta / 2, -self.betta / 2, self.betta / 2]

    def _angle_label_text(self):
        return f'𝛼 = {self.alpha:.3f}, 𝛽 = {self.betta:.3f}'

    def redraw(self):
        self.psf_plot.set_data(from_psf(fft.scale(fft.shift(self.psf)), float64_ok=True))
        self.out_plot.set_data(from_opt(self.out, float64_ok=True))
        self.fig.canvas.draw()

    def update(self):
        self.psf = PSF(
            D=self.D,
            D0=self.D0,
            betta=self.betta,
            d=self.view_dist,
            W=self.wavefront,
            N=self.N,
        )
        self.out = retinal(self.src, np.fliplr(self.psf))
        self.redraw()

    def update_pupil(self, change):
        self.D = change.new
        self.update()

    def update_h(self, change):
        self.opt_size = change.new
        self.update_angle()

    def update_d(self, change):
        self.view_dist = change.new
        self.update_angle()

    def update_angle(self):
        alpha, betta = PSF_params(self.opt_size, self.k_ratio, self.view_dist)
        self.alpha = alpha
        self.betta = betta
        self.angle_label.value = self._angle_label_text()
        self.input_plot.set_extent(self._plot_bounds())
        self.psf_plot.set_extent(self._plot_bounds())
        self.out_plot.set_extent(self._plot_bounds())
        self.update()

    def update_A(self, change):
        if self.mode == 'prescription':
            self.wavefront._prescription['A'] = change.new
            self.update()

    def update_C(self, change):
        if self.mode == 'prescription':
            self.wavefront._prescription['C'] = change.new
            self.update()

    def update_S(self, change):
        if self.mode == 'prescription':
            self.wavefront._prescription['S'] = change.new
            self.update()

    def update_Vd(self, change):
        if self.mode == 'prescription':
            self.wavefront._prescription['Vd'] = change.new
            self.update()
