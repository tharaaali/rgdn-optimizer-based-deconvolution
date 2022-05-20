import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib import __version__ as mpl_ver

import sweet.util.fft as fft
from sweet.transform.psf import PSF, PSF_params
from sweet.transform.filter import inverse_blur_filtering as ibf
from sweet.transform.hist import hist_clipping
from sweet.simulation import retinal as R
from sweet.util.wave import AberrationWavefront, PrescriptionWavefront
from sweet.util.viz.viz import std_imshow, prepare_img, autocontrast
from sweet.util.io import from_psf, to_psf, from_opt, to_opt

from .util import make_box_layout, make_margin_layout, make_vertical_layout
from .util import make_hist_cut_slider, separator
from .util import hist_bar, hist_bars
from .util import check_config, make_precompensaton_config


class PrecompensationPlots(widgets.Box):

    def __init__(self, input_image, config, wavefront):
        super().__init__()
        output = widgets.Output()

        check_config(config)

        # psf
        alpha, betta = PSF_params(
            config["opt_size_h"], config["canvas_opt_ratio_k"], config["view_dist"]
        )
        psf = PSF(
            config["D"],
            config["D0"],
            betta,
            config["view_dist"],
            wavefront,
            config["N"],
            norm=False
        )

        # precompensation
        compensated = ibf(input_image, psf, config["K_empirical"])
        scaled_compensated = R.scale(compensated, d=config["display_device_range"])
        clipped, clips = hist_clipping(
            scaled_compensated.copy(),
            clips=config["hist_clips"],
            d=config["display_device_range"]
        )

        psf = psf / psf.max()

        # retinal image simulation
        input_retinal = R.retinal(input_image, psf)
        precomp_retinal = R.retinal(scaled_compensated, psf)  # change to scaled_compensated?
        clipped_retinal = R.retinal(clipped, psf)

        # image fields & psf params
        self.src = input_image.copy()
        self.alpha = alpha
        self.betta = betta
        self.psf = psf
        self.precomp = scaled_compensated
        self.precomp_clipped = clipped
        self.distorted_retinal = input_retinal
        self.precomp_retinal = precomp_retinal
        self.precomp_retinal_clipped = clipped_retinal
        self.clips = clips

        with output:
            self.fig, axs = plt.subplots(
                3, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(8, 12)
            )

        # setup plot details
        axs[0, 0].set_title('Original image')
        axs[1, 0].set_title('Precompensated')
        axs[2, 0].set_title('Precompensated with clipping')
        axs[0, 1].set_title('Retinal image')

        self.fig.canvas.toolbar_position = 'bottom'
        self.fig.set_label('Retinal images comparison')

        self.src_plot = axs[0, 0].imshow(from_opt(input_image, float64_ok=True), extent=self._plot_bounds())
        self.compensated_plot = axs[1, 0].imshow(from_opt(scaled_compensated, float64_ok=True), extent=self._plot_bounds())
        self.clipped_plot = axs[2, 0].imshow(from_opt(clipped, float64_ok=True), extent=self._plot_bounds())

        self.retinal_plots = [
            axs[0, 1].imshow(from_opt(input_retinal, float64_ok=True), extent=self._plot_bounds()),
            axs[1, 1].imshow(from_opt(precomp_retinal, float64_ok=True), extent=self._plot_bounds()),
            axs[2, 1].imshow(from_opt(clipped_retinal, float64_ok=True), extent=self._plot_bounds()),
        ]

        # setup ui
        out_box = widgets.Box([output])
        out_box.layout = make_box_layout()
        # add to children
        self.children = [out_box]

    def _plot_bounds(self):
        return [-self.betta / 2, self.betta / 2, -self.betta / 2, self.betta / 2]

    def redraw(self):
        # set axis limits
        self.src_plot.set_extent(self._plot_bounds())
        self.compensated_plot.set_extent(self._plot_bounds())
        self.clipped_plot.set_extent(self._plot_bounds())
        self.retinal_plots[0].set_extent(self._plot_bounds())
        self.retinal_plots[1].set_extent(self._plot_bounds())
        self.retinal_plots[2].set_extent(self._plot_bounds())

        # update plots data
        self.compensated_plot.set_data(from_opt(self.precomp, float64_ok=True))
        self.clipped_plot.set_data(from_opt(self.precomp_clipped, float64_ok=True))
        self.retinal_plots[0].set_data(from_opt(self.distorted_retinal, float64_ok=True))
        self.retinal_plots[1].set_data(from_opt(self.precomp_retinal, float64_ok=True))
        self.retinal_plots[2].set_data(from_opt(self.precomp_retinal_clipped, float64_ok=True))
        self.fig.canvas.draw()

    def update(self, config, wavefront):

        check_config(config)

        # psf
        alpha, betta = PSF_params(config["opt_size_h"], config["canvas_opt_ratio_k"], config["view_dist"])
        psf = PSF(config["D"], config["D0"], betta, config["view_dist"], wavefront, config["N"], norm=False)

        # true_psf = np.fliplr(psf) - no flipping needed in opt mode

        # precompensation
        compensated = ibf(self.src, psf, config["K_empirical"])
        scaled_compensated = R.scale(compensated, d=config["display_device_range"])
        clipped, clips = hist_clipping(
            scaled_compensated.copy(),
            clips=config["hist_clips"],
            d=config["display_device_range"],
        )

        self.alpha = alpha
        self.betta = betta

        # we norm psf only after MTF/OTF are computed
        self.psf = psf / psf.max()
        self.precomp = scaled_compensated
        self.precomp_clipped = clipped
        self.clips = clips

        self.distorted_retinal = R.retinal(self.src, self.psf)
        self.precomp_retinal = R.retinal(scaled_compensated, self.psf)  # change to scaled compensated?
        self.precomp_retinal_clipped = R.retinal(clipped, self.psf)

        self.redraw()


class PSFPlot(widgets.Box):

    def __init__(self, psf, betta):
        super().__init__()
        output = widgets.Output()

        self.psf = psf.copy()
        self.betta = betta

        with output:
            self.fig, self.axs = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 4))

        # setup plot details
        self.fig.canvas.toolbar_position = 'bottom'
        self.fig.set_label('PSF')
        self.psf_plot = self.axs.imshow(
            from_psf(fft.shift(psf), float64_ok=True), extent=[-betta/2, betta/2, -betta/2, betta/2]
        )

        # setup ui
        out_box = widgets.Box([output])
        # add to children
        self.children = [out_box]

    def redraw(self):
        self.psf_plot.set_data(from_psf(fft.shift(self.psf), float64_ok=True))
        self.psf_plot.set_extent([
            -self.betta/2, self.betta/2, -self.betta/2, self.betta/2
        ])
        self.fig.canvas.draw()

    def update(self, psf, betta):
        self.psf = psf.copy()
        self.betta = betta
        self.redraw()


class HistPlots(widgets.Box):

    def __init__(self, source, compensated, clipped, clips):
        super().__init__()
        output = widgets.Output()

        assert len(source) == 2, f"Expected pair (input, retinal) for source"
        assert len(compensated) == 2, f"Expected pair (input, retinal) for compensated"
        assert len(clipped) == 2, f"Expected pair (input, retinal) for clipped"
        assert len(clips) == 2, f"Expected clips as pair of (low_clip, high_clip)"

        input_image, input_retinal = source
        precomp, precomp_retinal = compensated
        clipped, clipped_retinal = clipped
        cl, ch = clips

        with output:
            self.fig, self.axs = plt.subplots(2, 2, constrained_layout=True, figsize=(12, 6))

        # setup plot details
        self.axs[0, 0].set_title('Precompensated')
        self.axs[0, 1].set_title('Clipped')
        self.axs[1, 0].set_title('Simulated Retinal')
        self.axs[1, 1].set_title('Simulated Retinal')

        self.fig.canvas.toolbar_position = 'bottom'
        self.fig.set_label('Image Histograms')

        self.prec_bar, self.prec_retinal_bar = hist_bars(
            precomp, precomp_retinal, self.axs[0, 0], self.axs[1, 0]
        )
        self.clip_bar, self.clip_retinal_bar = hist_bars(
            clipped, clipped_retinal, self.axs[0, 1], self.axs[1, 1]
        )
        self.cl = self.axs[0, 0].axvline(x=cl, c='r')
        self.ch = self.axs[0, 0].axvline(x=ch, c='r')

        out_box = widgets.Box([output])
        out_box.layout = make_margin_layout()
        self.children = [out_box]

    def _check_hist(self, hist, new_freq):
        needs_update = False
        if mpl_ver >= '3.4.1':
            if (hist.datavalues != new_freq).any():
                needs_update = True
        else:
            if (np.asarray([bar.get_height() for bar in hist]) != new_freq).any():
                needs_update = True

        return needs_update

    def redraw(self):
        self.fig.canvas.draw()

    def update_precompensated(self, image, retinal):
        in_f, in_e = np.histogram(image, bins=256)
        out_f, out_e = np.histogram(retinal, bins=256)

        if self._check_hist(self.prec_bar, in_f):
            self.prec_bar.remove()
            self.prec_bar = hist_bar(in_f, in_e, self.axs[0, 0])
        if self._check_hist(self.prec_retinal_bar, out_f):
            self.prec_retinal_bar.remove()
            self.prec_retinal_bar = hist_bar(out_f, out_e, self.axs[1, 0])

    def update_clipped(self, image, retinal):
        in_f, in_e = np.histogram(image, bins=256)
        out_f, out_e = np.histogram(retinal, bins=256)

        if self._check_hist(self.clip_bar, in_f):
            self.clip_bar.remove()
            self.clip_bar = hist_bar(in_f, in_e, self.axs[0, 1])

        if self._check_hist(self.clip_retinal_bar, out_f):
            self.clip_retinal_bar.remove()
            self.clip_retinal_bar = hist_bar(out_f, out_e, self.axs[1, 1])

    def update_clip_lines(self, clips):
        cl, ch = clips
        if self.cl.get_xdata()[0] != cl:
            self.cl.set_xdata([cl, cl])
        if self.ch.get_xdata()[0] != ch:
            self.ch.set_xdata([ch, ch])

    def update(self, compensated, clipped, clips):
        precomp, precomp_retinal = compensated
        clipped, clipped_retinal = clipped
        self.update_precompensated(precomp, precomp_retinal)
        self.update_clipped(clipped, clipped_retinal)
        self.update_clip_lines(clips)
        self.redraw()


class PipelineWidget(widgets.VBox):
    def __init__(
        self,
        input_image,
        pupil_size=None,
        k_ratio=None,
        config=None,
        mode='prescription',
        zernike_coeffs=None
    ):
        super().__init__()

        assert len(input_image.shape) == 2 and input_image.shape[0] == input_image.shape[1], \
            f"Incorrect shape of input image: expected (NxN), got {input_image.shape}"
        assert mode in ('prescription', 'aberrometer'), f"Unknown mode: {mode}"

        if not config:
            assert pupil_size and k_ratio, \
                f"If config is not provided, `pupil_size` and `k_ratio` params should be specified"

            config = make_precompensaton_config(D0=pupil_size, k_ratio=k_ratio, N=input_image.shape[0])
        check_config(config)

        if mode == 'aberrometer':
            # zernike_coeffs != None
            # len(zernike_coeffs) > 0
            assert zernike_coeffs, "Provide zernike_coeffs for `aberrometer` mode"
            self.wavefront = AberrationWavefront(D0=config["D0"], coeffs=zernike_coeffs)
        else:
            self.wavefront = PrescriptionWavefront(D0=config["D0"], A=0.0, C=0.0, S=0.0)

        if type(config["hist_clips"]) != list:
            val = config["hist_clips"]
            config["hist_clips"] = [val] if type(val) != tuple else list(val)

        self.config = config
        self.mode = mode

        self.image_plots = PrecompensationPlots(input_image, config, self.wavefront)
        self.psf_plot = PSFPlot(self.image_plots.psf, self.image_plots.betta)
        self.hist_plots = HistPlots(
            source=(
                self.image_plots.src,
                self.image_plots.distorted_retinal
            ),
            compensated=(
                self.image_plots.precomp,
                self.image_plots.precomp_retinal
            ),
            clipped=(
                self.image_plots.precomp_clipped,
                self.image_plots.precomp_retinal_clipped
            ),
            clips=self.image_plots.clips
        )

        # basic widget controls
        # D
        pupil_d = widgets.FloatSlider(
            value=config["D"],
            min=2.0,
            max=config["D0"],
            step=0.1,
            description="Pupil's D:",
            style={'description_width': 'initial'},
            readout_format='.1f',
            continuous_update=False
        )
        # h
        opt_size = widgets.FloatSlider(
            value=config["opt_size_h"],
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
            value=config["view_dist"],
            min=20,
            max=1000,
            description="Viewing dist d:",
            style={'description_width': 'initial'},
            continuous_update=False
        )
        # Empirical const
        K = widgets.FloatSlider(
            value=config["K_empirical"],
            min=0.0,
            max=1.0,
            step=0.001,
            description="K:",
            readout_format='.3f',
            continuous_update=False
        )
        self.angle_label = widgets.Label(self._angle_label_text())

        widget_controls = [
            widgets.HBox([self.angle_label]),
            widgets.HBox([pupil_d, widgets.Label(value='mm')]),
            widgets.HBox([opt_size, widgets.Label(value='cm')]),
            widgets.HBox([view_dist, widgets.Label(value='cm')]),
            widgets.HBox([K])
        ]

        # observe stuff
        pupil_d.observe(self.update_pupil, 'value')
        opt_size.observe(self.update_h, 'value')
        view_dist.observe(self.update_d, 'value')
        K.observe(self.update_K, 'value')

        # histogram clipping controls
        cl = make_hist_cut_slider("CL:", self.config['hist_clips'][0])
        ch = make_hist_cut_slider("CH:", self.config['hist_clips'][-1])
        hist_controls = widgets.Accordion(
            [widgets.VBox([cl, ch])],
            selected_index=None,
        )
        hist_controls.set_title(0, "Histogram cutting thresholds")

        # observe hist controls stuff
        cl.observe(self.update_CL, 'value')
        ch.observe(self.update_CH, 'value')

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
            widget_controls += [
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

            # setup controls layout

        # setup controls layout
        controls = widgets.VBox(widget_controls, layout=make_vertical_layout())
        self.psf_plot.layout = make_margin_layout()
        left_column = widgets.VBox([
            self.psf_plot, separator(), controls, separator(), hist_controls
        ])
        left_column.layout = make_box_layout()
        upper_content = widgets.HBox([left_column, self.image_plots])
        self.hist_plots.layout = make_box_layout()

        # add to children
        self.children = [upper_content, self.hist_plots]

    def _angle_label_text(self):
        return f'𝛼 = {self.image_plots.alpha:.3f}, 𝛽 = {self.image_plots.betta:.3f}'

    def update(self):
        self.image_plots.update(self.config, self.wavefront)
        self.angle_label.value = self._angle_label_text()
        self.psf_plot.update(self.image_plots.psf, self.image_plots.betta)
        self.hist_plots.update(
            compensated=(
                self.image_plots.precomp,
                self.image_plots.precomp_retinal
            ),
            clipped=(
                self.image_plots.precomp_clipped,
                self.image_plots.precomp_retinal_clipped
            ),
            clips=self.image_plots.clips
        )

    def update_pupil(self, change):
        self.config['D'] = change.new
        self.update()

    def update_h(self, change):
        self.config["opt_size_h"] = change.new
        self.update()

    def update_d(self, change):
        self.config["view_dist"] = change.new
        self.update()

    def update_K(self, change):
        self.config['K_empirical'] = change.new
        self.update()

    def update_CL(self, change):
        clips = [change.new, self.config['hist_clips'][-1]]
        self.config['hist_clips'] = clips
        self.update()

    def update_CH(self, change):
        clips = [self.config['hist_clips'][0], change.new]
        self.config['hist_clips'] = clips
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
