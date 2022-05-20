import numpy as np
import ipywidgets as widgets


# ---------------------------------------------
# ---- widget UI & layout
# ---------------------------------------------

def separator():
    return widgets.HTML(value="<hr>")


def make_box_layout():
    return widgets.Layout(
        border='solid 1px black',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px'
    )


def make_horizontal_layout():
    return widgets.Layout(
        display='flex',
        align_items='center',
        flex_flow='row',
        width='100%'
    )


def make_vertical_layout():
    return widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='center',
        width='100%'
    )


def make_margin_layout():
    return widgets.Layout(margin='0px 10px 10px 0px')


def make_hist_cut_slider(title, value=1/1000, max_value=0.01, continuous_update=False):
    return widgets.FloatSlider(
        value=value,
        min=0.0,
        max=max_value,
        step=0.0001,
        description=title,
        readout_format='.2%',
        continuous_update=continuous_update
    )


# ---------------------------------------------
# ---- hist utils
# ---------------------------------------------

def hist_bar(freq, edges, axe):
    assert len(edges) == len(freq)+1
    bar = axe.bar(edges[:-1], freq, width=np.diff(edges), align='edge', color='b')
    axe.set_ylim(0, np.max(freq)*1.05)
    axe.set_xlim(-0.15, 1.15)
    return bar


def hist_bars(input_image, retinal_image, inaxe, oaxe):
    ifreq, iedges = np.histogram(input_image, bins=256)
    ofreq, oedges = np.histogram(retinal_image, bins=256)
    in_bar = hist_bar(ifreq, iedges, inaxe)
    out_bar = hist_bar(ofreq, oedges, oaxe)
    return in_bar, out_bar


# ---------------------------------------------
# ---- config utils
# ---------------------------------------------

def check_config(config: dict):
    for key in config.keys():
        assert key in (
            "D", "D0", "opt_size_h", "canvas_opt_ratio_k", "view_dist", "N",  # psf params
            "K_empirical",  # precompensation params
            "hist_clips", "display_device_range",  # clipping params
        )

    # TODO: Add all params values check?
    assert config["D"] >= 1.0, f"Expected D >= 1.0, got {config['D']}"
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


def make_precompensaton_config(
    D0: float,
    k_ratio: float,
    D: float = 2.0,
    opt_size_h: float = 1.0,
    view_dist: float = 75,
    N: int = 1024,
    K_empirical: float = 0.1,
    hist_clips: float or tuple or list = 1/1000,
    display_device_range: tuple = (0, 1)
):
    config = dict()
    config["D"] = D
    config["D0"] = D0
    config["opt_size_h"] = opt_size_h
    config["canvas_opt_ratio_k"] = k_ratio
    config["view_dist"] = view_dist
    config["N"] = N
    config["K_empirical"] = K_empirical
    config["hist_clips"] = hist_clips
    config["display_device_range"] = display_device_range

    check_config(config)

    return config
