from sweet.optimization.ds.ds import DSFactory
from sweet.optimization.nn.nn import NNFactory
from sweet.optimization.trainer.trainer import TrainerFactory
from sweet.optimization.loss.loss import LossFactory
from sweet.util.viz.viz import viz_dict
from sweet.util.io import from_opt, from_psf
from sweet.util.fft import shift
from sweet.sweet import _Core

import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import os
import yaml
from pathlib import Path
import imageio
import random



def yaml_load(fname):
    with open(fname) as f:
        data = yaml.safe_load(f)
    return data


def yaml_save(fname, data):
    with open(fname, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def json_load(fname):
    with open(fname) as f:
        data = json.load(f)
    return data


def parse_args():
    parser = argparse.ArgumentParser("NN training experiment pipeline")
    parser.add_argument('-c', '--config', required=True, help="configuration file")
    parser.add_argument('-o', '--out-dir', help="output dir (no output by default)")
    # parser.add_argument('-a', '--all', action='store_true', help="allname")
    return parser.parse_args()


def make_report(out_dir, imgs, kers, results):
    os.makedirs(str(out_dir), exist_ok=False)

    for i, (img, ker, res) in enumerate(zip(imgs, kers, results)):
        blurred = _Core.calc_blur(img, ker)

        # first_stage = _Core.calc_precompensated(img, ker, stage=1)
        second_stage = _Core.calc_precompensated(img, ker, stage=2)
        blurred_second_stage = _Core.calc_blur(second_stage, ker)

        res_img = viz_dict({
            'Modelled Blur': from_opt(blurred, float64_ok=True),
            'Modelled Stage II': from_opt(blurred_second_stage, float64_ok=True),
            'Modelled NN': from_opt(res),
            'Original Image': from_opt(img),
            'PSF': from_psf(shift(ker), float64_ok=True),
        }, return_image=True, figsize=(15, 10), cols=3)
        imageio.imsave(str(out_dir/f"{i}.png"), res_img)


def show_plots(out_dir, keras_history):
    for k, v in keras_history.history.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.savefig(str(out_dir/f"plots.png"))


def run_experiment(config_path, out_dir=None):
    config = yaml_load(config_path)
    if config['ds'].get('seed') == 'random':
        # hack to have different reproducible results of the same config
        config['ds']['seed'] = random.randint(0, 2**32-1)

    ds = DSFactory().make(config['ds'])
    loss = LossFactory().make(config['loss'])
    nn = NNFactory().make(config['nn'])
    trainer = TrainerFactory().make(config['trainer'])

    keras_history = trainer.train(nn, ds, loss)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=False)
        out_dir = Path(out_dir)
        yaml_save(str(out_dir/'config.yaml'), config)

        imgs, kers = ds.get_some_val_data()
        nn_res = np.array(nn((imgs, kers)))

        nn.save(str(out_dir / 'nn.h5'))

        show_plots(out_dir, keras_history)
        assert np.all(~np.isnan(nn_res)), 'Error! There are NaN values!'
        assert np.all(nn_res >= 0), 'Error! There are negative values!'
        assert np.all(nn_res <= 1), 'Error! There are values > 1'
        make_report(out_dir / 'report_imgs', imgs[..., 0], kers[..., 0], nn_res[..., 0], )


    return nn

    # TODO:
    # * predownload data with wget ... and check the downloading speed
    # * save some imgs example to the directory for the auto-report
    # * save the model to the directory





if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.config, args.out_dir)