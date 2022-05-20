import argparse
from pathlib import Path
from tensorflow.keras.models import load_model, Model
from sweet.optimization.nn.nn import NNFactory
from sweet.optimization.nn_experiment import yaml_load



def parse_args():
    parser = argparse.ArgumentParser("Program name")
    parser.add_argument('-c', '--model_config', default='sweet/optimization/config.yaml', help="config of experiment (model)")
    parser.add_argument('-i', '--input_path', required=True, help="input model (e.g. h5 format)")
    parser.add_argument('-o', '--output_path', required=True, help="output model (e.g. dir format)")
    return parser.parse_args()


def main(input_path, output_path, model_config):
    if output_path.endswith('.h5'):
        print("Warning, output .h5 format is not recommended, please use tf directory format (no extension)")

    config = yaml_load(model_config)
    config['nn']['mix_stage2'] = True

    NN = NNFactory().make(config['nn'])
    NN_cut = Model(NN.input, NN.layers[-3].output)
    # NN_cut.summary()

    NN_cut.load_weights(input_path)
    NN_cut.save(output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_path, args.output_path, args.model_config)