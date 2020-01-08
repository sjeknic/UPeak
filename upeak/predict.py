import argparse
from utils.data_processing import load_data
from keras.layers import Input
from pathlib import Path
from os.path import join
import json
from utils.model_generator import model_generator
import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser(description='model predictions')
    parser.add_argument('-t', '--traces', help='path to .npy file with raw traces', nargs='*')
    parser.add_argument('-o', '--output', help='path to save model predictions', default='./output')
    parser.add_argument('-s', '--structure', help='path to custom model structure dictionary json. otherwise default.')
    parser.add_argument('-w', '--weights', help='path to weights for model')
    parser.add_argument('-m', '--model', help='path to complete model to use for predicting')
    parser.add_argument('-c', '--classes', help='number of classes. must match model', default=3, type=int)
    parser.add_argument('-d', '--display', help='display figures with peak predictions', default=None)
    return parser.parse_args()

def _main():
    args = _parse_args()

    traces, _ = load_data(args.traces)
    input_dims = (traces.shape[1], 1, args.classes)

    if args.structure is not None:
        # recreate model that was used during training with new input layer
        # note that some dimension agreement is probably necessary

        with open(args.structure, 'r') as json_file:
            od = json.load(json_file)

        model = model_generator(input_dims=input_dims, steps=od['steps'], conv_layers=od['layers'],
            filters=od['filters'], kernel_size=od['kernel'], strides=od['stride'], transfer=od['transfer'],
            activation=od['activation'], padding=od['padding'])
    else:
        # generate default model structure
        model = model_generator(input_dims=input_dims)

    #load weights
    model.load_weights(args.weights)
    
    #apply predictions
    result = model.predict(traces)

    Path(args.output).mkdir(parents=False, exist_ok=True)
    np.save(join(args.output, 'predictions.npy'), result)

if __name__ == '__main__':
    _main()