import argparse
from utils.data_processing import load_data, pad_traces
from tensorflow.keras.layers import Input
from pathlib import Path
from os.path import join
import json
from utils.model_generator import model_generator
import numpy as np
from utils.plotting import display_results, save_figures
from _setting import NORM_FUNCS, NORM_OPTIONS, NORM_METHOD
from _setting import PAD_MODE, PAD_CV
from _setting import PRED_FNAME
from utils.augmenter import _normalize
from utils.utils import _parse_inputs

def _parse_args():
    parser = argparse.ArgumentParser(description='model predictions')
    parser.add_argument('-t', '--traces', help='path to .npy file with raw traces', nargs='*')
    parser.add_argument('-o', '--output', help='path to save model predictions', default='./output')
    parser.add_argument('-m', '--model', help='path to custom model structure dictionary json. otherwise default.')
    parser.add_argument('-w', '--weights', help='path to weights for model')
    parser.add_argument('-n', '--normalize', help='add this to include normalization of data. Set options in _setting.py', action='store_true')
    parser.add_argument('-c', '--classes', help='number of classes. must match model', default=3, type=int)
    parser.add_argument('-d', '--display', help='display figures with peak predictions. row col for fig display.', action='store_true')
    parser.add_argument('-s', '--save', help='if path provided, will save figure', default=None, type=str)
    return parser.parse_args()

def _main():
    args = _parse_args()

    traces = _parse_inputs(args.traces)
    traces, _ = load_data(traces)
    plot_traces = np.copy(traces)
    
    if args.normalize:
        traces = _normalize(NORM_FUNCS, NORM_OPTIONS, NORM_METHOD, traces)

    if args.model is not None:
        # recreate model that was used during training with new input layer

        with open(args.model, 'r') as json_file:
            od = json.load(json_file)

        traces = pad_traces(traces, od['steps'], pad_mode=PAD_MODE, cv=PAD_CV)
        input_dims = (traces.shape[1], traces.shape[2], args.classes)

        model = model_generator(input_dims=input_dims, steps=od['steps'], conv_layers=od['layers'],
            filters=od['filters'], kernel_size=od['kernel'], strides=od['stride'], transfer=od['transfer'],
            activation=od['activation'], padding=od['padding'])
    else:
        # generate default model structure
        traces = pad_traces(traces, 2, pad_mode=PAD_MODE, cv=PAD_CV)
        input_dims = (traces.shape[1], traces.shape[2], args.classes)
        model = model_generator(input_dims=input_dims)

    #load weights
    model.load_weights(args.weights)
    
    #apply predictions and trim to size
    result = model.predict(traces)
    result = result[:, :plot_traces.shape[1], :]
    
    Path(args.output).mkdir(parents=False, exist_ok=True)
    np.save(join(args.output, '{0}.npy'.format(PRED_FNAME)), result)

    if args.save is not None:
        save_figures(plot_traces[:, :, 0], result, args.save)

    if args.display:
        display_results(plot_traces[:, :, 0], result)

if __name__ == '__main__':
    _main()