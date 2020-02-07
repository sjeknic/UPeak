import json
import argparse
import numpy as np
from utils.data_processing import load_data, DataGenerator, gen_train_test
from utils.augmenter import _augment, _normalize
from utils.utils import save_model, _parse_inputs
import keras
from keras.models import model_from_json
from keras import callbacks
from utils.model_generator import model_generator
from utils.loss import weighted_categorical_crossentropy
from os.path import join
from pathlib import Path
from _setting import FRAC_TEST, VAL_STEPS
from _setting import AUG_FUNCS, AUG_OPTIONS, AUG_METHOD
from _setting import NORM_FUNCS, NORM_OPTIONS, NORM_METHOD

def define_callbacks(output_path):
    csv_logger = callbacks.CSVLogger(join(output_path, 'training.log'))
    earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    fpath = join(output_path, 'weights.{epoch:02d}-{val_loss:.2f}-{val_categorical_accuracy:.2f}.hdf5')
    cp_cb = callbacks.ModelCheckpoint(filepath=fpath, monitor='loss', save_best_only=True)
    return [csv_logger, earlystop, cp_cb]

def _parse_args():

    parser = argparse.ArgumentParser(description='model trainer')
    parser.add_argument('-t', '--traces', help='path to .npy file with raw traces', nargs='*')
    parser.add_argument('-l', '--labels', help='path to .npy file with labels', nargs='*')
    parser.add_argument('-o', '--output', help='path to save model weights', default='./output')
    parser.add_argument('-m', '--model', help='path to custom model structure. otherwise default.')
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    parser.add_argument('-s', '--steps', default=1000, type=int)
    parser.add_argument('-w', '--weights', help='weights for loss function', default=None, nargs='*', type=float) 
    parser.add_argument('-p', '--optimizer', help='optimizer for model compilation', default='rmsprop')
    parser.add_argument('-a', '--augment', help='add this to include augmented data too. Set options in _setting.py', action='store_true')
    parser.add_argument('-n', '--normalize', help='add this to include normalization of data. Set options in _setting.py', action='store_true')
    return parser.parse_args()

def _main():
    args = _parse_args()

    # Load data, pick training set, filter non_responders
    traces = _parse_inputs(args.traces)
    labels = _parse_inputs(args.labels)
    traces, labels = load_data(traces, labels)
    train_traces, train_labels, test_traces, test_labels = gen_train_test(traces, labels, FRAC_TEST)

    # Apply augmentation and normalization
    if args.augment:
        train_traces, train_labels = _augment(AUG_FUNCS, AUG_OPTIONS, AUG_METHOD, train_traces, train_labels)

    if args.normalize:
        train_traces = _normalize(NORM_FUNCS, NORM_OPTIONS, NORM_METHOD, train_traces)
        test_traces = _normalize(NORM_FUNCS, NORM_OPTIONS, NORM_METHOD, test_traces)

    input_features = train_traces.shape[2]
    output_classes = train_labels.shape[2]

    # Build model
    if args.model is not None:
        # skip model generation and use previously made model structure
        with open(args.model, 'r') as json_file:
            od = json.load(json_file)

        input_dims = (od['dims'][0], input_features, output_classes)

        model = model_generator(input_dims=input_dims, steps=od['steps'], conv_layers=od['layers'],
            filters=od['filters'], kernel_size=od['kernel'], strides=od['stride'], transfer=od['transfer'],
            activation=od['activation'], padding=od['padding'])
    else:
        # generate default model structure
        input_dims = (128, 1, classes) #default is len 64, dim 1, classes 3
        model = model_generator(input_dims=input_dims)

    # Make data generators
    training_set_generator = DataGenerator(train_traces, train_labels, length=input_dims[0], batch_size=args.batch, steps=args.steps)
    test_set_generator = DataGenerator(test_traces, test_labels, length=input_dims[0], batch_size=VAL_STEPS, steps=1)
    test_t, test_l = test_set_generator[0] # set static training set

    # If no weights are provided, training is done with equal weights
    if args.weights is None:
        args.weights = [1.0 for i in range(0, labels.shape[2])]

    # Compile
    # Currently only one loss function is possible
    model.compile(optimizer=args.optimizer, metrics=[keras.metrics.categorical_accuracy], loss=weighted_categorical_crossentropy(args.weights))

    # Define callbacks
    Path(args.output).mkdir(parents=False, exist_ok=True)
    cb = define_callbacks(args.output)

    # Fit model
    model.fit_generator(generator=training_set_generator, epochs=args.epochs, validation_data=(test_t, test_l), callbacks=cb)

    # Save model
    save_model(model, path=args.output)

if __name__ == "__main__":
    _main()