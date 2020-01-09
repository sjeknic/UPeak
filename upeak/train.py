import json
import argparse
import numpy as np
from utils.data_processing import load_data, DataGenerator, gen_train_test
from utils.utils import save_model
import keras
from keras.models import model_from_json
from keras import callbacks
from utils.model_generator import model_generator
from utils.loss import weighted_categorical_crossentropy
from os.path import join
from pathlib import Path

def define_callbacks(output_path):
    csv_logger = callbacks.CSVLogger(join(output_path, 'training.log'))
    # earlystop = callbacks.EarlyStopping(monitor='loss', patience=2)
    fpath = join(output_path, 'weights.{epoch:02d}-{loss:.2f}-{categorical_accuracy:.2f}.hdf5')
    cp_cb = callbacks.ModelCheckpoint(filepath=fpath, monitor='loss', save_best_only=True)
    return [csv_logger, cp_cb]

def _parse_args():

    parser = argparse.ArgumentParser(description='model trainer')
    parser.add_argument('-t', '--traces', help='path to .npy file with raw traces', nargs='*')
    parser.add_argument('-l', '--labels', help='path to .npy file with labels', nargs='*')
    parser.add_argument('-o', '--output', help='path to save model weights', default='./output')
    parser.add_argument('-m', '--model', help='path to custom model structure. otherwise default.')
    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    parser.add_argument('-s', '--steps', default=500, type=int)
    parser.add_argument('-w', '--weights', help='weights for loss function', default=None, nargs='*', type=float) 
    parser.add_argument('-f', '--frac', help='fraction to use as test set', default=0.2, type=float)
    parser.add_argument('-p', '--optimizer', help='optimizer for model compilation', default='rmsprop')
    parser.add_argument('-a', '--augment', help='add this to include augmented data too', action='store_true')
    return parser.parse_args()

def _main():
    args = _parse_args()

    # load data
    traces, labels = load_data(args.traces, args.labels)
    train_traces, train_labels, test_traces, test_labels = gen_train_test(traces, labels, args.frac)

    if args.model is not None:
        # skip model generation and use previously made model structure
        with open(args.model, 'r') as json_file:
            od = json.load(json_file)

        input_dims = (od['dims'][0], od['dims'][1], od['classes'])

        model = model_generator(input_dims=input_dims, steps=od['steps'], conv_layers=od['layers'],
            filters=od['filters'], kernel_size=od['kernel'], strides=od['stride'], transfer=od['transfer'],
            activation=od['activation'], padding=od['padding'])
    else:
        # generate default model structure
        input_dims = (64, 1, 3) #default is len 64, dim 1, classes 3
        model = model_generator(input_dims=input_dims)

    #make data generators
    training_set_generator = DataGenerator(train_traces, train_labels, length=input_dims[0], batch_size=args.batch, steps=args.steps, augment=args.augment)
    test_set_generator = DataGenerator(test_traces, test_labels, length=input_dims[0], batch_size=args.batch, steps=args.steps, augment=False)
    test_t, test_l = test_set_generator[0]

    # if no weights are provided, training is done with equal weights
    if args.weights is None:
        args.weights = [1.0 for i in range(0, labels.shape[2])]

    # should add options for different loss functions and different metrics
    model.compile(optimizer=args.optimizer, metrics=[keras.metrics.categorical_accuracy], loss=weighted_categorical_crossentropy(args.weights))

    #define callbacks at each epoch
    Path(args.output).mkdir(parents=False, exist_ok=True)
    cb = define_callbacks(args.output)

    model.fit_generator(generator=training_set_generator, epochs=args.epochs, validation_data=(test_t, test_l), callbacks=cb)

    save_model(model, path=args.output)

if __name__ == "__main__":
    _main()