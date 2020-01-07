import json
import argparse
import numpy as np
from utils.data_processing import load_data, DataGenerator, gen_train_test
from utils.utils import save_model
from keras.models import model_from_json
from utils.model_generator import model_generator
from utils.loss import weighted_categorical_crossentropy

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
    parser.add_argument('-f', '--frac', help='fraction to use as test set', default=0.1, type=float)
    parser.add_argument('-p', '--optimizer', help='optimizer for model compilation', default='rmsprop')
    parser.add_argument('-a', '--augment', help='add this to include augmented data too', action='store_true')
    return parser.parse_args()

def _main():
    args = _parse_args()

    # load data
    traces, labels = load_data(args.traces, args.labels)
    train_traces, train_labels, test_traces, test_labels = gen_train_test(traces, labels, args.frac)

    #make data generators
    training_set_generator = DataGenerator(train_traces, train_labels, batch_size=args.batch, steps=args.steps, augment=args.augment)
    test_set_generator = DataGenerator(test_traces, test_labels, batch_size=args.batch, steps=args.steps, augment=args.augment)
    input_dims = (training_set_generator[0][0].shape[1], training_set_generator[0][0].shape[2], labels.shape[2]) # (trace length, input dimension (1), output dimension (classes))

    if args.model is not None:
        # skip model generation and use previously made model structure
        with open(args.model, 'r') as json_file:
            model = model_from_json(json_file.read())
    else:
        # generate default model structure
        model = model_generator(input_dims=input_dims)

    # if no weights are provided, training is done with equal weights
    if args.weights is None:
        args.weights = [1.0 for i in range(0, labels.shape[2])]

    # should add options for different loss functions and different metrics
    model.compile(optimizer=args.optimizer, metrics=['accuracy'], loss=weighted_categorical_crossentropy(args.weights))

    model.fit_generator(generator=training_set_generator, epochs=args.epochs, validation_data=test_set_generator,
        validation_steps=test_traces.shape[0])

    save_model(model, path=args.output)

if __name__ == "__main__":
    _main()