from utils.model_generator import model_generator
from utils.loss import weighted_categorical_crossentropy
import utils.augmenter as aug
import argparse
import numpy as np
from utils.data_processing import load_data, pick_positions

def _parse_args():

    parser = argparse.ArgumentParser(description='model trainer')
    parser.add_argument('-t', '--traces', help='path to .npy file with raw traces', nargs='*')
    parser.add_argument('-l', '--labels', help='path to .npy file with labels', nargs='*')
    parser.add_argument('-o', '--output', help='path to save model weights', default='.')
    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    parser.add_argument('-r', '--stepsepoch', default=100, type=int)
    parser.add_argument('-c', '--classes', default=3, type=int) #this should be inferred from input shape
    parser.add_argument('-k', '--kernel', help='kernel size. can be list.', default=8)
    parser.add_argument('-s', '--stride', help='stride length', default=1, type=int)
    parser.add_argument('-f', '--filters', help='number of filters. can be list.', default=8)
    parser.add_argument('-m', '--model', help='path to custom model structure. other parameters overridden.')
    parser.add_argument('-p', '--optimizer', help='optimizer for model compilation', default='rmsprop')
    parser.add_argument('-w', '--weights', help='weights for loss function', default=None)
    parser.add_argument('-a', '--augment', help='add this to include augmented data too', default=1)
    return parser.parse_args()

def _main():
    args = _parse_args()

    # load data
    traces, labels = load_data(args.traces, args.labels)

    print(traces.shape)
    print(labels.shape)

    # augment data, currently returns original data stacked on an array of amp+noise
    if not args.augment: #runs if augment is None or False
        traces = aug.augment(traces)

    if args.model is not None:
        # skip model generation and use previously made model structure
        pass

    # ensure filter and kernel are correct type
    if type(args.filters) == list:
        filters = [int(f) for f in args.filters]
    else:
        filters = int(args.filters)

    if type(args.kernel) == list:
        kernel = [int(k) for k in args.kernel]
    else:
        kernel = int(args.kernel)

    # generate model structure
    model = model_generator(classes=args.classes, filters=filters, kernel_size=kernel,
        strides=args.stride)

    # temporary for trouble shooting
    for l in model.layers:
        print(l.name)
        print(l.input_shape, l.output_shape)

    # if no weights are provided, training is done with equal weights
    if args.weights is None:
        args.weights = [1.0 for i in range(0, args.classes)]

    # model compilation
    # shoudl add options for different loss functions and different metrics
    model.compile(optimizer=args.optimizer, metrics=['accuracy'], loss=weighted_categorical_crossentropy(args.weights))



    model.fit(traces, labels, epochs=args.epochs, batch_size=args.batch)

if __name__ == "__main__":
    _main()