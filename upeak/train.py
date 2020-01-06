from utils.model_generator import model_generator
from utils.loss import weighted_categorical_crossentropy
import argparse
import numpy as np
from utils.data_processing import load_data, DataGenerator
from utils.utils import save_model

def _parse_args():

    parser = argparse.ArgumentParser(description='model trainer')
    parser.add_argument('-t', '--traces', help='path to .npy file with raw traces', nargs='*')
    parser.add_argument('-l', '--labels', help='path to .npy file with labels', nargs='*')
    parser.add_argument('-o', '--output', help='path to save model weights', default='./output')
    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    parser.add_argument('-s', '--steps', default=500, type=int)
    parser.add_argument('-k', '--kernel', help='kernel size. can be list.', default=4) #this should be moved to define_model
    parser.add_argument('-r', '--stride', help='stride length', default=1, type=int) #this should be moved to define_model
    parser.add_argument('-f', '--filters', help='number of filters. can be list.', default=8) #this should be moved to define_model
    parser.add_argument('-m', '--model', help='path to custom model structure. other parameters overridden.')
    parser.add_argument('-p', '--optimizer', help='optimizer for model compilation', default='rmsprop')
    parser.add_argument('-w', '--weights', help='weights for loss function', default=None) 
    parser.add_argument('-a', '--augment', help='add this to include augmented data too', action='store_true')
    return parser.parse_args()

def _main():
    args = _parse_args()

    # load data
    traces, labels = load_data(args.traces, args.labels)
    training_set_generator = DataGenerator(traces, labels, batch_size=args.batch, steps=args.steps, augment=args.augment)
    input_dims = (training_set_generator[0][0].shape[1], training_set_generator[0][0].shape[2], labels.shape[2]) # (trace length, input dimension (1), output dimension (classes))

    if args.model is not None:
        # skip model generation and use previously made model structure
        # otherwise, just use default model
        model = keras.models.model_from_json(args.model)
        
    else:

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
        model = model_generator(input_dims=input_dims, filters=filters, kernel_size=kernel, strides=args.stride)

    # if no weights are provided, training is done with equal weights
    if args.weights is None:
        args.weights = [1.0 for i in range(0, labels.shape[2])]

    # model compilation
    # shoudl add options for different loss functions and different metrics
    model.compile(optimizer=args.optimizer, metrics=['accuracy'], loss=weighted_categorical_crossentropy(args.weights))

    model.fit_generator(generator=training_set_generator, epochs=args.epochs)

    save_model(model, path=args.output)

if __name__ == "__main__":
    _main()