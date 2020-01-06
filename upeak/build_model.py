import argparse
from utils.model_generator import model_generator

def _parse_args():

    parser = argparse.ArgumentParser(description='custom model structure')
    parser.add_argument('-d', '--dims', help='input dimensions to model, tuple. Must be 2D currently', nargs=2, type=int)
    parser.add_argument('-o', '--output', help='path to save model', default='.')
    parser.add_argument('-c', '--classes', help='number of classes the model will learn. Must match input/output shape.', default=2, type=int)
    parser.add_argument('-s', '--steps', help='number of pooling and upsampling steps.', default=3, type=int)
    parser.add_argument('-l', '--layers', help='number of convolutional layers per step', default=2, type=int)
    parser.add_argument('-k', '--kernel', help='kernel size. can be list of length steps.', default=8, nargs='*', type=int)
    parser.add_argument('-f', '--filters', help='number of filters in first step.', default=8, type=int)
    parser.add_argument('-r', '--stride', help='stride for each kernel', default=1, type=int)
    parser.add_argument('-t', '--transfer', help='horz info transfer?', action='store_true')
    parser.add_argument('-p', '--padding', help='padding method to use', default='same', type=str)
    return parser.parse_args()

def _main():
    args = _parse_args()

    input_dims = (args.dims[0], args.dims[1], args.classes)
    print(args.kernel)

    model = model_generator(input_dims=input_dims, steps=args.steps, conv_layers=args.layers,
        filters=args.filters, kernel_size=args.kernel, strides=args.stride, transfer=args.transfer)

    Path(args.output).mkdir(parents=False, exist_ok=True)
    model_json = model.to_json()

    with open(join(path, 'model_structure.json'), 'w') as json_file:
        json_file.write(model_json)

if __name__ == "__main__":
    _main()