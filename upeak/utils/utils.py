from os.path import join, isdir, isfile
from pathlib import Path
from glob import glob
from _setting import WEIGHT_FNAME

def save_model(model, path='./output'):
    Path(path).mkdir(parents=False, exist_ok=True)
    model_json = model.to_json()

    with open(join(path, 'model_structure.json'), 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(join(path, '{0}.h5'.format(WEIGHT_FNAME)), save_format='h5')
    model.save_weights(join(path, '{0}.tf'.format(WEIGHT_FNAME)), save_format='tf')
    model.save(join(path, 'complete_model.hd5'))

def _parse_inputs(inputs):
    '''
    returns sorted list of inputs. directories are replaced with sorted list of all files in the directory
    in the future, could add support for nested directories. Probably want to use os.walk or something.
    '''
    output = []
    for inp in inputs:
        if isfile(inp):
            output.append(inp)
        elif isdir(inp):
            for g in sorted(glob(join(inp, '*'))):
                if isfile(g):
                    output.append(g)

    return output
