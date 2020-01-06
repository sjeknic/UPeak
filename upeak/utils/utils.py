from os.path import join
from pathlib import Path

def save_model(model, path='./output'):
    Path(path).mkdir(parents=False, exist_ok=True)
    model_json = model.to_json()

    with open(join(path, 'model_structure.json'), 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(join(path, 'model_weights.h5'))
    model.save(join(path, 'complete_model.hd5'))