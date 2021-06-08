import gzip, pickle
import numpy as np
import json
import os
import cv2

def get_dataset_path():
    return "/misc/lmbraid18/zimmermc/datasets/FreiHAND_full/"

def load_ckpt(model, pretrained_dict):
    model_dict = model.state_dict()
    overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # only keys that are in the model
    overlap_dict = {k: v for k, v in overlap_dict.items() if np.all(v.shape == model_dict[k].shape)} # only when the shape matches

    if len(model_dict) != len(overlap_dict):
        print('Missing/Not Matching weights:')
        for k, v in model_dict.items():
            if k not in overlap_dict.keys():
                print(k, 'model:', v.shape)
    print(f'Given {len(pretrained_dict)} weights for {len(model_dict)} model weights. Loaded {len(overlap_dict)} matching weights!')
    if len(overlap_dict) == 0:
        for k, v in pretrained_dict.items():
            print('pretrained content', k, v.shape)
        for k, v in model_dict.items():
            print('model', k, v.shape)
        raise Expection('Not weights were loaded. This indicates and error.')

    model_dict.update(overlap_dict)
    model.load_state_dict(model_dict)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)

        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def json_dump(file_name, data, pretty_format=False, overwrite=True, verbose=False):
    msg = 'File does exists and should not be overwritten: %s' % file_name
    assert not os.path.exists(file_name) or overwrite, msg

    with open(file_name, 'w') as fo:
        if pretty_format:
            json.dump(data, fo, cls=NumpyEncoder, sort_keys=True, indent=4)
        else:
            json.dump(data, fo, cls=NumpyEncoder)

    if verbose:
        print('Dumped %d entries to file %s' % (len(data), file_name))


def json_load(file_name):
    with open(file_name, 'r') as fi:
        data = json.load(fi)
    return data


