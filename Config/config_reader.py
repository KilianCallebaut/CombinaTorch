import json
import os

dataset_options = [
    'asvspoof',
    'chen',
    'dcaseScene',
    'fsdkaggle',
    'ravdess',
    'speechcommands',
    'dcaseEvents'
]

network_options = [
    'cnn',
    'dnn'
]


def write_config(name: str, params: dict):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    if 'extraction_method' in params.keys():
        params['extraction_method'] = params['extraction_method'].name
    with open(os.path.join(dir_path,
                           'configs/{}.json'.format(name)), 'w') as outfile:
        json.dump(params, outfile)


def read_config(name: str):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    with open(os.path.join(dir_path,
                           'configs/{}.json'.format(name))) as outfile:
        data = json.load(outfile)
        if 'split' in data.keys():
            data['test_size'] = read_config('test_size_split') if data.pop('split') else read_config('test_size_val')
            data['test_size'] = data['test_size']['test_size']
        return data


def write_preparation_params(name: str, split: bool, window: bool, dic_of_labels_limits: dict):
    name_comp = 'preparation_params_{}'.format(name)
    param = dict(
        split=split,
        window=window,
        dic_of_labels_limits=dic_of_labels_limits
    )
    write_config(name_comp, param)
