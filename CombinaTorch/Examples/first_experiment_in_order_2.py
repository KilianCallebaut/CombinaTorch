from __future__ import print_function, division

import itertools
import sys

import torch

from CombinaTorch.Config.config_reader import *
from CombinaTorch.DataReaders.ASVspoof2015 import ASVspoof2015
from CombinaTorch.DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from CombinaTorch.DataReaders.DCASE2017_SE import DCASE2017_SE
from CombinaTorch.DataReaders.DCASE2017_SS import DCASE2017_SS
from CombinaTorch.DataReaders.ExtractionMethod import MelSpectrogramExtractionMethod, MFCCExtractionMethod
from CombinaTorch.DataReaders.FSDKaggle2018 import FSDKaggle2018
from CombinaTorch.DataReaders.Ravdess import Ravdess
from CombinaTorch.DataReaders.SpeechCommands import SpeechCommands
from CombinaTorch.MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from CombinaTorch.MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from CombinaTorch.MultiTask.MultiTaskModelFactory import MultiTaskModelFactory
from CombinaTorch.Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from CombinaTorch.Training.Training import Training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'E'
data_base = r'E:\Thesis_Results\Data_Readers'


def main(argv):
    print('--------------------------------------------------')
    print('test loop')
    print('--------------------------------------------------')

    extraction_params = {
        "extraction_params": {
            "n_fft": 2048,
            "hop_length": 256,
            "n_mels": 128,
            "window": "hann"
        },
    }

    meta_params = {
        "batch_size": 16,
        "num_epochs": 50,
        "learning_rate": 0.001
    }

    mtmf = MultiTaskModelFactory()
    mtmf.add_modelclass(MultiTaskHardSharingConvolutional)
    mtmf.add_static_model_parameters(MultiTaskHardSharingConvolutional.__name__,
                                     hidden_size=64,
                                     n_hidden=4,
                                     input_channels=1)
    mtmf.add_modelclass(MultiTaskHardSharing)
    mtmf.add_static_model_parameters(MultiTaskHardSharing.__name__,
                                     **{"hidden_size": 512, "n_hidden": 4})

    csc = ConcatTrainingSetCreator(random_state=123,
                                   nr_runs=4,
                                   index_mode=False,
                                   recalculate=False)
    csc.add_data_reader(ASVspoof2015(object_path=os.path.join(data_base, 'ASVspoof2015_{}'),
                                     data_path=os.path.join(drive + ":",
                                                            r"Thesis_Datasets\Automatic Speaker Verification Spoofing "
                                                            r"and Countermeasures Challenge 2015\DS_10283_853"),
                                     ))
    csc.add_data_reader(
        ChenAudiosetDataset(object_path=os.path.join(data_base, 'ChenAudiosetDataset'),
                            data_path=os.path.join(drive + r':\Thesis_Datasets\audioset_chen\audioset_filtered'),
                            )
    )
    csc.add_data_reader(DCASE2017_SS(object_path=os.path.join(data_base, 'DCASE2017_SS_{}'),
                                     data_path=os.path.join(drive + ":", r'Thesis_Datasets\DCASE2017')))
    csc.add_data_reader(DCASE2017_SE(object_path=os.path.join(data_base, 'DCASE2017_SE_{}'),
                                     data_path=os.path.join(drive + ":", 'Thesis_Datasets\\DCASE2017'),
                                     ))
    csc.add_data_reader(FSDKaggle2018(object_path=os.path.join(data_base, 'FSDKaggle2018_{}'),
                                      data_path=os.path.join(drive + ":",
                                                             r'Thesis_Datasets\FSDKaggle2018\freesound-audio-tagging'),
                                      ))
    csc.add_data_reader(Ravdess(object_path=os.path.join(data_base, 'Ravdess'),
                                data_path=os.path.join(drive + ':', r"Thesis_Datasets\Ravdess"),
                                ))
    csc.add_data_reader(Ravdess(object_path=os.path.join(data_base, 'Ravdess'),
                                data_path=os.path.join(drive + ':', r"Thesis_Datasets\Ravdess"),
                                mode=1))
    csc.add_data_reader(SpeechCommands(object_path=os.path.join(data_base, 'SpeechCommands_{}'),
                                       data_path=os.path.join(drive + ":", r'Thesis_Datasets\SpeechCommands'),
                                       ))

    csc.add_signal_preprocessing(preprocess_dict=dict(resample_to=32000, mono=True))
    csc.add_transformation_call('prepare_fit')
    csc.add_transformation_call('prepare_inputs')
    csc.add_transformation_call('normalize_fit')
    csc.add_transformation_call('normalize_inputs')

    mtmf = MultiTaskModelFactory()
    mtmf.add_modelclass(MultiTaskHardSharingConvolutional)
    mtmf.add_static_model_parameters(MultiTaskHardSharingConvolutional.__name__,
                                     hidden_size=64,
                                     n_hidden=4,
                                     input_channels=1)
    mtmf.add_modelclass(MultiTaskHardSharing)
    mtmf.add_static_model_parameters(MultiTaskHardSharing.__name__,
                                     **{"hidden_size": 512, "n_hidden": 4})
    for ex in range(2):
        csc.add_extraction_method(
            MelSpectrogramExtractionMethod(**extraction_params)) if ex == 0 else csc.add_extraction_method(
            MFCCExtractionMethod(**extraction_params))


        keys = list(csc.get_keys())
        comb_iterator = itertools.chain(*map(lambda x: itertools.combinations(keys, x), range(0, 2)))

        for combo in comb_iterator:
            key_list = list(combo)
            for train, test, fold in csc.generate_training_splits(key_list):
                for i in range(2):
                    model = mtmf.create_model(MultiTaskHardSharing.__name__,
                                              input_size=train.datasets[0].get_input(0).flatten().shape[0],
                                              task_list=train.get_task_list()) if i == 1 else mtmf.create_model(
                        MultiTaskHardSharingConvolutional.__name__,
                        task_list=train.get_task_list())

                    print('Model Created')

                    results = Training.create_results(modelname=model.name,
                                                      task_list=train.get_task_list(),
                                                      fold=fold,
                                                      results_path=drive + r":\Thesis_Results",
                                                      num_epochs=meta_params['num_epochs'])

                    Training.run_gradient_descent(model=model,
                                                  concat_dataset=train,
                                                  results=results,
                                                  batch_size=meta_params['batch_size'],
                                                  num_epochs=meta_params['num_epochs'],
                                                  learning_rate=meta_params['learning_rate'],
                                                  test_dataset=test)


# tensorboard --logdir F:\Thesis_Results\Training_Results\experiments
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
