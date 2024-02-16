import itertools
import os
import sys

from torch import optim
from torch.utils.data import DataLoader

from CombinaTorch.DataReaders.ASVspoof2015 import ASVspoof2015
from CombinaTorch.DataReaders.DCASE2017_SS import DCASE2017_SS
from CombinaTorch.DataReaders.ExtractionMethod import PerCelStandardizing, LogbankSummaryExtraction, \
    NeutralExtractionMethod, LogbankExtraction, SummaryPreparation
from CombinaTorch.DataReaders.Ravdess import Ravdess
from CombinaTorch.MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from CombinaTorch.MultiTask.MultiTaskModelFactory import MultiTaskModelFactory
from CombinaTorch.Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from CombinaTorch.Training.Training import Training

drive = r"E:/"


def main(argv):
    csc = ConcatTrainingSetCreator(nr_runs=5,
                                   # recalculate=True,
                                   multiply=False
                                   )

    csc.add_data_reader(Ravdess(object_path=drive + r'Thesis_Results\Data_Readers\Ravdess',
                                data_path=drive + r'Thesis_Datasets\Ravdess'))
    csc.add_data_reader(Ravdess(object_path=drive + r'Thesis_Results\Data_Readers\Ravdess',
                                data_path=drive + r'Thesis_Datasets\Ravdess',
                                mode=1), name='Ravdess_stress')
    csc.add_data_reader(ASVspoof2015(object_path=os.path.join(drive, r'Thesis_Results\Data_Readers\ASVspoof2015'),
                                     data_path=drive + r'Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015\DS_10283_853'))
    csc.add_data_reader(DCASE2017_SS(object_path=drive + r'Thesis_Results\Data_Readers\DCASE2017_SS_{}',
                                     data_path=drive + r'Thesis_Datasets\DCASE2017'))
    csc.add_signal_preprocessing(dict(resample_to=8000, mono=True))
    csc.add_extraction_method(extraction_method=LogbankExtraction(
        SummaryPreparation(PerCelStandardizing(NeutralExtractionMethod())),
        extraction_params=dict(winlen=0.03, winstep=0.01, nfilt=24, nfft=256),
        name='Georgiev')
    )
    csc.add_transformation_call('prepare_inputs')
    csc.add_transformation_call('inverse_normalize_inputs')
    csc.add_transformation_call('normalize_fit')
    csc.add_transformation_call('normalize_inputs')

    mtmf = MultiTaskModelFactory()
    mtmf.add_modelclass(MultiTaskHardSharing)
    mtmf.add_static_model_parameters(MultiTaskHardSharing.__name__, **{"hidden_size": 512, "n_hidden": 3})
    keys = list(csc.get_keys())
    # comb_iterator = itertools.chain(*map(lambda x: itertools.combinations(keys, x), range(1, len(keys) + 1)))
    comb_iterator = itertools.chain(*map(lambda x: itertools.combinations(keys, x), range(1, 0, -1)))

    for combo in comb_iterator:
        key_list = list(combo)
        for train, test, fold in csc.generate_training_splits(key_list):
            model = mtmf.create_model(MultiTaskHardSharing.__name__,
                                      input_size=train.datasets[0].get_input(0).flatten().shape[0],
                                      task_list=train.get_task_list())
            results = Training.create_results(modelname=model.name, task_list=train.get_task_list(), fold=fold,
                                              results_path=os.path.join(drive, 'Thesis_Results', 'Georgiev'),
                                              num_epochs=200)
            Training.run_gradient_descent(model=model, concat_dataset=train, test_dataset=test, results=results,
                                          batch_size=64, num_epochs=200 if key_list[0] != 0 else 500,
                                          optimizer=optim.SGD(model.parameters(), lr=0.001),
                                          train_loader=DataLoader(train, shuffle=True)
                                          )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
