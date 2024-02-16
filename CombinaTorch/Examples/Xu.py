import itertools
import os
import sys

from CombinaTorch.DataReaders.ASVspoof2015 import ASVspoof2015
from CombinaTorch.DataReaders.DCASE2017_SE import DCASE2017_SE
from CombinaTorch.DataReaders.DCASE2017_SS import DCASE2017_SS
from CombinaTorch.DataReaders.ExtractionMethod import PerCelStandardizing, LogbankSummaryExtraction, \
    NeutralExtractionMethod, MelSpectrogramExtraction, PerDimensionStandardizing, FramePreparation, \
    MinWindowSizePreparationFitter, WindowPreparation
from CombinaTorch.DataReaders.Ravdess import Ravdess
from CombinaTorch.MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from CombinaTorch.MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from CombinaTorch.MultiTask.MultiTaskModelFactory import MultiTaskModelFactory
from CombinaTorch.Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from CombinaTorch.Training.Training import Training

drive = r"E:/"


def main(argv):
    csc = ConcatTrainingSetCreator(nr_runs=4,
                                   # multiply=False,
                                   # recalculate=True
                                   )
    csc.add_data_reader(DCASE2017_SS(object_path=drive + r'Thesis_Results\Data_Readers\DCASE2017_SS_{}',
                                     data_path=drive + r'Thesis_Datasets\DCASE2017'))
    csc.add_data_reader(DCASE2017_SE(object_path=drive + r'Thesis_Results\Data_Readers\DCASE2017_SE_{}',
                                     data_path=drive + r'Thesis_Datasets\DCASE2017'))

    csc.add_extraction_method(extraction_method=
        MelSpectrogramExtraction(
            PerDimensionStandardizing(
                FramePreparation(
                    MinWindowSizePreparationFitter(
                        NeutralExtractionMethod(

                        )
                    )
                )
            ),
            name='MelSpectrogramXu',
            extraction_params=
            dict(
                n_fft=2048,
                hop_length=512
            )
        )
    )
    csc.add_signal_preprocessing(dict(mono=True))
    csc.add_transformation_call('prepare_fit')
    csc.add_transformation_call('prepare_inputs')
    csc.add_transformation_call('normalize_fit')
    csc.add_transformation_call('normalize_inputs')

    generator = csc.generate_training_splits()
    train, test, _ = next(generator)

    model = MultiTaskHardSharingConvolutional(input_channels=1,
                                              task_list=train.get_task_list(),
                                              hidden_size=64,
                                              n_hidden=4)
    results = Training.create_results(modelname=model.name,
                                      task_list=train.get_task_list(),
                                      results_path=os.path.join(drive, 'Thesis_Results', 'Xu'),
                                      num_epochs=200)
    Training.run_gradient_descent(model=model,
                                  concat_dataset=train,
                                  test_dataset=test,
                                  results=results,
                                  learning_rate=0.001,
                                  batch_size=32,
                                  num_epochs=200)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
