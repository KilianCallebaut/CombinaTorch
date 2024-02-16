import itertools
import os
import sys

from CombinaTorch.DataReaders.ASVspoof2015 import ASVspoof2015
from CombinaTorch.DataReaders.DCASE2017_SE import DCASE2017_SE
from CombinaTorch.DataReaders.DCASE2017_SS import DCASE2017_SS
from CombinaTorch.DataReaders.ExtractionMethod import PerCelStandardizing, LogbankSummaryExtraction, \
    NeutralExtractionMethod, MelSpectrogramExtraction, PerDimensionStandardizing, FramePreparation, \
    MinWindowSizePreparationFitter
from CombinaTorch.DataReaders.Ravdess import Ravdess
from CombinaTorch.MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from CombinaTorch.MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from CombinaTorch.MultiTask.MultiTaskModelFactory import MultiTaskModelFactory
from CombinaTorch.Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from CombinaTorch.Training.Training import Training

drive = r"E:/"

def read_data():
    extracted = dict(dcase_train=[], dcase_eval=[], ravdess_em=[], ravdess_s=[], asv=[])
    targets = dict(dcase_train=[], dcase_eval=[], ravdess_em=[], ravdess_s=[], asv=[])
    grouping = dict(dcase_train=[], dcase_eval=[], ravdess_em=[], ravdess_s=[], asv=[])
    data_path = 'E:\\Thesis_Datasets\\DCASE2017\\'
    # MetaDataContainer(filename=)
    devdataset = TUTAcousticScenes_2017_DevelopmentSet(
        data_path=data_path,
        log_system_progress=False,
        show_progress_in_console=True,
        use_ascii_progress_bar=True,
        name='TUTAcousticScenes_2017_DevelopmentSet',
        fold_list=[1, 2, 3, 4],
        # fold_list=[1],
        evaluation_mode='folds',
        storage_name='TUT-acoustic-scenes-2017-development'

    ).initialize()
    evaldataset = TUTAcousticScenes_2017_EvaluationSet(
        data_path=data_path,
        log_system_progress=False,
        show_progress_in_console=True,
        use_ascii_progress_bar=True,
        name=r'TUTAcousticScenes_2017_EvaluationSet',
        fold_list=[1, 2, 3, 4],
        evaluation_mode='folds',
        storage_name='TUT-acoustic-scenes-2017-evaluation'
    ).initialize()
    distinct_labels = devdataset.scene_labels()
    perc = 0
    for audio_idx in range(len(devdataset.audio_files)):
        read = soundfile.read(devdataset.audio_files[audio_idx])
        read = (np.mean(read[0], axis=1), read[1])
        read = librosa.core.resample(*read, 16000)
        extracted['dcase_train'].append(read)
        annotations = devdataset.meta.filter(devdataset.audio_files[audio_idx])[0]
        targets['dcase_train'].append([int(distinct_labels[label_id] == annotations.scene_label) for label_id in
                                       range(len(distinct_labels))])
        if perc < (audio_idx / len(devdataset.audio_files)) * 100:
            print("Percentage done: {}".format(perc))
            perc += 1
    perc = 0
    for audio_idx in range(len(evaldataset.audio_files)):
        read = soundfile.read(evaldataset.audio_files[audio_idx])
        read = (np.mean(read[0], axis=1), read[1])
        read = librosa.core.resample(*read, 16000)
        extracted['dcase_eval'].append(read)
        annotations = evaldataset.meta.filter(evaldataset.audio_files[audio_idx])[0]
        targets['dcase_eval'].append([int(distinct_labels[label_id] == annotations.scene_label) for label_id in
                                      range(len(distinct_labels))])
        if perc < (audio_idx / len(evaldataset.audio_files)) * 100:
            print("Percentage done: {}".format(perc))
            perc += 1

def main(argv):
    # csc = ConcatTrainingSetCreator(nr_runs=4)
    # csc.add_data_reader(DCASE2017_SS(object_path=drive + r'Thesis_Results\Data_Readers\DCASE2017_SS_{}',
    #                                  data_path=drive + r'Thesis_Datasets\DCASE2017'))
    # csc.add_data_reader(DCASE2017_SE(object_path=drive + r'Thesis_Results\Data_Readers\DCASE2017_SE_{}',
    #                                  data_path=drive + r'Thesis_Datasets\DCASE2017'))
    #
    # csc.add_extraction_method(extraction_method=
    #     MelSpectrogramExtraction(
    #         PerDimensionScaling(
    #             FramePreparation(
    #                 MinWindowSizePreparationFitter(
    #                     NeutralExtractionMethod(
    #
    #                     )
    #                 )
    #             )
    #         ),
    #         name='MelSpectrogramXu',
    #         extraction_params=
    #         dict(
    #             n_fft=2048,
    #             hop_length=512
    #         )
    #     )
    # )
    # csc.add_transformation_call('prepare_fit')
    # csc.add_transformation_call('prepare_inputs')
    # csc.add_transformation_call('normalize_fit')
    # csc.add_transformation_call('normalize_inputs')
    #
    # mtmf = MultiTaskModelFactory()
    # mtmf.add_modelclass(MultiTaskHardSharingConvolutional)
    # mtmf.add_static_model_parameters(MultiTaskHardSharingConvolutional.__name__, **{"hidden_size": 512, "n_hidden": 4})
    # generator = csc.generate_training_splits()
    # train, test, _ = next(generator)
    # model = mtmf.create_model(MultiTaskHardSharingConvolutional.__name__,
    #                           input_size=train.datasets[0].get_input(0).flatten().shape[0],
    #                           task_list=train.get_task_list())
    # results = Training.create_results(modelname=model.name,
    #                                   task_list=train.get_task_list(),
    #                                   results_path=os.path.join(drive, 'Thesis_Results'),
    #                                   num_epochs=200)
    # Training.run_gradient_descent(model=model,
    #                               concat_dataset=train,
    #                               test_dataset=test,
    #                               results=results,
    #                               learning_rate=0.001,
    #                               batch_size=32,
    #                               num_epochs=200)
    if __name__ == "__main__":
        try:
            sys.exit(main(sys.argv))
        except (ValueError, IOError) as e:
            sys.exit(e)
