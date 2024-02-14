import os

import joblib
from dcase_util.datasets import TUTAcousticScenes_2017_DevelopmentSet, TUTAcousticScenes_2017_EvaluationSet
from numpy import long

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class DCASE2017_SS(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SS_{}"
    # object_path = r"E:\Thesis_Results\Data_Readers\DCASE2017_SS_{}"
    data_path = 'F:\\Thesis_Datasets\\DCASE2017\\'

    def __init__(self, **kwargs):
        print('start DCASE2017 SS')
        super().__init__(**kwargs)
        print('done DCASE2017 SS')

    def get_base_path(self):
        return dict(base_path=self.object_path.format('train'),
                    testing_base_path=self.object_path.format('eval'))

    def get_path(self):
        return os.path.join(self.get_base_path()['base_path'],
                            'DCASE2017_SS.obj')

    def get_eval_path(self):
        return os.path.join(self.get_base_path()['testing_base_path'],
                            'DCASE2017_SS.obj')

    def get_task_name(self) -> str:
        return 'DCASE2017_SS'

    def check_files(self, taskDataset, **kwargs):
        return super().check_files(taskDataset) and \
               os.path.isfile(self.get_path()) and \
               os.path.isfile(self.get_eval_path())

    def load_files(self):
        # MetaDataContainer(filename=)
        self.devdataset = TUTAcousticScenes_2017_DevelopmentSet(
            data_path=self.get_data_path(),
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name='TUTAcousticScenes_2017_DevelopmentSet',
            fold_list=[1, 2, 3, 4],
            # fold_list=[1],
            evaluation_mode='folds',
            storage_name='TUT-acoustic-scenes-2017-development'

        ).initialize()
        self.evaldataset = TUTAcousticScenes_2017_EvaluationSet(
            data_path=self.get_data_path(),
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name=r'TUTAcousticScenes_2017_EvaluationSet',
            fold_list=[1, 2, 3, 4],
            evaluation_mode='folds',
            storage_name='TUT-acoustic-scenes-2017-evaluation'
        ).initialize()

        self.audio_files = self.devdataset.audio_files
        self.audio_files_eval = self.evaldataset.audio_files

    def read_files(self, taskDataset: HoldTaskDataset, **kwargs):
        self.audio_files = joblib.load(self.get_path())
        self.audio_files_eval = joblib.load(self.get_eval_path())
        return super().read_files(taskDataset)

    def write_files(self, taskDataset: HoldTaskDataset, **kwargs):
        super().write_files(taskDataset=taskDataset)
        dict = {'audio_files': self.audio_files}
        joblib.dump(dict, self.get_path())

        dict = {'audio_files_eval': self.audio_files_eval}
        joblib.dump(dict, self.get_eval_path())

    def calculate_input(self, taskDataset: HoldTaskDataset, preprocess_parameters: dict, **kwargs):
        perc = 0
        for audio_idx in range(len(self.audio_files)):
            read_wav = self.preprocess_signal(self.load_wav(self.audio_files[audio_idx]), **preprocess_parameters)
            taskDataset.extract_and_add_input(read_wav)
            if perc < (audio_idx / len(self.audio_files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1

        perc = 0
        for audio_idx in range(len(self.audio_files_eval)):
            read_wav = self.preprocess_signal(self.load_wav(self.audio_files_eval[audio_idx]), **preprocess_parameters)
            taskDataset.test_set.extract_and_add_input(read_wav)
            if perc < (audio_idx / len(self.audio_files_eval)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        print("Calculating input done")

    def calculate_taskDataset(self,
                              taskDataset: HoldTaskDataset,
                              **kwargs):
        distinct_labels = self.devdataset.scene_labels()
        targets = []

        for file_id in range(len(self.audio_files)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            annotations = self.devdataset.meta.filter(self.audio_files[file_id])[0]

            target = [int(distinct_labels[label_id] == annotations.scene_label) for label_id in
                      range(len(distinct_labels))]
            targets.append(target)

            print(file_id / len(self.audio_files))

        targets_val = []
        for file_id in range(len(self.audio_files_eval)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            annotations = self.evaldataset.meta.filter(self.audio_files_eval[file_id])[0]

            target = [int(distinct_labels[label_id] == annotations.scene_label) for label_id in
                      range(len(distinct_labels))]
            targets_val.append(target)
            print(file_id / len(self.audio_files_eval))

        task = MultiClassTask(name=self.get_task_name(), output_labels=distinct_labels)
        taskDataset.add_task_and_targets(
            task=task,
            targets=targets
        )
        taskDataset.test_set.add_task_and_targets(
            task=task,
            targets=targets_val
        )
