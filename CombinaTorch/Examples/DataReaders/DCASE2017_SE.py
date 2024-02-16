import math
import os

import dcase_util
import joblib
from dcase_util.datasets import TUTSoundEvents_2017_DevelopmentSet, TUTSoundEvents_2017_EvaluationSet

from CombinaTorch.DataReaders.DataReader import DataReader
from CombinaTorch.Tasks.Task import MultiLabelTask
from CombinaTorch.Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class DCASE2017_SE(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SE_{}"
    data_path = 'F:\\Thesis_Datasets\\DCASE2017\\'

    def __init__(self, **kwargs):
        print('start DCASE2017 SE')
        super().__init__(**kwargs)
        print('done DCASE2017 SE')

    def get_path(self):
        return os.path.join(self.get_base_path()['base_path'], 'DCASE2017_SE.obj')

    def get_base_path(self):
        return dict(base_path=self.object_path.format('train'),
                    testing_base_path=self.object_path.format('eval'))

    def get_task_name(self) -> str:
        return 'DCASE2017_SE'

    def check_files(self, taskDataset, **kwargs):
        return super().check_files(taskDataset) and \
               os.path.isfile(self.get_path())

    def read_files(self, taskDataset: HoldTaskDataset, **kwargs):
        info = joblib.load(self.get_path())
        self.audio_files = info['audio_files']
        super().read_files(taskDataset)

    def write_files(self, taskDataset, **kwargs):
        super().write_files(taskDataset)
        dict = {'audio_files': self.audio_files}
        joblib.dump(dict, self.get_path())

    def load_files(self):
        self.devdataset = TUTSoundEvents_2017_DevelopmentSet(
            data_path=self.get_data_path(),
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name='TUTSoundEvents_2017_DevelopmentSet',
            fold_list=[1, 2, 3, 4],
            evaluation_mode='folds',
            storage_name='TUT-sound-events-2017-development'

        ).initialize()
        self.evaldataset = TUTSoundEvents_2017_EvaluationSet(
            data_path=self.get_data_path(),
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name='TUTSoundEvents_2017_EvaluationSet',
            fold_list=[1, 2, 3, 4],
            evaluation_mode='folds',
            storage_name='TUT-sound-events-2017-evaluation'

        ).initialize()
        # if os.path.isfile(self.get_path()):
        #     info = joblib.load(self.get_path())
        #     self.audio_files = info['audio_files']
        #     return

        self.audio_files = self.devdataset.audio_files
        self.audio_files_eval = self.evaldataset.audio_files

    def calculate_input(self, taskDataset: HoldTaskDataset, preprocess_parameters: dict, **kwargs):
        perc = 0
        for audio_idx in range(len(self.audio_files)):
            audio_cont = dcase_util.containers.AudioContainer().load(
                filename=self.audio_files[audio_idx]
            )
            segmented_signals, meta = audio_cont.segments(segment_length_seconds=1.0)
            for ss in segmented_signals:
                read_wav = self.preprocess_signal((ss, audio_cont.fs),
                                                  **preprocess_parameters)
                taskDataset.extract_and_add_input(read_wav)
            while perc < (audio_idx / len(self.audio_files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        perc = 0
        print("Calculating input done")
        for audio_idx in range(len(self.audio_files_eval)):
            audio_cont = dcase_util.containers.AudioContainer().load(
                filename=self.audio_files_eval[audio_idx]
            )
            segmented_signals, meta = audio_cont.segments(segment_length_seconds=1.0)
            for ss in segmented_signals:
                read_wav = self.preprocess_signal((ss, audio_cont.fs),
                                                  **preprocess_parameters)
                taskDataset.test_set.extract_and_add_input(read_wav)
            while perc < (audio_idx / len(self.audio_files_eval)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        print("Calculating input done")

    def calculate_taskDataset(self,
                              taskDataset: HoldTaskDataset,
                              **kwargs):
        distinct_labels = self.devdataset.event_labels()
        task = MultiLabelTask(name=self.get_task_name(), output_labels=distinct_labels)
        targets = []

        for file_id in range(len(self.audio_files)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            audio_cont = dcase_util.containers.AudioContainer().load(filename=self.audio_files[file_id])
            annotations = self.devdataset.meta.filter(self.audio_files[file_id])
            target = annotations.to_event_roll(label_list=self.devdataset.event_labels(),
                                               time_resolution=1,
                                               length_seconds=math.floor(audio_cont.duration_sec)).data
            for t in target.T:
                targets.append(t)
            print(file_id / len(self.audio_files))

        taskDataset.add_task_and_targets(
            targets=targets,
            task=task
        )
        targets_val = []

        for file_id in range(len(self.audio_files_eval)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            audio_cont = dcase_util.containers.AudioContainer().load(filename=self.audio_files_eval[file_id])
            annotations = self.evaldataset.meta.filter(self.audio_files_eval[file_id])
            target = annotations.to_event_roll(label_list=self.devdataset.event_labels(),
                                               time_resolution=1,
                                               length_seconds=math.floor(audio_cont.duration_sec)).data
            for t in target.T:
                targets_val.append(t)
            print(file_id / len(self.audio_files_eval))

        taskDataset.test_set.add_task_and_targets(
            targets=targets_val,
            task=task
        )
