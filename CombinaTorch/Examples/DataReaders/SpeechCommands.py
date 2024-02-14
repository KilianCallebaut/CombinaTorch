import os

import tensorflow_datasets as tfds

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class SpeechCommands(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\SpeechCommands_{}"
    # object_path = r"E:\Thesis_Results\Data_Readers\SpeechCommands_{}"
    data_path = r"F:\Thesis_Datasets\SpeechCommands"

    def __init__(self, **kwargs):
        self.sample_rate = 16000
        print('start Speech commands')
        super().__init__(**kwargs)
        print('Done loading Speech Commands')

    def get_base_path(self):
        return dict(base_path=self.object_path.format('train'),
                    testing_base_path=self.object_path.format('eval'))

    def get_task_name(self) -> str:
        return "SpeechCommands"

    def load_files(self):
        self.ds, self.ds_info = tfds.load('speech_commands', split=['train', 'test'], shuffle_files=False,
                                          data_dir=self.get_data_path(), with_info=True)
        print('Done loading Speech Commands dataset')

    def calculate_input(self, taskDataset: HoldTaskDataset, preprocess_parameters: dict, **kwargs):
        for audio_label in self.ds[0].as_numpy_iterator():
            audio = audio_label["audio"]
            fs = self.sample_rate
            sig_samplerate = self.preprocess_signal((audio, fs), **preprocess_parameters)
            taskDataset.extract_and_add_input(sig_samplerate)

        for audio_label in self.ds[1].as_numpy_iterator():
            audio = audio_label["audio"]
            fs = self.sample_rate
            sig_samplerate = self.preprocess_signal((audio, fs), **preprocess_parameters)
            taskDataset.test_set.extract_and_add_input(sig_samplerate)

    def calculate_taskDataset(self,
                              taskDataset: HoldTaskDataset,
                              **kwargs):
        # Training Set
        print("Calculate Training Set")
        targets = []
        for audio_label in self.ds[0].as_numpy_iterator():
            targets.append(audio_label["label"])

        targets = [[int(b == f) for b in range(len(self.ds_info.features['label'].names))] for f in targets]

        print("Calculate Test Set")
        # Test Set
        targets_t = []
        for audio_label in self.ds[1].as_numpy_iterator():
            targets_t.append(audio_label["label"])
        targets_t = [[int(b == f) for b in range(len(self.ds_info.features['label'].names))] for f in targets_t]

        task = MultiClassTask(name=self.get_task_name(),
                              output_labels=self.ds_info.features['label'].names)
        taskDataset.add_task_and_targets(
            task=task,
            targets=targets)
        taskDataset.add_task_and_targets(
            task=task,
            targets=targets_t
        )
