import math
import os

import joblib

from CombinaTorch.DataReaders.DataReader import DataReader
from CombinaTorch.Tasks.Task import MultiClassTask
from CombinaTorch.Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class Ravdess(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\Ravdess"
    # object_path = r"E:\Thesis_Results\Data_Readers\Ravdess"
    data_path = r"F:\Thesis_Datasets\Ravdess"

    def __init__(self, mode=0, **kwargs):
        print('start ravdess')
        super().__init__(**kwargs)
        print('Done loading Ravdess')
        self.mode = mode

    def get_path(self):
        return os.path.join(self.get_base_path()['base_path'], 'Ravdess.obj')

    def get_base_path(self):
        return dict(base_path=self.object_path)

    def get_task_name(self) -> str:
        return 'Ravdess_stress' if self.mode == 1 else 'Ravdess'

    def check_files(self, taskDataset, **kwargs):
        return super().check_files(taskDataset) and \
               os.path.isfile(self.get_path())

    def load_files(self):
        if os.path.isfile(self.get_path()):
            info = joblib.load(self.get_path())
            self.files = info['files']
            return
        song_folder = 'Audio_Song_actors_01-24'
        speech_folder = 'Audio_Speech_Actors_01-24'
        self.files = []
        for fold in [song_folder, speech_folder]:
            for _, songs, _ in os.walk(os.path.join(self.get_data_path(), fold)):
                for ss_dir in songs:
                    for file in os.listdir(os.path.join(self.get_data_path(), song_folder, ss_dir)):
                        mod, voc, em, emi, stat, rep, act = file[:-4].split('-')
                        self.files.append(
                            {'modality': mod,  # (01 = full-AV, 02 = video-only, 03 = audio-only)
                             'vocal_channel': voc,  # (01 = speech, 02 = song)
                             'emotion': em,
                             # (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
                             'emotional_intensity': emi,  # (01 = normal, 02 = strong)
                             'statement': stat,
                             # 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
                             'repetition': rep,  # 1st or 2nd rep
                             'actor': act,  # Odd numbered are male, even female
                             'file': os.path.join(self.get_data_path(), song_folder, ss_dir, file)
                             }
                        )

    def read_files(self, taskDataset, **kwargs):
        info = joblib.load(self.get_path())
        self.files = info['files']
        return super().read_files(taskDataset)

    def write_files(self, taskDataset, **kwargs):
        dict = {'files': self.files}
        joblib.dump(dict, self.get_path())
        super().write_files(taskDataset=taskDataset)

    def calculate_input(self, taskDataset: HoldTaskDataset, preprocess_parameters: dict, **kwargs):
        mode = self.mode
        perc = 0
        for file_idx in range(len(self.files)):
            file = self.files[file_idx]
            if mode == 0 or mode == 2:
                read_wav = self.preprocess_signal(self.load_wav(file['file']), **preprocess_parameters)
                taskDataset.extract_and_add_input(read_wav)
                if perc < (file_idx / len(self.files)) * 100:
                    print("Percentage done: {}".format(perc))
                    perc += 1
            elif mode == 1 and (int(file['emotion']) == 6 or int(file['emotion']) == 1):
                sig, samplerate = self.load_wav(file['file'])
                sig = sig[0:math.floor(1.28 * samplerate)]
                read_wav = self.preprocess_signal((sig, samplerate), **preprocess_parameters)
                taskDataset.extract_and_add_input(read_wav)
                if perc < (file_idx / len(self.files)) * 100:
                    print("Percentage done: {}".format(perc))
                    perc += 1

    def calculate_taskDataset(self,
                              taskDataset: HoldTaskDataset,
                              **kwargs):
        print('Calculating input')
        if self.mode == 1:
            targets = [f['emotion'] for f in self.files if int(f['emotion']) == 6 or int(f['emotion']) == 1]
            taskname = self.get_task_name()
            grouping = [f['actor'] for f in self.files if int(f['emotion']) == 6 or int(f['emotion']) == 1]
        else:
            targets = [f['emotion'] for f in self.files]
            taskname = self.get_task_name()
            grouping = [f['actor'] for f in self.files]
        distinct_targets = list(set(targets))
        targets = [[int(b == f) for b in distinct_targets] for f in targets]
        taskDataset.add_task_and_targets(
            targets=targets,
            task=MultiClassTask(name=taskname, output_labels=distinct_targets))
        taskDataset.add_grouping(grouping=grouping)

        if self.mode == 2:
            targets = ['male' if int(f['actor']) % 2 == 1 else 'female' for f in self.files]
            taskname = 'gender_detection'
            distinct_targets = list(set(targets))
            distinct_targets.sort()
            targets = [[int(b == f) for b in distinct_targets] for f in targets]
            taskDataset.add_task_and_targets(
                targets=targets,
                task=MultiClassTask(name=taskname, output_labels=distinct_targets))
