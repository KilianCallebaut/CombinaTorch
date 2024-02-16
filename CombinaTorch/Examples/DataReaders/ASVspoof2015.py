import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from numpy import long

from CombinaTorch.DataReaders.DataReader import DataReader
from CombinaTorch.Tasks.Task import MultiClassTask
from CombinaTorch.Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class ASVspoof2015(DataReader):
    # Wav_folder = r"F:\Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015\DS_10283_853\wav"
    data_path = r"F:\Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015\DS_10283_853"
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\ASVspoof2015_{}"

    def __init__(self, **kwargs):
        print('start asvspoof2015')
        super().__init__(**kwargs)
        print('done asvspoof2015')

    def get_path(self):
        return os.path.join(self.get_base_path()['base_path'], 'ASVspoof2015.obj')

    def get_base_path(self):
        return dict(base_path=self.object_path.format('train'))

    def get_task_name(self) -> str:
        return 'ASVspoof2015'

    def check_files(self, taskDataset, **kwargs):
        return super().check_files(taskDataset) and \
               os.path.isfile(self.get_path())

    def load_files(self):
        # self.truths = pd.read_csv(os.path.join(self.get_data_path(), 'CM_protocol', 'cm_train.trn'), sep=' ', header=None,
        #                           names=['folder', 'file', 'method', 'source'])
        # self.truths = self.truths.append(add)
        self.truths = pd.read_csv(os.path.join(self.get_data_path(), 'Joint_ASV_CM_protocol', 'ASV_male_development.ndx'),
                                  sep=' ',
                                  header=None,
                                  names=['folder', 'file', 'method', 'source'])
        self.truths['gender'] = 'male'
        truths_female = pd.read_csv(
            os.path.join(self.get_data_path(), 'Joint_ASV_CM_protocol', 'ASV_female_development.ndx'), sep=' ',
            header=None,
            names=['folder', 'file', 'method', 'source'])
        truths_female['gender'] = 'female'
        male_eval = pd.read_csv(os.path.join(self.get_data_path(), 'Joint_ASV_CM_protocol', 'ASV_male_evaluation.ndx'),
                                sep=' ', header=None,
                                names=['folder', 'file', 'method', 'source'])
        male_eval['gender'] = 'male'
        female_eval = pd.read_csv(os.path.join(self.get_data_path(), 'Joint_ASV_CM_protocol', 'ASV_female_evaluation.ndx'),
                                  sep=' ',
                                  header=None,
                                  names=['folder', 'file', 'method', 'source'])
        female_eval['gender'] = 'female'
        male_enrol = pd.read_csv(os.path.join(self.get_data_path(), 'Joint_ASV_CM_protocol', 'ASV_male_enrolment.ndx'),
                                 sep=' ', header=None,
                                 names=['folder', 'file', 'method', 'source'])
        male_enrol['gender'] = 'male'
        female_enrol = pd.read_csv(os.path.join(self.get_data_path(), 'Joint_ASV_CM_protocol', 'ASV_female_enrolment.ndx'),
                                   sep=' ',
                                   header=None,
                                   names=['folder', 'file', 'method', 'source'])
        truths_female['gender'] = 'female'

        # self.truths.append(truths_male)
        self.truths.append(truths_female)
        self.truths.append(male_eval)
        self.truths.append(female_eval)
        self.truths.append(male_enrol)
        self.truths.append(female_enrol)

        # self.truths = self.truths[(self.truths.method == 'human')]
        self.truths.sort_values(['folder', 'file'], inplace=True)

        self.files = [os.path.join(self.get_data_path(), 'wav', x[0], x[1]) for x in self.truths.to_numpy()]

    def read_files(self, taskDataset: HoldTaskDataset, **kwargs) -> HoldTaskDataset:
        info = joblib.load(self.get_path())
        self.files = info['files']
        self.truths = info['truths']
        return super().read_files(taskDataset)

    def write_files(self, taskDataset: HoldTaskDataset, **kwargs):
        super().write_files(taskDataset)
        dict = {'files': self.files,
                'truths': self.truths,
                }
        joblib.dump(dict, self.get_path())

    def calculate_input(self,
                        taskDataset: HoldTaskDataset,
                        preprocess_parameters: dict,
                        **kwargs):
        print('training')
        perc = 0

        for audio_idx in range(len(self.files)):
            read_wav = self.preprocess_signal(self.load_wav(self.files[audio_idx] + '.wav'), **preprocess_parameters)
            taskDataset.extract_and_add_input(read_wav)
            if perc < (audio_idx / len(self.files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1

    def calculate_taskDataset(self, taskDataset: HoldTaskDataset, **kwargs):
        distinct_labels = self.truths.folder.unique()
        distinct_labels.sort()
        # distinct_labels = np.append(distinct_labels, 'unknown')

        targets = []
        genders = ['male', 'female']
        genders.sort()
        targets_gender = []
        for i in range(self.truths.shape[0]):
            target = [int(distinct_labels[label_id] == self.truths.loc[i].folder)
                      if (self.truths.loc[i].method == 'genuine' or self.truths.loc[i].method == 'human')
                      else int(label_id == len(distinct_labels) - 1)
                      for label_id in range(len(distinct_labels))]
            targets.append(target)
            target = [int(self.truths.loc[i].gender == label) for label_id, label in enumerate(distinct_labels)]
            targets_gender.append(target)

        taskDataset.add_task_and_targets(
            task=MultiClassTask(
                name=self.get_task_name(),
                output_labels=distinct_labels),
            targets=targets
        )
        taskDataset.add_task_and_targets(
            task=MultiClassTask(
                name='gender_detection',
                output_labels=genders
            ),
            targets=targets_gender
        )
