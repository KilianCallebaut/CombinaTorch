import os
from datetime import timedelta
from timeit import default_timer as timer

import joblib
import pandas as pd

from CombinaTorch.DataReaders.DataReader import DataReader
from CombinaTorch.Tasks.Task import MultiLabelTask, MultiClassTask
from CombinaTorch.Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class ParkAudiosetDataset(DataReader):
    data_path = r'E:\Thesis_Datasets\audioset_park\audioset_1sec'
    object_path = r"E:\Thesis_Results\Data_Readers\ParkAudiosetDataset"
    # object_path = r"E:\Thesis_Results\Data_Readers\ChenAudiosetDataset"

    labels = []
    wav_files = []

    leftovers = []
    target_list = []

    def __init__(self, mode=0, **kwargs):
        print('start park')
        super().__init__(**kwargs)
        print('done park')
        self.mode = mode

    def get_base_path(self):
        return dict(base_path=self.object_path)

    def get_task_name(self) -> str:
        if self.mode == 1:
            return 'park_audioset'
        elif self.mode == 2:
            return 'park_audioset_multiclass'
        return "audioset"

    def write_files(self, taskDataset: HoldTaskDataset, **kwargs):
        super().write_files(taskDataset, **kwargs)
        files = dict(labels=self.labels, wav_files=self.wav_files)
        joblib.dump(files, os.path.join(self.object_path, 'files.obj'))

    def load_files(self):
        if os.path.isfile(os.path.join(self.object_path, 'files.obj')):
            files = joblib.load(os.path.join(self.object_path, 'files.obj'))
            self.labels = files['labels']
            self.wav_files = files['wav_files']
            return
        labels = []
        wav_files = []

        for _, dirs, _ in os.walk(self.get_data_path()):
            cdt = len(dirs)
            cd = 0
            cn = 0
            start = timer()

            for dir in dirs:
                perc = (cd / cdt) * 100

                cd += 1
                lab_dir = []
                wav_dir = []
                for file in os.listdir(os.path.join(self.get_data_path(), dir)):
                    filepath = os.path.join(self.get_data_path(), dir, file)
                    if file.endswith('.txt'):
                        name = file.split('_labels.txt')[0]
                        wav_loc = os.path.join(self.get_data_path(), dir, name + '.wav')
                        lab = pd.read_csv(filepath, names=['id', 'str', 'full'])['full'].values
                        wav_dir.append(wav_loc)
                        lab_dir.append(lab)

                labels.append(lab_dir)
                wav_files.append(wav_dir)

                if perc > cn * 10:
                    print((cd / cdt) * 100)
                    end = timer()
                    timedel = end - start
                    print('estimated time: {}'.format(timedelta(seconds=timedel * (10 - cn))))
                    start = end
                    cn += 1
        self.labels = labels
        self.wav_files = wav_files

    def calculate_taskDataset(self, taskDataset: HoldTaskDataset, **kwargs):
        targets, distinct_targets = self.calculate_targets()

        name = self.get_task_name()
        taskDataset.add_task_and_targets(targets=targets,
                                         task=MultiClassTask(name=name,
                                                             output_labels=distinct_targets) if self.mode == 2 else
                                         MultiLabelTask(name=name,
                                                        output_labels=distinct_targets))
        taskDataset.add_grouping(grouping=[fold for fold in range(len(self.wav_files)) for fi in
                                           self.wav_files[fold] if fi not in self.skip_files])
        print('Task Calculated')

    def calculate_input(self,
                        taskDataset: HoldTaskDataset,
                        preprocess_parameters: dict,
                        **kwargs):
        i = 0
        start = timer()
        for fold_i, folder in enumerate(self.wav_files):
            for file_i, file in enumerate(folder):
                try:
                    if self.mode == 2 and len(self.labels[fold_i][file_i]) > 1:
                        raise RuntimeError
                    read_wav = self.preprocess_signal(self.load_wav(file), **preprocess_parameters)
                    taskDataset.extract_and_add_input(read_wav)
                except RuntimeError:
                    self.skip_files.append(file)
            i += 1

            end = timer()
            timedel = (end - start) / i
            print('Percentage done: {} estimated time: {}'.format(i / len(self.wav_files),
                                                                  timedelta(seconds=timedel * (len(self.wav_files) - i))
                                                                  ), end='\r')
        print('Input calculated')

    def calculate_targets(self):
        targets = [self.labels[f_id][x_id] for f_id in range(len(self.labels))
                   for x_id in range(len(self.labels[f_id])) if
                   not self.wav_files[f_id][x_id] in self.skip_files]
        if self.mode == 1:
            targets = [self.group_events(e) for e in targets]
        print(set(self.leftovers))
        # distinct_targets = list(set([x for l in targets for x in l if x != 'None of the above']))
        distinct_targets = list(set([x for l in targets for x in l]))

        # Targets are translated as binary strings with 1 for each target
        # at the index where it is in distinct_targets order
        targets = [[int(b in f) for b in distinct_targets] for f in targets]
        return targets, distinct_targets

    def group_events(self, list_of_events):
        grouped_list = []
        for e in list_of_events:
            if e in ['Crowd', 'Chatter', 'Hubbub, speech noise, speech babble']:
                grouped_list.append('crowd')
            elif e in ['Applause', 'Clapping']:
                grouped_list.append('applause')
            elif e in ['Laughter']:
                grouped_list.append('laughter')
            elif e in ['Typing', 'Clicking']:
                grouped_list.append('typing/clicking')
            elif e in ['Door', 'Knock']:
                grouped_list.append('door')
            elif e in ['Silence']:
                grouped_list.append('silence')
            elif e in ['Television']:
                grouped_list.append('television')
            elif e in ['Walk, footsteps']:
                grouped_list.append('walk')
            elif e in ['Speech', 'Female speech, woman speaking', 'Male speech, man speaking', 'Conversation']:
                grouped_list.append('speech')
            else:
                self.leftovers.append(e)
                grouped_list.append('others')

        grouped_list = list(set(grouped_list))
        if 'silence' in grouped_list and len(grouped_list) > 1:
            grouped_list.remove('silence')
        if 'others' in grouped_list and len(grouped_list) > 1:
            grouped_list.remove('others')
        return grouped_list
