import os
from datetime import timedelta
from timeit import default_timer as timer

import joblib
import numpy as np
import torch

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiLabelTask
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class ChenAudiosetDataset(DataReader):
    data_path = r'E:\Thesis_Datasets\audioset_chen\audioset_filtered'
    train_dir = "balanced_train_segments"
    object_path = r"E:\Thesis_Results\Data_Readers\ChenAudiosetDataset"
    # object_path = r"E:\Thesis_Results\Data_Readers\ChenAudiosetDataset"

    # Bools for choosing random selection because overrepresented in dataset
    limit_speech = True
    limit_other = False

    np_objects = []
    files = []
    wav_files = []

    AllLabels = []
    AllFlatenedFielsLabels = []
    AllFlatenedLabels = []
    AllLabelSets = []
    AllFlatenedLabelSets = []
    CountPerLabel = []
    CountFolderPerLabel = []
    CountFolderPerLabelSet = []

    leftovers = []
    target_list = []

    def __init__(self, mode=0, **kwargs):
        print('start chen')
        super().__init__(**kwargs)
        print('done chen')
        self.mode = mode

    def get_path(self):
        return os.path.join(self.object_path, "ChenAudiosetDataset.obj")

    def get_base_path(self):
        return dict(base_path=self.object_path)

    def get_task_name(self) -> str:
        if self.mode == 1:
            return 'park_audioset'
        return "chen_audioset"

    def load_files(self):
        if os.path.isfile(self.get_path()):
            obj = joblib.load(self.get_path())
            self.files = obj['files']
            self.np_objects = obj['np_objects']
            self.wav_files = obj['wav_files']
            return
        files = []
        np_objects = []
        wav_files = []
        print(os.path.join(self.get_data_path(), self.train_dir))

        for _, dirs, _ in os.walk(os.path.join(self.get_data_path(), self.train_dir)):
            cdt = len(dirs)
            cd = 0
            cn = 0
            start = timer()

            for dir in dirs:
                perc = (cd / cdt) * 100

                cd += 1
                filedir = []
                np_dir = []
                wav_dir = []
                for file in os.listdir(os.path.join(self.get_data_path(), self.train_dir, dir)):
                    filepath = os.path.join(self.get_data_path(), self.train_dir, dir, file)
                    if file.endswith('.npy'):
                        np_obj = np.load(filepath, allow_pickle=True)
                        filedir.append(np_obj.item())
                        if len(np_obj.item()['labels']) > 1:
                            b = 1
                        np_dir.append(np_obj)

                        wav_loc = np_obj.item()['wav_file'].split(r'/')[6]
                        wav_loc = os.path.join(self.get_data_path(), self.train_dir, dir, wav_loc)
                        wav_dir.append(wav_loc)

                files.append(filedir)
                np_objects.append(np_dir)
                wav_files.append(wav_dir)

                if perc > cn * 10:
                    print((cd / cdt) * 100)
                    end = timer()
                    timedel = end - start
                    print('estimated time: {}'.format(timedelta(seconds=timedel * (10 - cn))))
                    start = end
                    cn += 1
        self.files = files
        self.np_objects = np_objects
        self.wav_files = wav_files

    def calculate_taskDataset(self, taskDataset: HoldTaskDataset, **kwargs):
        targets, distinct_targets = self.calculate_targets()

        name = self.get_task_name()
        taskDataset.add_task_and_targets(targets=targets,
                                         task=MultiLabelTask(name=name,
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
        for folder in self.wav_files:
            for file in folder:
                if self.mode == 2:
                    taskDataset.add_input(torch.tensor(
                        self.files[self.wav_files.index(folder)][folder.index(file)]['embedding_normalized']))
                    continue
                try:
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
        targets = [[l[2] for l in x['labels']] for f in self.files for x in f if
                   os.path.join(self.get_data_path(), *x['wav_file'].split(r'/')[4:]) not in self.skip_files]
        if self.mode == 1 or self.mode == 2:
            targets = [self.group_events(e) for e in targets]
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
                grouped_list.append('Laughter')
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
                grouped_list.append('others')

        grouped_list = list(set(grouped_list))
        if 'silence' in grouped_list and len(grouped_list) > 1:
            grouped_list.remove('silence')
        if 'others' in grouped_list and len(grouped_list) > 1:
            grouped_list.remove('others')
        return grouped_list

    # def contains_label(self, folder, label):
    #     for f in folder:
    #         if f[self.taskDataset.task.output_labels.index(label)] == 1:
    #             return True
    #     return False
    #
    # def open_audio(url, filename):
    #     # Opening file
    #     # url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
    #     # filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"
    #
    #     r = requests.get(url)
    #
    #     with open(filename, 'wb') as f:
    #         f.write(r.content)
    #
    #     waveform, sample_rate = torchaudio.load(filename)
    #     return waveform, sample_rate
    #
    # # Returns list (labels in file) of strings (labels)
    # def getLabelsFile(self, folder_ind: int, audio_ind_in_folder: int) -> list:
    #     return [l[2] for l in self.files[folder_ind][audio_ind_in_folder]['labels'] if l[2] != 'None of the above']
    #
    # # Returns list (files in folder)  of list (labels in file) of strings (labels)
    # def getLabelsFolder(self, folder_ind: int) -> list:
    #     return [self.getLabelsFile(folder_ind, f) for f in range(0, len(self.files[folder_ind]))]
    #
    # # Returns list (folders in dataset) of list (files in folder)  of list (labels in file) of strings (labels)
    # def getAllLabels(self) -> list:
    #     if not self.AllLabels:
    #         self.AllLabels = [self.getLabelsFolder(f) for f in range(0, len(self.files))]
    #     return self.AllLabels
    #
    # # Returns list (for each file in db) of lists (for each label in in file) of labels
    # def getAllFlatenedFilesLabels(self) -> list:
    #     if not self.AllFlatenedFielsLabels:
    #         self.AllFlatenedFielsLabels = [files for folders in self.getAllLabels() for files in folders]
    #     return self.AllFlatenedFielsLabels
    #
    # # Returns list (each label of each file in database) of labels
    # def getAllFlatenedLabels(self) -> list:
    #     if not self.AllFlatenedLabels:
    #         self.AllFlatenedLabels = [label for files in self.getAllFlatenedFilesLabels() for label in files]
    #     return self.AllFlatenedLabels
    #
    # # Returns list of distinct labels in folder
    # def getLabelSet(self, folder_ind: int) -> list:
    #     return list(set([label for files in self.getLabelsFolder(folder_ind) for label in files]))
    #
    # # Returns list of labelsets
    # def getAllLabelSets(self) -> list:
    #     if not self.AllLabelSets:
    #         self.AllLabelSets = [self.getLabelSet(folder_ind) for folder_ind in range(0, len(self.files))]
    #     return self.AllLabelSets
    #
    # # Returns flatened list of labelsets
    # def getAllFlatenedLabelSets(self) -> list:
    #     if not self.AllFlatenedLabelSets:
    #         self.AllFlatenedLabelSets = [label for labelset in self.getAllLabelSets() for label in labelset]
    #     return self.AllFlatenedLabelSets
    #
    # # Returns dictionary of occurence counts per label in db
    # def getCountPerLabel(self) -> dict:
    #     if not self.CountPerLabel:
    #         self.CountPerLabel = dict(sorted(dict(Counter(self.getAllFlatenedLabels())).items(), key=lambda x: x[1]))
    #     return self.CountPerLabel
    #
    # # Returns dictionary with the count of occurence in folders per label
    # def getCountFolderPerLabel(self) -> dict:
    #     if not self.CountFolderPerLabel:
    #         self.CountFolderPerLabel = dict(
    #             sorted(dict(Counter(self.getAllFlatenedLabelSets())).items(), key=lambda x: x[1]))
    #     return self.CountFolderPerLabel
    #
    # # Returns dictionary of occurrence in folder counts per labelset
    # def getCountFolderPerLabelSet(self) -> dict:
    #     if not self.CountFolderPerLabelSet:
    #         self.CountFolderPerLabelSet = dict(
    #             sorted(dict(Counter(map(tuple, self.getAllLabelSets()))).items(), key=lambda x: x[1]))
    #     return self.CountFolderPerLabelSet
    #
    # # Returns the labels given label occurs along side of
    # def findLabelsOccuringWith(self, label) -> list:
    #     return [y for y in list(set(self.getAllFlatenedLabels())) if
    #             y in [z for x in self.getCountFolderPerLabelSet() if label in x for z in x]]
    #
    # def findLabelsNotOccuringWith(self, label) -> list:
    #     return [y for y in list(set(self.getAllFlatenedLabels())) if
    #             y not in [z for x in self.getCountFolderPerLabelSet() if label in x for z in x]]
    #
    # def findLabelsOccuringAtSameTimeWith(self, label) -> list:
    #     return [y for y in list(set(self.getAllFlatenedLabels())) if
    #             y in [l for folder in self.getAllLabels() for file in folder for labels in file if
    #                   label in labels and len(labels) > 1 for l in labels]]
    #
    # def findLabelsNotOccuringAtSameTimeWith(self, label) -> list:
    #     return [y for y in list(set(self.getAllFlatenedLabels())) if
    #             y not in [l for folder in self.getAllLabels() for file in folder for labels in file if
    #                       label in labels and len(labels) > 1 for l in labels]]
    #
    # # def findFilesWithLabels(self, labels: list) ->list:
    #
    # def findIndexesFoldersWithLabels(self, labels: list) -> list:
    #     return [fold_idx for fold_idx in range(0, len(self.files)) if
    #             len(set(self.getAllLabelSets()[fold_idx]).intersection(labels)) > 0]
    #
    # def findFoldersWithLabels(self, labels: list) -> list:
    #     return [self.files[x] for x in self.findIndexesFoldersWithLabels(labels)]
    #
    # # def playAudioFiles(self, toplayindexes: list):
    # #     for i in toplayindexes:
    # #         for w in self.files[i]:
    # #             wav = w['wav_file'].replace('/home/cj/', 'C:\\Users\\mrKC1\\Documents\\TU Delft\\Thesis '
    # #                                                      '2\\Dataset\\audioset\\')
    # #             print(wav)
    # #             print(w['labels'])
    # #             winsound.PlaySound(wav, winsound.SND_FILENAME)
    #
    # def printStats(self):
    #     # total count
    #     print('total folders: {}'.format(len(self.files)))
    #     print('total files: {}'.format(len(self.getAllFlatenedFilesLabels())))
    #
    #     # total count per label
    #     print('counts of labels: {}'.format(self.getCountPerLabel()))
    #     # histogram of labels, sorted by highest count
    #     self.showBarChart(self.getCountPerLabel())
    #
    #     # Count sequences per label
    #     print('counts of sequences per sets of labels: {}'.format(self.getCountFolderPerLabel()))
    #
    #     # Count sequences per groups of labels
    #     # Show label sets and counts grouped per label
    #     print('sequence count per set of labels grouped by label: {}'.format(self.getCountFolderPerLabelSet()))
    #
    #     # Show bar chart, grouping by label, stacking the sets
    #     # 1 bar = (positions = single labels, numbers = counts of set)
    #     # for each set plot a bar with
    #
    # def showHistogram(self, data):
    #     n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
    #                                 alpha=0.7, rwidth=0.85)
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.xlabel('Label')
    #     plt.ylabel('Count')
    #     plt.title('Count of labels')
    #     pass
    #
    # def showBarChart(self, data: dict):
    #     plt.bar(*zip(*data.items()))
    #     plt.xticks(rotation=90)
    #     plt.show()
    #
    # def showStackedFor(self):
    #     bottomdict = dict(zip(list(set(self.flattenedlabels)), [0 for _ in set(self.flattenedlabels)]))
    #
    #     for l in self.labelsetcounts:
    #         x = list(l[0])
    #         heights = [l[1] for _ in x]
    #         bottoms = [bottomdict[lab] for lab in x]
    #         for labid in range(0, len(x)):
    #             bottomdict[x[labid]] += heights[labid]
    #         plt.bar(x, heights, bottom=bottoms)
    #
    #     # plt.legend([l[0] for l in self.labelsetcounts], loc=3)
    #     plt.xticks(rotation=90)
    #     plt.show()
