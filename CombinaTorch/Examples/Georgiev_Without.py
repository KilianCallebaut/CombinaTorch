import itertools
import math
import os
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile
import torch
from dcase_util.datasets import TUTAcousticScenes_2017_DevelopmentSet, TUTAcousticScenes_2017_EvaluationSet
from python_speech_features import logfbank
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter

from CombinaTorch.MultiTask.GeorgievMultiDNN import GeorgievMultiDNN

drive = r"E:/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Data(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.total_target_size = len(self.targets[0])
        self.start_index = 0
        self.stop_index = len(self.targets[0]),
        self.group = 0

    def __getitem__(self, item):
        return self.inputs[item], torch.tensor(self.targets[item]).float(), torch.tensor(self.group)

    def get_target(self, item):
        targets = np.zeros(self.total_target_size, dtype=int)
        targets[self.start_index:self.stop_index] = self.targets[item]
        return targets

    def __len__(self):
        return len(self.inputs)


def extract(sig_samplerate):
    lb = torch.transpose(torch.tensor(logfbank(*sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512)), 0, 1)
    return torch.stack([torch.min(lb, 1).values, torch.max(lb, 1).values, torch.std(lb, 1), torch.mean(lb, 1),
                        torch.median(lb, 1).values, torch.tensor(np.percentile(lb, 25, axis=1)),
                        torch.tensor(np.percentile(lb, 75, axis=1))])


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
    data_path = r"E:\Thesis_Datasets\Ravdess"
    song_folder = 'Audio_Song_actors_01-24'
    speech_folder = 'Audio_Speech_Actors_01-24'
    for fold in [song_folder, speech_folder]:
        for _, songs, _ in os.walk(os.path.join(data_path, fold)):
            for ss_dir in songs:
                for file in os.listdir(os.path.join(data_path, song_folder, ss_dir)):
                    mod, voc, em, emi, stat, rep, act = file[:-4].split('-')
                    read = soundfile.read(os.path.join(data_path, song_folder, ss_dir, file))
                    if len(read[0].shape) > 1:
                        read = (np.mean(read[0], axis=1), read[1])
                    read = (librosa.core.resample(*read, 16000), 16000)
                    extracted['ravdess_em'].append(extract(read))
                    targets['ravdess_em'].append(em)
                    grouping['ravdess_em'].append(act)
                    if int(em) == 6 or int(em) == 1:
                        read = (read[0][0:math.floor(1.28 * 16000)], 16000)
                        extracted['ravdess_s'].append(extract(read))
                        targets['ravdess_s'].append(em)
                        grouping['ravdess_s'].append(act)
    distinct_targets_em = list(set(targets['ravdess_em']))
    distinct_targets_s = list(set(targets['ravdess_s']))
    targets['ravdess_em'] = [[int(b == f) for b in distinct_targets_em] for f in targets['ravdess_em']]
    targets['ravdess_s'] = [[int(b == f) for b in distinct_targets_s] for f in targets['ravdess_s']]
    data_path = r"E:\Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015\DS_10283_853"
    truths = pd.read_csv(os.path.join(data_path, 'Joint_ASV_CM_protocol', 'ASV_male_development.ndx'),
                         sep=' ',
                         header=None,
                         names=['folder', 'file', 'method', 'source'])
    truths_female = pd.read_csv(
        os.path.join(data_path, 'Joint_ASV_CM_protocol', 'ASV_female_development.ndx'), sep=' ',
        header=None,
        names=['folder', 'file', 'method', 'source'])
    male_eval = pd.read_csv(os.path.join(data_path, 'Joint_ASV_CM_protocol', 'ASV_male_evaluation.ndx'),
                            sep=' ', header=None,
                            names=['folder', 'file', 'method', 'source'])
    female_eval = pd.read_csv(os.path.join(data_path, 'Joint_ASV_CM_protocol', 'ASV_female_evaluation.ndx'),
                              sep=' ',
                              header=None,
                              names=['folder', 'file', 'method', 'source'])
    male_enrol = pd.read_csv(os.path.join(data_path, 'Joint_ASV_CM_protocol', 'ASV_male_enrolment.ndx'),
                             sep=' ', header=None,
                             names=['folder', 'file', 'method', 'source'])
    female_enrol = pd.read_csv(os.path.join(data_path, 'Joint_ASV_CM_protocol', 'ASV_female_enrolment.ndx'),
                               sep=' ',
                               header=None,
                               names=['folder', 'file', 'method', 'source'])
    truths.append(truths_female)
    truths.append(male_eval)
    truths.append(female_eval)
    truths.append(male_enrol)
    truths.append(female_enrol)
    truths.sort_values(['folder', 'file'], inplace=True)
    files = [os.path.join(data_path, 'wav', x[0], x[1]) for x in truths.to_numpy()]
    distinct_ravdess = truths.folder.unique()
    distinct_ravdess.sort()
    distinct_ravdess = np.append(distinct_ravdess, 'unknown')
    perc = 0
    for audio_idx in range(len(files)):
        try:
            read = soundfile.read(files[audio_idx] + '.wav')
            read = (librosa.core.resample(*read, 16000), 16000)
            extracted['asv'].append(extract(read))
            target = [int(distinct_ravdess[label_id] == truths.loc[audio_idx].folder)
                      if (truths.loc[audio_idx].method == 'genuine' or truths.loc[audio_idx].method == 'human')
                      else int(label_id == len(distinct_ravdess) - 1)
                      for label_id in range(len(distinct_ravdess))]
            targets['asv'].append(target)
        except Exception as e:
            print(e)
        if perc < (audio_idx / len(files)) * 100:
            print("Percentage done: {}".format(perc))
            perc += 1
    return extracted, targets, grouping, [distinct_labels, distinct_targets_em, distinct_targets_s, distinct_ravdess]


def model_loop(loader, distinct_targets, model, writer, optimizer, criterium, phase):
    if phase == 'train':
        model.train()  # Set model to training mode
    elif phase == 'test':
        model.eval()

    for epoch in range(200):
        model.train()  # Set model to training mode

        print('Epoch {}'.format(epoch))
        print('===========================================================')

        truths = [[] for dt in loader.dataset.datasets]
        predictions = [[] for dt in loader.dataset.datasets]
        # iterate over data
        for inputs, labels, groups in loader:
            batch_flags = [[True if dt.group in g else False for g in groups] for dt in loader.dataset.datasets]
            label_flags = [[labels[i] for i in range(dt.start_index, dt.stop_index)] for dt in loader.dataset.datasets]

            # define .to(device) on dataloader(s) to make it run on gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            # optimizer.zero_grad() to zero parameter gradients
            optimizer.zero_grad()

            # model
            output_s = model(inputs)
            losses = []
            for dt_i in range(len(loader.dataset.datasets)):
                losses.append(criterium(output_s[dt_i][batch_flags[dt_i]], label_flags[dt_i]))

                truths[dt_i] += label_flags[dt_i].tolist()
                predictions[dt_i] += (output_s[dt_i] >= 0.5).float().tolist()
            loss = sum(losses)
            if phase == 'train':
                # update
                loss.backward()
                optimizer.step()
        for dt_i in range(len(loader.dataset.datasets)):
            report = metrics.classification_report(truths[dt_i], predictions[dt_i], output_dict=True,
                                                   target_names=distinct_targets[dt_i])
            for key, value in report.items():
                for key2, value2 in value.items():
                    writer.add_scalar(r'{}/{}/{}/{}'.format(dt_i, phase, key, key2), value2, epoch)


def main(argv):
    ## Data Reading
    inputs, targets, groupings, distinct_labels = read_data()

    ## Data Loading
    dcase_train_dataset = Data(
        inputs=inputs['dcase_train'],
        targets=targets['dcase_train']
    )
    dcase_test_dataset = Data(
        inputs=inputs['dcase_test'],
        targets=targets['dcase_test']
    )
    kf = GroupKFold(n_splits=5)
    ravdess_splitter = kf.split(inputs['ravdess_em'], groups=groupings['ravdess_em'])
    asv_splitter = kf.split(inputs['asv'], groups=groupings['asv'])
    for train_idx, test_idx in ravdess_splitter:
        ravdess_em_dataset_train = Data(
            inputs=[inputs['ravdess_em'][i] for i in range(len(inputs['ravdess_em'])) if i in train_idx],
            targets=[targets['ravdess_em'][i] for i in range(len(inputs['ravdess_em'])) if i in train_idx])
        ravdess_em_dataset_test = Data(
            inputs=[inputs['ravdess_em'][i] for i in range(len(inputs['ravdess_em'])) if i in test_idx],
            targets=[targets['ravdess_em'][i] for i in range(len(inputs['ravdess_em'])) if i in test_idx])
        ravdess_s_dataset_train = Data(
            inputs=[inputs['ravdess_s'][i] for i in range(len(inputs['ravdess_s'])) if i in train_idx],
            targets=[targets['ravdess_s'][i] for i in range(len(inputs['ravdess_s'])) if i in train_idx])
        ravdess_s_dataset_test = Data(
            inputs=[inputs['ravdess_s'][i] for i in range(len(inputs['ravdess_s'])) if i in test_idx],
            targets=[targets['ravdess_s'][i] for i in range(len(inputs['ravdess_s'])) if i in test_idx])
        train_idx_asv, test_idx_asv = next(asv_splitter)
        asv_dataset_train = Data(
            inputs=[inputs['asv'][i] for i in range(len(inputs['asv'])) if i in train_idx_asv],
            targets=[targets['asv'][i] for i in range(len(inputs['asv'])) if i in train_idx_asv])
        asv_dataset_test = Data(
            inputs=[inputs['asv'][i] for i in range(len(inputs['asv'])) if i in test_idx_asv],
            targets=[targets['asv'][i] for i in range(len(inputs['asv'])) if i in test_idx_asv])
        training = ConcatDataset(
            datasets=[dcase_train_dataset, ravdess_em_dataset_train, ravdess_s_dataset_train, asv_dataset_train])
        testing = ConcatDataset(
            datasets=[dcase_test_dataset, ravdess_em_dataset_test, ravdess_s_dataset_test, asv_dataset_test])

        scalers = []
        for i in range(7):
            scalers[i] = StandardScaler()
            for inp_idx in range(len(training)):
                scalers[i].partial_fit(training.__getitem__(inp_idx))
            for inp_idx in range(len(training)):
                input_tensor = training.__getitem__(inp_idx)
                input_tensor[i, :] = torch.tensor(scalers[i].transform(input_tensor[i, :].reshape(1, -1)))
            for inp_idx in range(len(testing)):
                input_tensor = testing.__getitem__(inp_idx)
                input_tensor[i, :] = torch.tensor(scalers[i].transform(input_tensor[i, :].reshape(1, -1)))

        ## Training
        keys = [i for i in range(4)]
        comb_iterator = itertools.chain(*map(lambda x: itertools.combinations(keys, x), range(0, len(keys) + 1)))
        for i_list in comb_iterator:
            i_list = list(i_list)
            if len(i_list) > 1:
                for i in range(len(i_list[1:])):
                    index = i_list[i + 1]
                    offset = sum([len(training.datasets[ind].targets[0]) for ind in i_list[:i]])
                    training.datasets[index].start_index = offset
                    testing.datasets[index].start_index = offset
                    training.datasets[index].stop_index = offset + len(training.datasets[index].targets[0])
                    testing.datasets[index].stop_index = offset + len(training.datasets[index].targets[0])
            for i in i_list:
                training.datasets[i].total_target_size = training.dataset[-1].stop_index
                training.datasets[i].group = i_list.index(i)
                testing.datasets[i].total_target_size = testing.dataset[-1].stop_index
                testing.datasets[i].total_target_size = i_list.index(i)

            model = GeorgievMultiDNN(hidden_size=512, n_hidden=4, input_size=7 * 24,
                                     output_sizes=[len(training.datasets[i].targets[0]) for i in
                                                   range(len(i_list))]).to(device)
            train_data = ConcatDataset([training.datasets[i] for i in i_list])
            test_data = ConcatDataset([testing.datasets[i] for i in i_list])
            distinct_labels_it = [distinct_labels[l] for l in i_list]
            criterium = torch.nn.CrossEntropyLoss().to(device)
            writer = SummaryWriter(
                log_dir=r"E:\Thesis_Results\Park\TensorBoard"
            )
            loader = torch.utils.data.DataLoader(
                train_data,
                num_workers=0,
                pin_memory=False,
                batch_size=32
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model_loop(loader=loader, distinct_targets=distinct_labels_it, model=model, writer=writer,
                       optimizer=optimizer, criterium=criterium, phase='train')

            loader = torch.utils.data.DataLoader(
                test_data,
                num_workers=0,
                pin_memory=False,
                batch_size=32
            )
            with torch.no_grad():
                model_loop(loader=loader, distinct_targets=distinct_labels_it, model=model, writer=writer,
                           optimizer=optimizer, criterium=criterium, phase='test')
            writer.close()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
