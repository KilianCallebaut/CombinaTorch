import os
import random
import sys
from datetime import timedelta
from timeit import default_timer as timer

import librosa
import numpy as np
import soundfile
import torch
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from MultiTask.ParkClassifier import ParkClassifier

drive = r"E:/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Park(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, item):
        return self.inputs[item], torch.tensor(self.targets[item]).float()

    def __len__(self):
        return len(self.inputs)


def group_events(list_of_events):
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
    return grouped_list


def model_loop(loader, distinct_targets, model, writer, optimizer, criterium, phase):
    if phase == 'train':
        model.train()  # Set model to training mode
    elif phase == 'test':
        model.eval()
    for epoch in range(200):

        print('Epoch {}'.format(epoch))
        print('===========================================================')

        truths = []
        predictions = []
        # iterate over data
        for inputs, labels in loader:
            # define .to(device) on dataloader(s) to make it run on gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            # optimizer.zero_grad() to zero parameter gradients
            optimizer.zero_grad()

            # model
            output = model(inputs)[0]
            loss = criterium(output, labels)

            truths += labels.tolist()
            predictions += (output >= 0.5).float().tolist()

            if phase == 'train':
                # update
                loss.backward()
                optimizer.step()

        report = metrics.classification_report(truths, predictions, output_dict=True, target_names=distinct_targets)
        for key, value in report.items():
            for key2, value2 in value.items():
                writer.add_scalar(r'{}/{}/{}'.format(phase, key, key2), value2, epoch)


def sample(lb, limit, targets, inputs, grouping):
    label_set = [i for i in range(len(targets))
                 if lb in targets[i]]
    random_label_set = random.sample(label_set, limit)
    sampled_targets = [targets[i] for i in range(len(targets)) if (i not in label_set or i in random_label_set)]
    sampled_inputs = [inputs[i] for i in range(len(inputs)) if (i not in label_set or i in random_label_set)]
    sampled_grouping = [grouping[i] for i in range(len(grouping)) if (i not in label_set or i in random_label_set)]
    return sampled_inputs, sampled_targets, sampled_grouping


def load(sampled_inputs, sampled_targets, sampled_grouping):
    kf = GroupKFold(n_splits=4).split(sampled_inputs, groups=sampled_grouping)
    train_indexes, test_indexes = next(kf)
    train_inputs = [sampled_inputs[i] for i in range(len(sampled_inputs)) if i in train_indexes]
    train_targets = [sampled_targets[i] for i in range(len(sampled_targets)) if i in train_indexes]
    train_grouping = [sampled_grouping[i] for i in range(len(sampled_grouping)) if i in train_indexes]
    test_inputs = [sampled_inputs[i] for i in range(len(sampled_inputs)) if i in test_indexes]
    test_targets = [sampled_targets[i] for i in range(len(sampled_targets)) if i in test_indexes]
    test_grouping = [sampled_grouping[i] for i in range(len(sampled_grouping)) if i in test_indexes]

    distinct_targets = list(set([x for l in sampled_targets for x in l]))
    train_targets = [[int(b in f) for b in distinct_targets] for f in train_targets]
    park_dataset = Park(train_inputs, train_targets)
    test_targets = [[int(b in f) for b in distinct_targets] for f in test_targets]
    park_dataset_test = Park(test_inputs, test_targets)
    return park_dataset, park_dataset_test, distinct_targets


def main(argv):
    ##DATA READING##
    data_path = r'E:\Thesis_Datasets\audioset_chen\audioset_filtered'
    train_dir = "balanced_train_segments"
    inputs = []
    targets = []
    grouping = []

    extraction_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    extraction_model.eval()
    with torch.no_grad():
        for _, dirs, _ in os.walk(os.path.join(data_path, train_dir)):
            cdt = len(dirs)
            cd = 0
            cn = 0
            start = timer()

            for dir in dirs:
                perc = (cd / cdt) * 100

                cd += 1

                for file in os.listdir(os.path.join(data_path, train_dir, dir)):
                    filepath = os.path.join(data_path, train_dir, dir, file)
                    if file.endswith('.npy'):
                        np_obj = np.load(filepath, allow_pickle=True).item()
                        wav_loc = np_obj['wav_file'].split(r'/')[6]
                        wav_loc = os.path.join(data_path, train_dir, dir, wav_loc)
                        read = soundfile.read(wav_loc)
                        resampled = librosa.core.resample(*read, 16000)
                        if len(resampled) < 16000:
                            resampled = np.pad(resampled, (0, 16000 - len(resampled)))
                        read = (resampled, 16000)
                        try:
                            embedding = extraction_model.forward(*read)[None,:]
                        except:
                            continue

                        labels = group_events([l[2] for l in np_obj['labels']])
                        inputs.append(embedding)
                        targets.append(labels)
                        grouping.append(cd)
                if perc > cn * 10:
                    print((cd / cdt) * 100)
                    end = timer()
                    timedel = end - start
                    print('estimated time: {}'.format(timedelta(seconds=timedel * (10 - cn))))
                    start = end
                    cn += 1

    ## data loading
    sample_i, sample_t, sample_g = sample('others', 500, targets, inputs, grouping)
    train, test, distinct_targets = load(sample_i, sample_t, sample_g)

    ## Training
    model = ParkClassifier(output_amount=10).to(device)
    criterium = nn.BCELoss().to(device)
    writer = SummaryWriter(
        log_dir=r"E:\Thesis_Results\Park\TensorBoard"
    )
    loader = torch.utils.data.DataLoader(
        train,
        num_workers=0,
        pin_memory=False,
        batch_size=256
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_loop(loader=loader,
               distinct_targets=distinct_targets,
               model=model,
               writer=writer,
               optimizer=optimizer,
               criterium=criterium,
               phase='train')

    loader = torch.utils.data.DataLoader(
        test,
        num_workers=0,
        pin_memory=False,
        batch_size=256
    )
    with torch.no_grad():
        model_loop(loader=loader, distinct_targets=distinct_targets, model=model, writer=writer, optimizer=optimizer,
                   criterium=criterium, phase='test')
    writer.close()

    ## data loading
    sample_i, sample_t, sample_g = sample('speech', 0, sample_t, sample_i, sample_g)
    train, test, distinct_targets = load(sample_i, sample_t, sample_g)

    ## Training
    model = ParkClassifier(output_amount=len(distinct_targets)).to(device)
    criterium = nn.BCELoss().to(device)
    writer = SummaryWriter(
        log_dir=r"E:\Thesis_Results\Park\TensorBoard"
    )
    loader = torch.utils.data.DataLoader(
        train,
        num_workers=0,
        pin_memory=False,
        batch_size=256
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_loop(loader=loader, distinct_targets=distinct_targets, model=model, writer=writer, optimizer=optimizer,
               criterium=criterium, phase='train')

    loader = torch.utils.data.DataLoader(
        test,
        num_workers=0,
        pin_memory=False,
        batch_size=256
    )
    with torch.no_grad():
        model_loop(loader=loader, distinct_targets=distinct_targets, model=model, writer=writer, optimizer=optimizer,
                   criterium=criterium, phase='test')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
