import os
import random
import types
from typing import List, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from CombinaTorch.DataReaders.ExtractionMethod import ExtractionMethod
from CombinaTorch.Tasks.Task import Task


# In index mode:
# before save -> full tensors in input list
# after save or after to_index -> fragment id's in inputs
# when splitting into train and test -> filenames in inputs

class TaskDataset(Dataset):
    """
    Unified object to store dataset and task information and perform transformations.

    Has 2 modes: index and non-index mode.
    In index mode, the object only stores indices pointing to input files. These should be created first using the
    write_index_files method. When indexing, the inputs are live read from the corresponding file.
    In non-index mode, the whole dataset is loaded into memory.
    """

    # data_lists: lists of lists of instances per task
    # now requires sampling to be called beforehand, save scalers to be called beforehand
    def __init__(
            self,
            extraction_method: ExtractionMethod,
            base_path: str,
            index_mode=False
    ):
        """
        Initializes the Taskdataset object

        :param extraction_method: ExtractionMethod object, responsible for appropriate transformations and name of the data for saving and loading
        :param base_path: The path to the folder where the data should be saved
        :param index_mode: Boolean to switch on/off index mode
        """
        self.index_mode = index_mode
        self.extraction_method = extraction_method
        self.base_path = base_path

        self.start_index_list = []
        self.stop_index_list = []

        self.flag_prepared = False
        self.flag_scaled = False
        self._total_target_size = 0

        self.inputs = []
        self.targets = []
        self.task = None
        self.grouping = []
        self.extra_tasks = []

        if self.index_mode:
            self.switch_index_methods()
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)

    ########################################################################################################
    # Initialization
    ########################################################################################################

    def add_input(self, input_tensor: torch.tensor):
        """
        Adds an input tensor to the dataset
        :param input_tensor: The feature matrix to insert into the dataset
        """
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor[None, :]
        self.inputs.append(input_tensor)

    def extract_and_add_input(self,
                              sig_samplerate: Tuple[np.ndarray, int]):
        """
        Adds an input tensor to the dataset
        :param sig_samplerate: The tuple with the signal object and the samplerate
        """

        if len(sig_samplerate[0].shape) > 1:
            channel = 1
            input_tensor = torch.stack(
                [self.extraction_method.extract_features((sig_samplerate[0][:, i], sig_samplerate[1])) for i in
                 range(sig_samplerate[0].shape[channel])])
        else:
            input_tensor = self.extraction_method.extract_features(sig_samplerate)
        self.add_input(input_tensor)

    def add_task_and_targets(self, task: Task, targets: List[List[int]]):
        """
        Adds a task object with a list of targets to the dataset.
        The targets should be in the same order as their corresponding inputs
        :param task: The task object to add
        :param targets: The list of target vectors to add
        """
        assert len(targets) == len(self.inputs), 'There must be as many targets as there are inputs'
        for t in targets:
            assert len(t) == len(targets[0]), 'Each target vector must have the same length'
        if not self.targets:
            self.targets = targets
            self.task = task
        else:
            self.extra_tasks.append((task, targets))

        self.start_index_list += [self.total_target_size]
        self.total_target_size += len(task.output_labels)
        self.stop_index_list += [self.total_target_size]

    def add_grouping(self, grouping: List[int]):
        """
        Adds the grouping list.
        The groupings should be in the same order as their corresponding inputs
        :param grouping: optional Grouping list for defining which data cannot be split up in k folds (see sklearn.model_selection.GroupKfold)
        """
        assert len(grouping) == len(self.inputs), 'There must be as many elements in the grouping as there are inputs'
        self.grouping = grouping

    def validate(self):
        """
        Checks if the object is valid
        :return: Bool
        """
        assert len(self.targets) == len(self.inputs), 'There must be as many targets as there are inputs'
        for t in self.targets:
            assert len(t) == len(self.targets[0]), 'Each target vector must have the same length'
        for t_e in self.extra_tasks:
            for t in t_e[1]:
                assert len(t) == len(t_e[1][0]), 'Each target vector must have the same length'
        if self.grouping:
            assert len(self.grouping) == len(
                self.inputs), 'There must be as many elements in the grouping as there are inputs'

        if self.inputs and self.targets and self.task:
            return True
        return False

    ########################################################################################################
    # Getters
    ########################################################################################################

    def __getitem__(self, index):
        """
        The getter method for returning inputs for the data loading of the training
        :param index: The index of the data instance to get
        :return: The feature matrix, the (combined) target vector and the task group
        """
        return self.get_input(index), \
               torch.from_numpy(self.get_all_targets(index)), \
               torch.tensor([t.task_group for t in self.get_all_tasks()])

    def get_input(self, index):
        return self.inputs[index].float()

    def get_all_targets(self, index):
        targets = np.zeros(self.total_target_size, dtype=int)
        targets[self.start_index_list[0]:self.start_index_list[0] + len(self.task.output_labels)] = self.targets[index]
        for i in range(len(self.extra_tasks)):
            targets[
            self.start_index_list[i + 1]:self.start_index_list[i + 1] + len(self.extra_tasks[i][0].output_labels)] = \
            self.extra_tasks[i][1][index]
        return targets

    def get_all_tasks(self):
        tasks = [self.task]
        if self.extra_tasks:
            tasks += [t[0] for t in self.extra_tasks]
        return tasks

    def copy_non_data_variables(self, taskDataset):
        self.start_index_list = taskDataset.start_index_list
        # self.stop_index_list = taskDataset.stop_index_list
        # self.total_target_size = taskDataset.total_target_size

        self.flag_prepared = taskDataset.flag_prepared
        self.flag_scaled = taskDataset.flag_scaled

    def __len__(self):
        return len(self.inputs)

    @property
    def total_target_size(self):
        return self._total_target_size + sum(
            [len(t.output_labels) for t in [self.task] + [tsk[0] for tsk in self.extra_tasks]])

    @total_target_size.setter
    def total_target_size(self, value):
        self._total_target_size = value - sum(
            [len(t.output_labels) for t in [self.task] + [tsk[0] for tsk in self.extra_tasks]])

    ########################################################################################################
    # Combining
    ########################################################################################################
    def pad_targets(self, start_index_list: List[int], stop_index_list: List[int], total_size: int):
        """
        Store the necessary data to combine the targets from each tasks with others
        :param start_index_list: List of starting indexes per task for the combined target vector
        :param stop_index_list: List of ending indexes per task for the combined target vector
        :param total_size: Total size of the combined target vector
        """
        self.start_index_list = start_index_list
        self.stop_index_list = stop_index_list
        self.total_target_size = total_size

    ########################################################################################################
    # Filtering
    ########################################################################################################

    def sample_labels(self, dic_of_labels_limits, random_state=None):
        """
        Samples the instances according to predefined limits of number of instances per label
        :param dic_of_labels_limits: Dictionary specifying the maximum amount of instances a label can have in the dataset
        :param random_state: optional int for reproducability purposes
        """
        print('Sampling labels')
        # sampled_targets = self.targets
        # sampled_inputs = self.inputs
        # sampled_grouping = self.grouping
        # sampled_extra_tasks = self.extra_tasks
        if random_state is not None:
            random.seed(random_state)

        for l in dic_of_labels_limits.keys():
            if l not in self.task.output_labels:
                print('{} not in dataset'.format(l))
            if dic_of_labels_limits[l] == 0:
                self.remove_label(l)
                continue
            label_set = [i for i in range(len(self.targets))
                         if self.targets[i][self.task.output_labels.index(l)] == 1]
            if len(label_set) > dic_of_labels_limits[l]:
                random_label_set = random.sample(label_set, dic_of_labels_limits[l])
                self.remove_elements([ind for ind in label_set if ind not in random_label_set])
                # sampled_targets = [sampled_targets[i] for i in range(len(sampled_targets)) if
                #                    (i not in label_set or i in random_label_set)]
                # sampled_inputs = [sampled_inputs[i] for i in range(len(sampled_inputs)) if
                #                   (i not in label_set or i in random_label_set)]
                # if self.grouping:
                #     sampled_grouping = [sampled_grouping[i] for i in range(len(sampled_grouping)) if
                #                         (i not in label_set or i in random_label_set)]
                # if self.extra_tasks:
                #     for t in sampled_extra_tasks:
                #         t[1] = [t[1][i] for i in range(len(t[1])) if
                #                 (i not in label_set or i in random_label_set)]

        # self.inputs = sampled_inputs
        # self.targets = sampled_targets
        # self.grouping = sampled_grouping
        # self.extra_tasks = sampled_extra_tasks

    def remove_label(self, label=None, index=None):
        """
        Deletes a label from the task and all of its (singular) instances
        :param label: the label to remove
        :param index: the index of the label to remove
        """
        if isinstance(label, type(self.task.output_labels[0])):
            index = self.task.output_labels.index(label)
        elif isinstance(index, int):
            label = self.task.output_labels[index]
        else:
            raise ValueError('either a label or an index must be given')
        self.remove_label_instances(index)
        self.task.output_labels.remove(label)

    def remove_label_instances(self, index):
        """
        Remove all singualre instances of a certain label and its place in one hot encoding
        :param index: the index of the label to remove
        """
        removal = []
        for i in range(len(self)):
            if self.targets[i][index] == 1 and sum(self.targets[i]) == 1:
                removal.append(i)
            else:
                self.targets[i].pop(index)
        if removal:
            self.remove_elements(removal)

    def remove_elements(self, indexes):
        """
        Removes all elements at index
        :param indexes: the indexes to remove input and targets
        """
        self.targets = [self.targets[index] for index in range(len(self)) if index not in indexes]
        if self.grouping:
            self.grouping = [self.grouping[index] for index in range(len(self)) if index not in indexes]
        for t in self.extra_tasks:
            t[1] = [t[1][index] for index in range(len(self)) if index not in indexes]
        self.inputs = [self.inputs[index] for index in range(len(self)) if index not in indexes]

    ########################################################################################################
    # Transformation
    ########################################################################################################

    def normalize_fit(self):
        """
        Calculates the necessary metrics on the datset to use later for scaling the inputs
        """
        print("Calculate Normalize Fit")

        # self.extraction_method.scale_reset()
        if self.flag_scaled:
            self.inverse_normalize_inputs()
        for i in range(len(self.inputs)):
            self.extraction_method.partial_scale_fit(self.get_input(i))

    def normalize_inputs(self):
        """
        Scales the inputs using the normalization defined in the ExtractionMethod object
        """
        print("Calculate Normalization")
        for i in range(len(self.inputs)):
            self.inputs[i] = self.extraction_method.scale_transform(self.get_input(i))
        self.flag_scaled = True

    def inverse_normalize_inputs(self):
        """
        Inverses the scaling of inputs (if they have been scaled)
        """
        if self.flag_scaled:
            print('Inversing normalization')
            for i in range(len(self.inputs)):
                self.inputs[i] = self.extraction_method.inverse_scale_transform(self.get_input(i))
            self.flag_scaled = False

    def prepare_fit(self):
        """
        Calculates the necessary metrics on the dataset to use later for preparation
        """
        print("Calculate Preparation Fit")

        self.extraction_method.prepare_reset()
        self.flag_prepared = False
        for i in range(len(self.inputs)):
            self.extraction_method.prepare_fit(self.get_input(i))

    def prepare_inputs(self):
        """
        Applies preparation of the input instances defined in the extraction method
        on each input instance in the TaskDataset.
        """
        print("Prepare Inputs")
        self.flag_prepared = False
        if not self.grouping:
            self.grouping = [i for i in range(len(self.inputs))]
        for i in range(len(self.inputs)):
            prepared_inputs = self.extraction_method.prepare_input(self.get_input(i))
            self.inputs[i] = prepared_inputs[0]
            for inp_id in range(1, len(prepared_inputs)):
                self.inputs.append(prepared_inputs[inp_id])
                self.targets.append(self.targets[i])
                self.grouping.append(self.grouping[i])
                for t in self.extra_tasks:
                    t[1].append(t[1][i])

        self.flag_prepared = True

    ########################################################################################################
    # Index mode
    ########################################################################################################

    def to_index_mode(self):
        # save inputs in separate files (if not done so already)
        self.index_mode = True
        # replace inputs with index lists
        self.inputs = [(i, 0) for i in range(len(self.inputs))]
        self.switch_index_methods()

    def switch_index_methods(self):
        # replace getitem, get_split_by_index by index based functions
        self.add_input = types.MethodType(add_input_index_mode, self)
        self.get_input = types.MethodType(get_input_index_mode, self)
        self.save = types.MethodType(save_index_mode, self)
        self.load_inputs = types.MethodType(load_inputs_index_mode, self)
        self.normalize_inputs = types.MethodType(normalize_inputs_index_mode, self)
        self.inverse_normalize_inputs = types.MethodType(inverse_normalize_inputs_index_mode, self)
        self.prepare_inputs = types.MethodType(prepare_inputs_index_mode, self)

    ########################################################################################################
    # I/O
    ########################################################################################################
    def save(self):
        joblib.dump(self.inputs,
                    os.path.join(self.base_path, '{}_{}_inputs.gz'.format(self.task.name, self.extraction_method.name)))
        self.save_task_info()

    def save_task_info(self):
        diction = {
            'task': self.task,
            'targets': self.targets,
            'grouping': self.grouping,
            'extra_tasks': self.extra_tasks
        }
        joblib.dump(diction, os.path.join(self.base_path, '{}_info.obj'.format(self.task.name)))

        diction = {
            'extraction_method': self.extraction_method
        }
        joblib.dump(diction,
                    os.path.join(self.base_path,
                                 '{}_{}_extraction_method_params'.format(self.task.name, self.extraction_method.name)))

    def load(self, taskname):
        self.load_inputs(taskname)
        self.inputs = [i.cpu() for i in self.inputs]
        diction = joblib.load(os.path.join(self.base_path, '{}_info.obj'.format(taskname)))
        self.task = diction['task']
        self.targets = diction['targets']
        self.grouping = diction['grouping']
        self.extra_tasks = diction['extra_tasks']

        diction = joblib.load(
            os.path.join(self.base_path,
                         '{}_{}_extraction_method_params'.format(taskname, self.extraction_method.name)))
        self.extraction_method.__dict__.update(diction['extraction_method'].__dict__)

    def load_inputs(self, taskname):
        self.inputs = joblib.load(
            os.path.join(self.base_path, '{}_{}_inputs.gz'.format(taskname, self.extraction_method.name)))

    def check(self, taskname):
        check = os.path.isfile(os.path.join(self.base_path, '{}_info.obj'.format(taskname))) and os.path.isfile(
            os.path.join(self.base_path,
                         '{}_{}_extraction_method_params'.format(taskname, self.extraction_method.name)))

        if self.index_mode:
            return os.path.isdir(
                os.path.join(self.base_path,
                             '{}_input_{}_separated'.format(taskname, self.extraction_method.name))) \
                   and check
        else:
            return os.path.isfile(
                os.path.join(self.base_path, '{}_{}_inputs.gz'.format(taskname, self.extraction_method.name))) \
                   and check


########################################################################################################
# Initialization
########################################################################################################

def add_input_index_mode(self, input_tensor: torch.tensor):
    """
    Adds an input tensor to the dataset
    :param input_tensor:
    """
    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor[None, :]
    separated_dir = os.path.join(self.base_path,
                                 '{}_input_{}_separated'.format(self.task.name, self.extraction_method.name))
    if not os.path.exists(separated_dir):
        os.mkdir(separated_dir)

    input_path = os.path.join(separated_dir, 'input_{}.pickle'.format(len(self.inputs)))
    torch.save(input_tensor, input_path)
    self.inputs.append((len(self.inputs), 0))


########################################################################################################
# Getters
########################################################################################################

def get_input_index_mode(self, index):
    separated_file = os.path.join(self.base_path,
                                  '{}_input_{}_separated'.format(self.task.name, self.extraction_method.name),
                                  'input_{}.pickle'.format(self.inputs[index][0]))
    input_tensor = torch.load(separated_file)
    if self.flag_prepared:
        input_tensor = self.extraction_method.prepare_input(input_tensor)[self.inputs[index][1]]
    if self.flag_scaled:
        input_tensor = self.extraction_method.scale_transform(input_tensor)
    return input_tensor.float()


########################################################################################################
# Transformation
########################################################################################################

def normalize_inputs_index_mode(self):
    self.flag_scaled = True


def inverse_normalize_inputs_index_mode(self):
    self.flag_scaled = False


def prepare_inputs_index_mode(self):
    if not self.grouping:
        self.grouping = [i[0] for i in self.inputs]
    for i in range(len(self.inputs)):
        prep_inputs = self.extraction_method.prepare_input(self.get_input(i))
        for j in range(1, len(prep_inputs)):
            self.inputs.append((i, j))
            self.targets.append(self.targets[i])
            self.grouping.append(self.grouping[i])
            for t in self.extra_tasks:
                t[1].append(t[1][i])
    self.flag_prepared = True


########################################################################################################
# I/O
########################################################################################################

def save_index_mode(self):
    self.save_task_info()


def load_inputs_index_mode(self, taskname):
    separated_dir = os.path.join(self.base_path,
                                 '{}_input_{}_separated'.format(taskname, self.extraction_method.name))
    dir_list = os.listdir(separated_dir)
    self.inputs = [(i, 0) for i in range(len(dir_list))]
