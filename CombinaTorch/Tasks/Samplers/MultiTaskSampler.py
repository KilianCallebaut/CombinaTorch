import math
import random

import torch
from torch.utils.data import RandomSampler, BatchSampler, ConcatDataset


class MultiTaskSampler(torch.utils.data.sampler.BatchSampler):

    """
    Iterate over datasets and provide a batch of a single random dataset interchangeably.
    Each batch has a maximum length of the given batch size.
    """

    def __init__(self, dataset: ConcatDataset, batch_size: int):
        """
        :param dataset: ConcatDataset containing 1 or more Datasets
        :param batch_size: Wanted size of each individual batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)

        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = BatchSampler(RandomSampler(cur_dataset), self.batch_size, False)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for i in range(self.number_of_datasets):
            for sample in sampler_iterators[i]:
                final_samples_list.append([push_index_val[i] + s for s in sample])
        random.shuffle(final_samples_list)
        self.final_samples_list = final_samples_list
        self.iterator = iter(self.final_samples_list)

    def __len__(self):
        return sum([math.ceil(len(d) / self.batch_size) for d in self.dataset.datasets])

    def __iter__(self):
        while True:
            try:
                yield next(self.iterator)
            except StopIteration:
                random.shuffle(self.final_samples_list)
                self.iterator = iter(self.final_samples_list)
                break
