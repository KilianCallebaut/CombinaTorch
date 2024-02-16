import os
import sys
from random import randrange

import torch

from CombinaTorch.DataReaders.ExtractionMethod import  MelSpectrogramExtractionMethod
from CombinaTorch.MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from CombinaTorch.Tasks.TaskDatasets.ConcatTaskDataset import ConcatTaskDataset
from CombinaTorch.Tasks.Task import MultiClassTask
from CombinaTorch.Tasks.TaskDatasets.TaskDataset import TaskDataset
from CombinaTorch.Training.Training import Training


def test_single_task_Multi_Class(argv):
    if not os.path.isdir('./TestPath'):
        os.mkdir('./TestPath')
    extraction_method = MelSpectrogramExtractionMethod()
    dataset = TaskDataset(inputs=[torch.rand(100, 100) for _ in range(64)],
                          targets=[[randrange(1) for _ in range(5)] for _ in range(64)],
                          task=MultiClassTask(name='Test', output_labels=[0, 1, 2, 3, 4]),
                          extraction_method=extraction_method, base_path='./TestPath', index_mode=False)

    concat_dataset = ConcatTaskDataset([dataset])

    testdataset = TaskDataset(inputs=[torch.rand(100, 100) for _ in range(64)],
                              targets=[[randrange(1) for _ in range(5)] for _ in range(64)],
                              task=MultiClassTask(name='Test_test', output_labels=[0, 1, 2, 3, 4]),
                              extraction_method=extraction_method, base_path='./TestPath', index_mode=False)

    concat_dataset_test = ConcatTaskDataset([testdataset])

    model = MultiTaskHardSharingConvolutional(1, 64, 4, concat_dataset.get_task_list())
    model = model.cuda()
    results = Training.create_results(modelname=model.name, task_list=concat_dataset.get_task_list(),
                                      model_checkpoints_path='./TestPath', num_epochs=2)
    model, results = Training.run_gradient_descent(model=model,
                                                   concat_dataset=concat_dataset,
                                                   results=results,
                                                   batch_size=64,
                                                   num_epochs=2,
                                                   test_dataset=concat_dataset_test)
    print('ran')

def test_single_task_Multi_Class_TrainingSetCreator(argv):
    extraction_method = MelSpectrogramExtractionMethod()
    dataset = TaskDataset(inputs=[torch.rand(100, 100) for _ in range(64)],
                          targets=[[randrange(1) for _ in range(5)] for _ in range(64)],
                          task=MultiClassTask(name='Test', output_labels=[0, 1, 2, 3, 4]),
                          extraction_method=extraction_method, base_path='./TestPath', index_mode=False)

    concat_dataset = ConcatTaskDataset([dataset])

    testdataset = TaskDataset(inputs=[torch.rand(100, 100) for _ in range(64)],
                              targets=[[randrange(1) for _ in range(5)] for _ in range(64)],
                              task=MultiClassTask(name='Test_test', output_labels=[0, 1, 2, 3, 4]),
                              extraction_method=extraction_method, base_path='./TestPath', index_mode=False)

    concat_dataset_test = ConcatTaskDataset([testdataset])

    model = MultiTaskHardSharingConvolutional(1, 64, 4, concat_dataset.get_task_list())
    model = model.cuda()
    results = Training.create_results(modelname=model.name, task_list=concat_dataset.get_task_list(),
                                      model_checkpoints_path='./TestPath', num_epochs=2)
    model, results = Training.run_gradient_descent(model=model,
                                                   concat_dataset=concat_dataset,
                                                   results=results,
                                                   batch_size=64,
                                                   num_epochs=2,
                                                   test_dataset=concat_dataset_test)
    print('ran')

if __name__ == "__main__":
    try:
        sys.exit(test_single_task_Multi_Class(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
