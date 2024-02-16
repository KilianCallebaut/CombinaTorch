import unittest

import torch

from CombinaTorch.DataReaders.ExtractionMethod import NeutralExtractionMethod
from CombinaTorch.Tasks.Task import Task, MultiClassTask
from CombinaTorch.Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class MyTestCase(unittest.TestCase):
    def test_return_and_distribute_data(self):
        ex = NeutralExtractionMethod(name='test_ex')
        t1 = HoldTaskDataset(extraction_method=ex,
                         base_path=r'./TaskDatasetTest',
                         index_mode=False)
        ex_input = torch.rand(4, 4)
        ex_input_2 = torch.rand(4, 4)
        ex_input_3 = torch.rand(4, 4)
        t1.add_input(ex_input)
        t1.add_input(ex_input_2)
        t1.add_input(ex_input_3)
        t1.add_task_and_targets(MultiClassTask('testtask', ['a', 'b']), [[0, 1], [1, 0], [1, 1]])
        t1.test_indexes = [0]
        t1.distribute_data()
        self.assertTrue(torch.equal(ex_input_2, t1.inputs[0]))
        self.assertEqual(t1.targets[0], [1, 0])
        t1.return_data()
        self.assertTrue(torch.equal(ex_input, t1.inputs[0]))
        self.assertTrue(torch.equal(ex_input_2, t1.inputs[1]))
        self.assertTrue(torch.equal(ex_input_3, t1.inputs[2]))



if __name__ == '__main__':
    unittest.main()
