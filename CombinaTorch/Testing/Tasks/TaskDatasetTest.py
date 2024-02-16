import copy

import soundfile
import torch
import unittest
from CombinaTorch.DataReaders.ExtractionMethod import NeutralExtractionMethod, LogbankExtraction, FilterBankExtractionMethod, \
    MelSpectrogramExtractionMethod
from CombinaTorch.Tasks.TaskDataset import TaskDataset

class TaskDatasetTest(unittest.TestCase):
    def test_index_mode_storage(self):
        ex = NeutralExtractionMethod(name='test_ex')
        t1 = TaskDataset(extraction_method=ex,
                         base_path=r'./TaskDatasetTest',
                         index_mode=False)
        t2 = TaskDataset(extraction_method=ex,
                         base_path=r'./TaskDatasetTest',
                         index_mode=True)
        ex_input = torch.rand(4, 4)
        ex_input_2 = torch.rand(4, 4)
        t1.add_input(ex_input)
        t1.add_input(ex_input_2)
        t2.add_input(ex_input)
        t2.add_input(ex_input_2)
        self.assertTrue(torch.equal(t1.get_input(0), t2.get_input(0)))
        self.assertTrue(torch.equal(t1.get_input(1), t2.get_input(1)))

    def create_read_in_data(self, ex):
        t1 = TaskDataset(extraction_method=ex,
                         base_path=r'./TaskDatasetTest',
                         index_mode=False)
        t2 = TaskDataset(extraction_method=copy.copy(ex),
                         base_path=r'./TaskDatasetTest',
                         index_mode=True)
        loc = r"C:\Users\mrKC1\PycharmProjects\Thesis\Testing\TestFiles\test.wav"
        read = soundfile.read(loc)
        t1.extract_and_add_input(read)
        t2.extract_and_add_input(read)
        loc = r"C:\Users\mrKC1\PycharmProjects\Thesis\Testing\TestFiles\test2.wav"
        read = soundfile.read(loc)
        t1.extract_and_add_input(read)
        t2.extract_and_add_input(read)
        return t1, t2

    def test_index_mode_extract_storage(self):
        ex = LogbankExtraction(name='test_ex_log', extraction_method=NeutralExtractionMethod())
        t1, t2 = self.create_read_in_data(ex)
        for i in range(len(t1)):
            self.assertTrue(torch.equal(t1.get_input(i), t2.get_input(i)))

    def test_index_mode_prepare(self):
        ex = MelSpectrogramExtractionMethod()
        t1, t2 = self.create_read_in_data(ex)
        t1.prepare_fit()
        t2.prepare_fit()
        t1.prepare_inputs()
        t2.prepare_inputs()
        for i in range(len(t1)):
            self.assertTrue(torch.equal(t1.get_input(i), t2.get_input(i)))

    def test_index_mode_scale(self):
        ex = MelSpectrogramExtractionMethod()
        t1, t2 = self.create_read_in_data(ex)
        t1.normalize_fit()
        t2.normalize_fit()
        for i in range(len(t1)):
            self.assertTrue(torch.equal(t1.get_input(i), t2.get_input(i)))



if __name__ == "__main__":
    unittest.main()
    print('All Passed')