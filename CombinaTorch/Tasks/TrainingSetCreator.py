import copy
from types import FunctionType
from typing import Optional, List

from sklearn.model_selection import BaseCrossValidator

from CombinaTorch.DataReaders.DataReader import DataReader
from CombinaTorch.DataReaders.ExtractionMethod import ExtractionMethod
from CombinaTorch.Tasks.ConcatTaskDataset import ConcatTaskDataset
from CombinaTorch.Tasks.TaskDataset import TaskDataset


class ConcatTrainingSetCreator:
    """
    Pipeline creator used to assemble and execute extraction and pre-processing of raw datasets into valid input for
    multi-task learning schemes.
    """

    def __init__(self,
                 random_state: Optional[int] = None,
                 nr_runs: Optional[int] = 5,
                 index_mode: bool = False,
                 recalculate: bool = False,
                 multiply: bool = True
                 ):
        """
        Initializes the pipeline creator
        :param random_state: Optional seed for reproducability of splits etc.
        :param nr_runs: Number of runs in k-fold validation
        :param index_mode: Boolean to activate index mode which turns the TaskDataset into a streaming set-up reading files from disk in stead of loading into memory
        :param recalculate: Recalculate the extracted features every time in stead of using the stored results
        :param multiply: Make a copy of the pipeline additions. Only use this in case the combined datasets need to be processed as if they were one big dataset.
        """
        self.random_state = random_state
        self.nr_runs = nr_runs
        self.index_mode = index_mode
        self.recalculate = recalculate
        self.multiply = multiply

        self.data_readers = {}
        self.extraction_methods = {}
        self.sample_rates = {}
        self.preproccesing = {}
        self.extra_arguments = {}
        self.dics_of_label_limits = {}
        self.validators = {}
        self.taskdatasets = {}
        self.transformations = {}

        self.taskdataset_methods = [tf for tf, y in TaskDataset.__dict__.items() if type(y) == FunctionType]
        self.model = None

    def get_keys(self):
        """
        Returns the keys of the datareaders present in the TrainingSetCreator
        :return: All the keys of the DataReaders.
        """
        return self.data_readers.keys()

    def reset_taskDatasets(self, class_list: List[str] = None):
        """
        Clears the created TaskDatasets.
        :param class_list: List of DataReader keys whos TaskDatasets have to be cleared
        """
        if class_list and self.taskdatasets:
            for k in self.taskdatasets.keys():
                if k not in class_list:
                    del self.taskdatasets[k]
        else:
            del self.taskdatasets
            self.taskdatasets = {}

    def add_data_reader(self,
                        data_reader: DataReader,
                        name=None):
        """
        Adds a new DataReader entry point to the pipeline creator.
        :param data_reader:
        :param name:
        :return:
        """
        self.data_readers[name if name else type(data_reader).__name__] = data_reader

    def __add__pipe__(self,
                      addition,
                      dictionary: dict,
                      key: str = None):
        if not key:
            for k in self.data_readers.keys():
                dictionary[k] = copy.copy(addition) if self.multiply else addition
            self.reset_taskDatasets()
        else:
            assert key in self.data_readers, 'There is no dataset with this key'
            dictionary[key] = addition
            self.reset_taskDatasets([k for k in self.data_readers.keys() if k is not key])

    def __get__pipe__(self,
                      dictionary: dict,
                      key: str = None):
        if key in dictionary:
            return dictionary[key]
        else:
            return None

    def __clear__pipe__(self,
                        dictionary: dict,
                        key: str = None):
        if key in dictionary:
            del dictionary[key]
        else:
            return None

    ###############
    # Extraction
    ###############

    def add_extraction_method(self,
                              extraction_method: ExtractionMethod,
                              key: str = None):
        self.__add__pipe__(addition=extraction_method,
                           dictionary=self.extraction_methods,
                           key=key)

    ###############
    # Signal Preprocessing
    ###############

    def add_signal_preprocessing(self,
                                 preprocess_dict: dict,
                                 key: str = None):
        self.__add__pipe__(addition=preprocess_dict,
                           dictionary=self.preproccesing,
                           key=key)

    ###############
    # Extra Arguments
    ###############

    def add_extra_taskDataset_args(self,
                                   kwargs: dict,
                                   key: str = None):
        self.__add__pipe__(addition=kwargs,
                           dictionary=self.extra_arguments,
                           key=key)

    ###############
    # Filtering
    ###############

    def add_sampling(self,
                     dic_of_labels_limits: dict,
                     key: str = None):
        self.__add__pipe__(addition=dic_of_labels_limits,
                           dictionary=self.dics_of_label_limits,
                           key=key)

    ###############
    # Splitting
    ###############

    def add_cross_validator(self,
                            validator: BaseCrossValidator,
                            key: str = None):
        self.__add__pipe__(addition=validator,
                           dictionary=self.validators,
                           key=key)

    ###############
    # Transformations
    ###############
    def add_transformation_call(self,
                                function: str,
                                key: str = None,
                                **kwargs):
        assert function in self.taskdataset_methods, 'Input is not a TaskDataset method'
        if not key:
            for dr in list(self.get_keys()):
                if dr not in self.transformations:
                    self.transformations[dr] = []
                tr_names = [t[0] for t in self.transformations[dr]]
                if function in tr_names:
                    self.transformations[dr][tr_names.index(function)] = (function, kwargs)
                else:
                    self.transformations[dr].append((function, kwargs))

        else:
            assert key in self.data_readers, 'There is no dataset with this key'
            if key not in self.transformations:
                self.transformations[key] = []
            tr_names = [t[0] for t in self.transformations[key]]
            if function in tr_names:
                self.transformations[key][tr_names.index(function)] = (function, kwargs)
            else:
                self.transformations[key].append((function, kwargs))

    def __execute_functions__(self,
                              key: str = None):
        if not set(self.transformations.keys()).intersection(self.taskdatasets.keys()):
            return
        if not key:
            transformation_names = [[fname for fname, _ in self.transformations[key]] for key in self.transformations]
            shared_transformations = list(set(transformation_names[0]).intersection(*transformation_names))
            iterators = {key: 0 for key in self.transformations}
            shared_run = False
            while all(iterators[key] < len(self.transformations[key]) for key in self.transformations):
                if not shared_run:
                    for key in set(self.transformations.keys()).intersection(self.taskdatasets.keys()):
                        for i in range(iterators[key], len(self.transformations[key])):
                            iterators[key] = i
                            if self.transformations[key][i][0] not in shared_transformations:
                                self.__execute_function__(task=self.taskdatasets[key],
                                                          fname=self.transformations[key][i][0],
                                                          kwargs=self.transformations[key][i][1])
                            else:
                                break
                    shared_run = True
                else:
                    for key in set(self.transformations.keys()).intersection(self.taskdatasets.keys()):
                        self.__execute_function__(task=self.taskdatasets[key],
                                                  fname=self.transformations[key][iterators[key]][0],
                                                  kwargs=self.transformations[key][iterators[key]][1])
                        iterators[key] += 1
                        if iterators[key] < len(self.transformations[key]) and \
                                self.transformations[key][iterators[key]][0] not in shared_transformations:
                            shared_run = False
        elif key in self.transformations:
            task = self.taskdatasets[key]
            for fname, kwargs in self.transformations[key]:
                self.__execute_function__(task, fname, kwargs)

    def __execute_function__(self,
                             task,
                             fname,
                             kwargs=None):
        func = getattr(task, fname)
        if kwargs:
            func(**kwargs)
        else:
            func()

    def create_taskdatasets(self,
                            class_list: List[str] = None,
                            execute_transformations=True):
        self.reset_taskDatasets(class_list)
        for dr in self.data_readers.keys():
            if (class_list and dr not in class_list) or dr in self.taskdatasets.keys():
                continue

            task_input_dict = dict(
                extraction_method=self.__get__pipe__(key=dr, dictionary=self.extraction_methods),

                recalculate=self.recalculate,
                index_mode=self.index_mode,
            )
            if self.__get__pipe__(key=dr, dictionary=self.preproccesing):
                task_input_dict.update(
                    dict(preprocess_parameters=self.__get__pipe__(key=dr, dictionary=self.preproccesing)))
            if self.__get__pipe__(key=dr, dictionary=self.extra_arguments):
                task_input_dict.update(**self.__get__pipe__(key=dr, dictionary=self.extra_arguments))
            tsk = self.data_readers[dr].return_taskDataset(
                **task_input_dict
            )
            if dr in self.dics_of_label_limits:
                tsk.sample_labels(self.__get__pipe__(key=dr, dictionary=self.dics_of_label_limits))

            self.taskdatasets[dr] = tsk
        if execute_transformations:
            self.__execute_functions__()
        return ConcatTaskDataset(list(self.taskdatasets.values()))

    def generate_training_splits(self, class_list: List[str] = None):
        self.create_taskdatasets(class_list,
                                 execute_transformations=False)
        task_gens = []
        for dr in self.taskdatasets.keys():
            tsk = self.taskdatasets[dr]
            task_gens.append(
                tsk.generate_train_test_set(random_state=self.random_state,
                                            n_splits=self.nr_runs,
                                            kf=self.__get__pipe__(
                                                dictionary=self.validators,
                                                key=dr)
                                            )
            )

        fold = 0
        for train_tests in zip(*task_gens):
            print("Fold: {}".format(fold))
            fold += 1
            self.__execute_functions__()
            yield ConcatTaskDataset([t[0] for t in train_tests]), ConcatTaskDataset([t[1] for t in train_tests]), fold
