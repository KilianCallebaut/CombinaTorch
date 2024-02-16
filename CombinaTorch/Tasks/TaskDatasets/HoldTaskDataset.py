import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator, KFold
from sklearn.preprocessing import LabelEncoder
from skmultilearn.model_selection import IterativeStratification

from CombinaTorch.DataReaders.ExtractionMethod import ExtractionMethod
from CombinaTorch.Tasks.TaskDataset import TaskDataset


class HoldTaskDataset(TaskDataset):
    """
    The TaskDataset object that unifies and splits the training/test sets
    """

    def __init__(self,
                 extraction_method: ExtractionMethod,
                 base_path: str = '',
                 index_mode=False,
                 testing_base_path: str = None):
        super().__init__(extraction_method, base_path, index_mode)
        self.__init_test__(testing_base_path if testing_base_path else base_path)

    ########################################################################################################
    # Initializers
    ########################################################################################################

    def __init_test__(self, testing_base_path):
        self.test_set = TaskDataset(
            extraction_method=self.extraction_method,
            base_path=testing_base_path,
            index_mode=self.index_mode
        )
        self.test_indexes = []

    ########################################################################################################
    # Splitters
    ########################################################################################################
    def k_folds(self, random_state: int = None, n_splits: int = 5, kf: BaseCrossValidator = None, stratified=False):
        """
        Produces a k_fold training/test split generator, depending on the task type

        :param stratified:
        :param kf: optional input for inserting own sklearn kfold splitter
        :param n_splits: number of folds
        :param random_state: optional int for reproducability purposes
        :return: _BaseKFold object
        """
        # Examples
        # --------
        # >> > import numpy as np
        # >> > from sklearn.model_selection import KFold
        # >> > X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # >> > y = np.array([1, 2, 3, 4])
        # >> > kf = KFold(n_splits=2)
        # >> > kf.get_n_splits(X)
        # 2
        # >> > print(kf)
        # KFold(n_splits=2, random_state=None, shuffle=False)
        # >> > for train_index, test_index in kf.split(X):
        #     ...
        #     print("TRAIN:", train_index, "TEST:", test_index)
        # ...
        # X_train, X_test = X[train_index], X[test_index]
        # ...
        # y_train, y_test = y[train_index], y[test_index]
        # TRAIN: [2 3]
        # TEST: [0 1]
        # TRAIN: [0 1]
        # TEST: [2 3]

        # Stratified KFold
        # Examples
        # --------
        # >> > import numpy as np
        # >> > from sklearn.model_selection import StratifiedKFold
        # >> > X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # >> > y = np.array([0, 0, 1, 1])
        # >> > skf = StratifiedKFold(n_splits=2)
        # >> > skf.get_n_splits(X, y)
        # 2
        # >> > print(skf)
        # StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
        # >> > for train_index, test_index in skf.split(X, y):
        #     ...
        #     print("TRAIN:", train_index, "TEST:", test_index)
        # ...
        # X_train, X_test = X[train_index], X[test_index]
        # ...
        # y_train, y_test = y[train_index], y[test_index]
        # TRAIN: [1 3]
        # TEST: [0 2]
        # TRAIN: [0 2]
        # TEST: [1 3]

        # kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        # if dic_of_labels_limits is None:
        #     dic_of_labels_limits = {}
        # if dic_of_labels_limits:
        #     self.sample_labels(dic_of_labels_limits)
        inputs = [i for i in range(len(self))]
        targets = [t for t in self.targets]
        for tst_i in self.test_indexes:
            targets.insert(tst_i, self.test_set.targets[tst_i])

        if kf:
            return kf.split(inputs, np.array(targets), groups=self.grouping)

        if self.grouping:
            kf = GroupKFold(n_splits=n_splits)
            return kf.split(inputs, groups=self.grouping)

        if not stratified:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return kf.split(inputs, targets)
        elif self.task.classification_type == 'multi-label':
            kf = IterativeStratification(n_splits=n_splits)
            targets = np.array(targets)
            return kf.split(inputs, targets)
        else:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            targets = LabelEncoder().fit_transform([''.join(str(l)) for l in targets])
            return kf.split(inputs, targets)

    def get_split_by_index(self, train_index, test_index):
        """
        Returns the training and test set from the given lists of indexes

        :param train_index: indexes of the training set
        :param test_index: indexes of the test set
        :return: the taskdataset for the training and test data
        """
        self.return_data()

        self.test_indexes = list(test_index)

        self.distribute_data()

    def return_data(self):
        """
        Returns the test set data to the main object
        """
        for i in reversed(range(len(self.test_indexes))):
            self.inputs.insert(self.test_indexes[i], self.test_set.inputs[i])
            self.targets.insert(self.test_indexes[i], self.test_set.targets[i])
            if self.grouping:
                self.grouping.insert(self.test_indexes[i], self.test_set.grouping[i])
            for t in self.extra_tasks:
                t[1].insert(self.test_indexes[i], self.test_set.extra_tasks[1][i])
        self.test_set.inputs = []
        self.test_set.targets = []
        self.test_set.grouping = []
        self.test_set.extra_tasks = []
        self.test_indexes = []

    def distribute_data(self):
        """
        Distributes the data over the test set according to the saved test_indexes
        """
        self.test_indexes.sort(reverse=True)
        for t_i in self.test_indexes:
            self.test_set.inputs.append(self.inputs.pop(t_i))
        self.test_set.add_task_and_targets(self.task, [self.targets.pop(i) for i in self.test_indexes])
        if self.grouping:
            self.test_set.add_grouping([self.grouping.pop(i) for i in self.test_indexes])
        for t in self.extra_tasks:
            self.test_set.add_task_and_targets(t[0], [t[1].pop(i) for i in self.test_indexes])
        self.test_set.copy_non_data_variables(self)

    def generate_train_test_set(self, random_state: int = None, n_splits=5, kf: BaseCrossValidator = None):
        """
        Generates and returns the train and test split
        :param random_state:
        :param n_splits:
        :param kf:
        :return:
        """
        if len(self.test_set):
            for i in range(n_splits):
                yield self, self.test_set
        else:
            for train, test in self.k_folds(random_state=random_state, n_splits=n_splits, kf=kf):
                self.get_split_by_index(train_index=train, test_index=test)
                yield self, self.test_set

    def disconnect_test(self):
        """
        Disconnects the test set and resets the train_test generator function
        :return: TaskDataset The test set
        """
        assert len(
            self.test_set) > 0, 'There must be a test set present. Use the generate_train_test_set function to create one'
        test = self.test_set
        self.__init_test__(test.base_path)

    ########################################################################################################
    # Filtering
    ########################################################################################################
    # def sample_labels(self, dic_of_labels_limits, random_state=None):
    #     """
    #
    #     :param dic_of_labels_limits:
    #     :param random_state:
    #     :return:
    #     """
    #     super().sample_labels(dic_of_labels_limits=dic_of_labels_limits,
    #                           random_state=random_state)
    #     if len(self.test_set):
    #         self.test_set.sample_labels(dic_of_labels_limits=dic_of_labels_limits,
    #                                     random_state=random_state)

    def remove_label_instances(self, index):
        """
        Remove all singualre instances of a certain label and its place in one hot encoding
        :param index: the index of the label to remove
        """
        super().remove_label_instances(index)
        if len(self.test_set):
            self.test_set.remove_label_instances(index)

    ########################################################################################################
    # Transformation
    ########################################################################################################

    def normalize_inputs(self):
        super().normalize_inputs()
        if len(self.test_set):
            self.test_set.normalize_inputs()

    def inverse_normalize_inputs(self):
        super().inverse_normalize_inputs()
        if len(self.test_set):
            self.test_set.inverse_normalize_inputs()

    def prepare_inputs(self):
        """
        Applies preparation of the input instances defined in the extraction method
        on each input instance in the TaskDataset.
        """
        super().prepare_inputs()
        if len(self.test_set):
            self.test_set.prepare_inputs()

    ########################################################################################################
    # I/O
    ########################################################################################################
    def save(self):
        super().save()
        if self.test_set.base_path != self.base_path:
            self.test_set.save()

    def load(self, taskname):
        super().load(taskname)
        if self.test_set.base_path != self.base_path:
            self.test_set.load(taskname)
            self.test_set.task = self.task
            self.test_set.extraction_method = self.extraction_method

    def check(self, taskname):
        """
        Checks if there are stored taskdatsets for this extraction method
        :return: Whether or not there is stored data
        """
        if self.test_set.base_path != self.base_path:
            return self.test_set.check(taskname) \
                   and super().check(taskname)
        else:
            return super().check(taskname)
