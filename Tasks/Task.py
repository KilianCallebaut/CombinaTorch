import types
from abc import abstractmethod, ABC

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Task(ABC):

    def __init__(
            self,
            name: str,
            output_labels,
            loss_function=None,
            task_group=0
    ):
        super().__init__()
        self.output_labels = output_labels
        self.name = name
        self.classification_type = ''
        self.loss_function = loss_function
        self.task_group = task_group

    def __eq__(self, other):
        if isinstance(other, Task):
            return all(self.output_labels[i] == other.output_labels[i] for i in range(len(self.output_labels))) \
                   and self.name == other.name and \
                   self.classification_type == other.classification_type and \
                   type(self.loss_function) == type(other.loss_function)
        return False

    @abstractmethod
    def decision_making(self, output) -> torch.tensor:
        # if self.classification_type == 'multi-class':
        #     return default_max_multi_class(output)
        # elif self.classification_type == 'multi-label':
        #     return default_max_multi_label(output)
        pass

    @abstractmethod
    def translate_labels(self, output) -> torch.tensor:
        # if self.classification_type == 'multi-class':
        #     return torch.max(output, 1)[1]
        # else:
        #     return output
        pass

    def set_task_group(self, group: int):
        self.task_group = group


class MultiClassTask(Task):

    def __init__(self,
                 name,
                 output_labels,
                 loss_function=None):
        super().__init__(name, output_labels, loss_function)
        self.classification_type = 'multi-class'
        if not loss_function:
            self.loss_function = nn.NLLLoss().to(device)
        else:
            self.loss_function = loss_function

    def decision_making(self, output):
        if len(output) == 0:
            return output
        return torch.max(output, 1)[1]

    def translate_labels(self, output):
        if len(output) == 0:
            return output
        return torch.max(output, 1)[1]


class MultiLabelTask(Task):

    def __init__(self,
                 name,
                 output_labels,
                 loss_function=None):
        super().__init__(name, output_labels, loss_function)
        self.classification_type = 'multi-label'
        if not loss_function:
            self.loss_function = nn.BCELoss().to(device)
        else:
            self.loss_function = loss_function

    def decision_making(self, output):
        if len(output) == 0:
            return output
        return (output >= 0.5).float()

    def translate_labels(self, output):
        if len(output) == 0:
            return output
        return output.float()
