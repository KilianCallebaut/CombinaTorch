import torch
from torch import nn
import torch.nn.functional as F


class GeorgievMultiDNN(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            output_sizes
    ):
        super().__init__()
        self.name = 'dnn'
        self.output_sizes = output_sizes
        h_sizes = [input_size] + [hidden_size for _ in range(n_hidden)]

        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(
                nn.Linear(
                    h_sizes[k],
                    h_sizes[k + 1]
                )
            )
        for k in range(len(self.hidden)):
            torch.nn.init.xavier_uniform_(self.hidden[k].weight)
            nn.init.constant_(self.hidden[k].bias, 0)

        self.task_nets = nn.ModuleList()
        for t in output_sizes:
            self.task_nets.append(
                nn.Linear(
                    hidden_size,
                    t
                )
            )

        for t in range(len(self.task_nets)):
            torch.nn.init.xavier_uniform_(self.task_nets[t].weight)
            nn.init.constant_(self.task_nets[t].bias, 0)



    def forward(self, x):
        x = x.flatten(1)
        for layer in self.hidden[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.hidden[-1](x)
        return tuple(F.softmax(self.task_nets[task_model_id](x), dim=-1)
                     for task_model_id in range(len(self.output_sizes)))
